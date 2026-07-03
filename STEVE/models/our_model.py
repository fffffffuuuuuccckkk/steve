import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fpem import AGCRNEncoder, ConvexGatedFusion, EnvConditionedInvariantHeads, EnvMask, EnvRouteHeads
from models.fpem.losses import (
    future_mi_loss,
    gain_weighted_swap_loss,
    route_losses,
    tensor_float,
    weighted_flow_mae,
)


class StableST(nn.Module):
    """FPEM core with AGCRN-only backbone."""

    def __init__(
            self,
            args,
            adj,
            in_channels=1,
            embed_size=64,
            T_dim=12,
            output_T_dim=1,
            output_dim=2,
            device="cuda",
    ):
        super().__init__()
        self.args = args
        self.adj = adj
        self.embed_size = embed_size
        self.output_dim = output_dim
        self.latest_fpem_logs = {}
        self.latest_fpem_outputs = {}
        self._fpem_gc_primary_grads = {}

        if str(getattr(args, "fpem_backbone", "agcrn")).lower() != "agcrn":
            raise ValueError("FPEM rewrite keeps AGCRN only; set --fpem_backbone agcrn")

        agcrn_embed_dim = int(getattr(args, "agcrn_embed_dim", 10))
        agcrn_layers = int(getattr(args, "agcrn_num_layers", 2))
        agcrn_cheb_k = int(getattr(args, "agcrn_cheb_k", 2))
        self.encoder_inv = AGCRNEncoder(
            args.num_nodes, args.d_input, embed_size, agcrn_cheb_k, agcrn_embed_dim, agcrn_layers
        )
        self.encoder_env = AGCRNEncoder(
            args.num_nodes, args.d_input, embed_size, agcrn_cheb_k, agcrn_embed_dim, agcrn_layers
        )

        self.inv_trunk = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Dropout(args.dropout),
        )
        self.inv_pred = nn.Linear(embed_size, output_dim)

        self.fpem_use_env_mask = self._as_bool(getattr(args, "fpem_use_env_mask", False))
        self.fpem_use_env_route = self._as_bool(getattr(args, "fpem_use_env_route", False))
        self.fpem_env_route_k = int(getattr(args, "fpem_env_route_k", 3))
        self.fpem_env_route_head_mode = str(getattr(args, "fpem_env_route_head_mode", "concat_input")).lower()
        self.fpem_env_route_use_inv_fallback_expert = self._as_bool(
            getattr(args, "fpem_env_route_use_inv_fallback_expert", True)
        )
        self.fpem_env_route_train_mode = str(getattr(args, "fpem_env_route_train_mode", "soft_oracle")).lower()
        self.fpem_use_env_fusion = self._as_bool(getattr(args, "fpem_use_env_fusion", True))
        self.fpem_use_swap = self._as_bool(getattr(args, "fpem_use_swap", False))
        self.fpem_use_future_mi = self._as_bool(getattr(args, "fpem_use_future_mi", False))
        self.fpem_use_grad_consensus = self._as_bool(getattr(args, "fpem_use_grad_consensus", False))
        self.fpem_gc_pred_loss_only = self._as_bool(getattr(args, "fpem_gc_pred_loss_only", True))

        mask_hidden = int(getattr(args, "fpem_env_mask_hidden_dim", embed_size))
        self.env_mask = EnvMask(embed_size, mask_hidden, args.dropout)

        route_hidden = int(getattr(args, "fpem_env_route_hidden_dim", embed_size))
        self.route_heads = EnvRouteHeads(embed_size, output_dim, self.fpem_env_route_k, route_hidden, args.dropout)
        self.fusion = ConvexGatedFusion(embed_size, self.fpem_env_route_k, args.dropout)
        self.hyper_inv_heads = EnvConditionedInvariantHeads(
            embed_size,
            self.fpem_env_route_k,
            int(getattr(args, "fpem_hyper_hidden_dim", route_hidden)),
            args.dropout,
            alpha_mode=str(getattr(args, "fpem_hyper_alpha_mode", "sample_gate")),
        )
        hyper_router_dim = self.fpem_env_route_k + (1 if self.fpem_env_route_use_inv_fallback_expert else 0)
        self.hyper_router = nn.Sequential(
            nn.Linear(embed_size * 2, route_hidden),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(route_hidden, hyper_router_dim),
        )

        if self.fpem_use_future_mi:
            future_hidden = int(getattr(args, "fpem_future_mi_hidden_dim", embed_size))
            self.future_env_encoder = nn.Sequential(
                nn.Linear(output_dim, future_hidden),
                nn.ReLU(),
                nn.Linear(future_hidden, embed_size),
            )
            self.future_env_mu = nn.Linear(embed_size, embed_size)
            self.future_env_logvar = nn.Linear(embed_size, embed_size)
        else:
            self.future_env_encoder = None
            self.future_env_mu = None
            self.future_env_logvar = None

        self.reset_parameters()

    @staticmethod
    def _as_bool(value):
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def _epoch_index(self, p):
        if p is None:
            return 0
        value = float(p)
        if value <= 1.0:
            return int(round(value * float(getattr(self.args, "epochs", 1))))
        return int(round(value))

    def forward_features(self, x):
        z_inv_seq = self.encoder_inv(x)
        z_env_seq = self.encoder_env(x)
        return z_inv_seq, z_env_seq

    def forward(
        self,
        x,
        adj=None,
        target=None,
        c=None,
        time_label=None,
        scaler=None,
        loss_weights=None,
        p=None,
        training_loss=False,
        batch_idx=None,
        return_loss=False,
        return_output=False,
    ):
        z_inv_seq, z_env_seq = self.forward_features(x)
        if return_loss:
            return self.calculate_loss(
                z_inv_seq, z_env_seq, target, c, time_label, scaler, loss_weights, p, bool(training_loss)
            )
        if return_output:
            return self.forward_output_from_features(
                z_inv_seq, z_env_seq, exog=c, training=bool(training_loss), epoch=self._epoch_index(p)
            )
        return z_inv_seq, z_env_seq

    def _mask_enabled_for_epoch(self, training, epoch):
        if not self.fpem_use_env_mask:
            return False
        warmup = int(getattr(self.args, "fpem_env_mask_warmup_epochs", 0))
        return not (training and epoch is not None and int(epoch) < warmup)

    def _route_enabled_for_epoch(self, training, epoch):
        if not self.fpem_use_env_route:
            return False
        warmup = int(getattr(self.args, "fpem_env_route_warmup_epochs", 0))
        return not (training and epoch is not None and int(epoch) < warmup)

    def _use_hyper_inv_film(self):
        return self.fpem_env_route_head_mode == "hyper_inv_film"

    def _hyper_router_logits(self, z_inv, e_useful):
        ctx = torch.cat([z_inv.mean(dim=1), e_useful.mean(dim=1)], dim=-1)
        return self.hyper_router(ctx)

    def _predict_from_nodes(self, z_inv, e_useful, training=False, epoch=None):
        h_inv = self.inv_trunk(z_inv)
        y_inv = self.inv_pred(h_inv).unsqueeze(1)
        output = {
            "Z_inv": z_inv,
            "E_useful": e_useful,
            "h_inv": h_inv,
            "y_inv": y_inv,
            "y_global": y_inv,
        }

        route_active = self._route_enabled_for_epoch(training, epoch)
        hyper_mode = route_active and self._use_hyper_inv_film()
        fusion_active = route_active and self.fpem_use_env_fusion and not hyper_mode
        fusion_logs = self.fusion.zero_logs(y_inv)
        if hyper_mode:
            hyper_out = self.hyper_inv_heads(z_inv, e_useful, h_inv, self.inv_pred)
            y_hyper_heads = hyper_out["y_hyper_heads"]
            logits = self._hyper_router_logits(z_inv, e_useful)
            q = torch.softmax(logits / max(float(getattr(self.args, "fpem_env_route_tau", 1.0)), 1e-6), dim=-1)
            if self.fpem_env_route_use_inv_fallback_expert:
                y_candidates = torch.cat([y_inv.unsqueeze(1), y_hyper_heads], dim=1)
                q_env = q[:, 1:]
                fallback_q = q[:, 0]
            else:
                y_candidates = y_hyper_heads
                q_env = q
                fallback_q = q.new_zeros(q.shape[0])
            y_final = (q.view(q.shape[0], q.shape[1], 1, 1, 1) * y_candidates).sum(dim=1)
            q_env_sum = q_env.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            y_route = ((q_env / q_env_sum).view(q.shape[0], self.fpem_env_route_k, 1, 1, 1) * y_hyper_heads).sum(dim=1)
            alpha = fallback_q.view(-1, 1, 1, 1)
            q_prob = q.float().clamp(1e-8, 1.0)
            output.update({
                "y_route_heads": y_hyper_heads,
                "y_hyper_heads": y_hyper_heads,
                "y_candidates": y_candidates,
                "route_q": q,
                "env_route_q": q,
                "env_route_logits": logits,
                "env_route_entropy_per_sample": (-(q_prob * q_prob.log()).sum(dim=-1)).to(dtype=q.dtype),
                "env_route_q_max": q.max(dim=-1).values,
                "env_route_selected_head": q.argmax(dim=-1),
                "fallback_q": fallback_q,
                "y_route": y_route,
                "hyper_alpha": hyper_out["hyper_alpha"],
                "hyper_delta_norm": hyper_out["hyper_delta_norm"],
                "route_head_mode": "hyper_inv_film",
            })
        elif route_active:
            route_out = self.route_heads(
                z_inv,
                e_useful,
                tau=float(getattr(self.args, "fpem_env_route_tau", 1.0)),
            )
            output.update(route_out)
            y_route = route_out["y_route"]
            output["route_q"] = route_out["env_route_q"]
            output["fallback_q"] = y_inv.new_zeros(y_inv.shape[0])
            output["route_head_mode"] = "concat_input"
            output["hyper_delta_norm"] = y_inv.new_zeros(())
            output["hyper_alpha"] = y_inv.new_zeros(y_inv.shape[0], self.fpem_env_route_k)
            if fusion_active:
                y_final, alpha, fusion_logs = self.fusion(y_inv, y_route, z_inv, e_useful, route_out["env_route_q"])
            else:
                alpha = y_inv.new_zeros(y_inv.shape[0], 1, 1, 1)
                y_final = y_route
        else:
            y_route = y_inv
            alpha = y_inv.new_zeros(y_inv.shape[0], 1, 1, 1)
            y_final = y_inv
            output.update({
                "y_route": y_route,
                "y_route_heads": y_route.unsqueeze(1).expand(-1, self.fpem_env_route_k, -1, -1, -1),
                "y_hyper_heads": y_route.unsqueeze(1).expand(-1, self.fpem_env_route_k, -1, -1, -1),
                "env_route_q": y_inv.new_full((y_inv.shape[0], self.fpem_env_route_k), 1.0 / self.fpem_env_route_k),
                "route_q": y_inv.new_full((y_inv.shape[0], self.fpem_env_route_k), 1.0 / self.fpem_env_route_k),
                "env_route_logits": y_inv.new_zeros(y_inv.shape[0], self.fpem_env_route_k),
                "env_route_selected_head": y_inv.new_zeros(y_inv.shape[0], dtype=torch.long),
                "fallback_q": y_inv.new_zeros(y_inv.shape[0]),
                "hyper_delta_norm": y_inv.new_zeros(()),
                "hyper_alpha": y_inv.new_zeros(y_inv.shape[0], self.fpem_env_route_k),
                "route_head_mode": self.fpem_env_route_head_mode,
            })

        output.update({
            "prediction": y_final,
            "y_final": y_final,
            "y_route": y_route,
            "fusion_alpha": alpha,
            "primary_uses_route": y_inv.new_tensor(float(route_active)),
            "primary_uses_env_fusion": y_inv.new_tensor(float(fusion_active)),
            "fusion_logs": fusion_logs,
        })
        return output

    def forward_output(self, x, exog=None, training=False, epoch=None):
        z_inv_seq, z_env_seq = self.forward_features(x)
        return self.forward_output_from_features(z_inv_seq, z_env_seq, exog=exog, training=training, epoch=epoch)

    def forward_output_from_features(self, z_inv_seq, z_env_seq, exog=None, training=False, epoch=None):
        z_inv = z_inv_seq[:, -1, :, :]
        e_raw = z_env_seq[:, -1, :, :]
        use_mask = self._mask_enabled_for_epoch(training, epoch)
        e_useful, e_discard, env_mask, mask_loss, mask_logs = self.env_mask(
            e_raw,
            use_mask=use_mask,
            lambda_sparse=float(getattr(self.args, "fpem_lambda_mask_sparse", 0.0)),
            lambda_entropy=float(getattr(self.args, "fpem_lambda_mask_entropy", 0.0)),
            temperature=float(getattr(self.args, "fpem_env_mask_temperature", 1.0)),
        )
        output = self._predict_from_nodes(z_inv, e_useful, training=training, epoch=epoch)
        output.update({
            "Z_inv_seq": z_inv_seq,
            "Z_env_seq": z_env_seq,
            "E_raw": e_raw,
            "E_discard": e_discard,
            "env_mask": env_mask,
            "mask_loss": mask_loss,
            "mask_logs": mask_logs,
        })
        return output

    def predict_test(self, Z_tensor, H_tensor, exog=None):
        output = self.forward_output_from_features(Z_tensor, H_tensor, exog=exog, training=False, epoch=None)
        att = output["prediction"].new_zeros(1)
        return output["prediction"], att, output["E_useful"], output["Z_inv"].unsqueeze(1)

    def _permute_env(self, e_useful, route_q=None):
        b = e_useful.shape[0]
        device = e_useful.device
        perm = torch.arange(b, device=device)
        valid = torch.ones(b, dtype=torch.bool, device=device)
        if b <= 1:
            valid.zero_()
            return e_useful, perm, valid
        only_diff_route = self._as_bool(getattr(self.args, "fpem_swap_only_diff_route", True))
        if only_diff_route and route_q is not None and route_q.shape[0] == b:
            route = route_q.detach().argmax(dim=-1)
            if torch.unique(route).numel() > 1:
                for i in range(b):
                    candidates = torch.nonzero(route != route[i], as_tuple=False).flatten()
                    if candidates.numel() == 0:
                        valid[i] = False
                    else:
                        perm[i] = candidates[torch.randint(candidates.numel(), (1,), device=device)]
                return e_useful[perm], perm, valid
        perm = torch.randperm(b, device=device)
        same = perm == torch.arange(b, device=device)
        if bool(same.any()):
            perm[same] = (perm[same] + 1) % b
        return e_useful[perm], perm, valid

    def _swap_terms(self, output, target, scaler, training, epoch):
        zero = output["prediction"].new_zeros(())
        logs = {
            "fpem/swap_loss": zero,
            "fpem/swap_diff_loss": zero,
            "fpem/swap_same_loss": zero,
            "fpem/swap_gain_mean": zero,
            "fpem/swap_s_gain_mean": zero,
            "fpem/swap_fallback_router_loss": zero,
            "fpem/swap_fallback_q_mean": zero,
        }
        if not (self.fpem_use_swap and training):
            return zero, logs
        warmup = int(getattr(self.args, "fpem_swap_warmup_epochs", 0))
        if epoch is not None and int(epoch) < warmup:
            return zero, logs
        e_perm, _perm, valid = self._permute_env(output["E_useful"], output.get("env_route_q"))
        if self._as_bool(getattr(self.args, "fpem_swap_detach_env", False)):
            e_perm = e_perm.detach()
        z_inv = output["Z_inv"].detach() if self._as_bool(getattr(self.args, "fpem_swap_detach_inv", True)) else output["Z_inv"]
        swap_output = self._predict_from_nodes(z_inv, e_perm, training=training, epoch=epoch)
        swap_loss, logs = gain_weighted_swap_loss(
            output["prediction"],
            output["y_inv"],
            swap_output["prediction"],
            target,
            scaler,
            self.args,
            valid_sample=valid,
        )
        logs.setdefault("fpem/swap_fallback_router_loss", zero)
        logs.setdefault("fpem/swap_fallback_q_mean", zero)
        use_fallback_router = (
            self._use_hyper_inv_film()
            and self.fpem_env_route_use_inv_fallback_expert
            and self._as_bool(getattr(self.args, "fpem_use_swap_fallback_router_loss", False))
        )
        fallback_warmup = int(getattr(self.args, "fpem_swap_fallback_warmup_epochs", 10))
        if use_fallback_router and epoch is not None and int(epoch) >= fallback_warmup:
            router_logits = self._hyper_router_logits(output["Z_inv"].detach(), e_perm.detach())
            target_fallback = torch.zeros(router_logits.shape[0], dtype=torch.long, device=router_logits.device)
            fallback_loss = F.cross_entropy(router_logits.float(), target_fallback)
            fallback_q = torch.softmax(router_logits.detach(), dim=-1)[:, 0]
            swap_loss = swap_loss + float(getattr(self.args, "fpem_lambda_swap_fallback_router", 0.005)) * fallback_loss
            logs["fpem/swap_fallback_router_loss"] = fallback_loss.detach()
            logs["fpem/swap_fallback_q_mean"] = fallback_q.mean()
        return swap_loss, logs

    def calculate_loss(self, Z_tensor, H_tensor, target, c, time_label, scaler, loss_weights, p=None, training=False):
        epoch = self._epoch_index(p)
        output = self.forward_output_from_features(Z_tensor, H_tensor, exog=c, training=training, epoch=epoch)
        primary_loss = weighted_flow_mae(output["prediction"], target, scaler, getattr(self.args, "yita", 0.5))
        inv_pred_loss = weighted_flow_mae(output["y_inv"], target, scaler, getattr(self.args, "yita", 0.5))
        primary_weight = float(loss_weights[0]) if loss_weights is not None else 1.0
        loss = primary_weight * primary_loss + output["mask_loss"]
        if training:
            inv_lambda = float(getattr(
                self.args,
                "fpem_lambda_inv_pred",
                getattr(self.args, "fpem_env_route_lambda_global", 0.2),
            ))
            loss = loss + inv_lambda * inv_pred_loss
            if self._use_hyper_inv_film():
                loss = loss + float(getattr(self.args, "fpem_lambda_hyper_delta_norm", 0.0)) * output["hyper_delta_norm"]

        route_loss = primary_loss.new_zeros(())
        route_logs = self._zero_route_logs(primary_loss)
        q_oracle = None
        if self.fpem_use_env_route:
            route_loss, route_logs, q_oracle = route_losses(output, target, scaler, self.args)
            if training:
                loss = loss + route_loss

        future_loss, future_logs = future_mi_loss(
            output["E_useful"],
            output["y_inv"],
            target,
            self.future_env_encoder,
            self.future_env_mu,
            self.future_env_logvar,
            self.args,
            training and self.fpem_use_future_mi,
            epoch,
        )
        if training:
            loss = loss + float(getattr(self.args, "fpem_lambda_future_mi", 0.0)) * future_loss
            swap_loss, swap_logs = self._swap_terms(output, target, scaler, training, epoch)
            loss = loss + swap_loss
        else:
            swap_loss = primary_loss.new_zeros(())
            swap_logs = self._zero_swap_logs(primary_loss)

        logs = {}
        logs.update(output["mask_logs"])
        logs.update(output["fusion_logs"])
        logs.update(route_logs)
        logs.update(future_logs)
        logs.update(swap_logs)
        logs.update({
            "fpem/primary_loss": primary_loss.detach(),
            "fpem/inv_pred_loss": inv_pred_loss.detach(),
            "fpem/primary_uses_route": output["primary_uses_route"].detach(),
            "fpem/primary_uses_env_fusion": output["primary_uses_env_fusion"].detach(),
            "fpem/backbone_agcrn": primary_loss.new_tensor(1.0),
            "fpem/gc_pred_loss_only": primary_loss.new_tensor(float(self.fpem_gc_pred_loss_only)),
            "fpem/env_route_head_mode": primary_loss.new_tensor(1.0 if self._use_hyper_inv_film() else 0.0),
        })
        self.latest_fpem_logs = {key: tensor_float(value) for key, value in logs.items()}

        self.latest_fpem_outputs = {
            "primary_loss": primary_loss,
            "prediction": output["prediction"],
            "y_inv": output["y_inv"],
            "y_route": output["y_route"],
            "env_route_q": output["env_route_q"],
            "env_route_q_oracle": q_oracle,
            "gc_inv_tensor": output["Z_inv"],
            "gc_env_tensor": output["E_useful"],
            "gc_env_route_q": output["env_route_q"].detach(),
        }

        if not training and str(getattr(self.args, "lr_mode", "only")).lower() == "only":
            loss = primary_loss

        route_sep = route_loss.detach()
        aux_sep = (future_loss.detach() + swap_loss.detach())
        sep_loss = [
            tensor_float(primary_loss.detach()),
            tensor_float(route_sep if tensor_float(route_sep) > 0.0 else primary_loss.detach().new_tensor(1.0)),
            tensor_float(aux_sep if tensor_float(aux_sep) > 0.0 else primary_loss.detach().new_tensor(1.0)),
        ]
        return loss, sep_loss, 0

    def _zero_route_logs(self, ref):
        logs = {
            "fpem/env_route_loss": ref.new_zeros(()),
            "fpem/env_route_L_final": ref.new_zeros(()),
            "fpem/env_route_L_global": ref.new_zeros(()),
            "fpem/env_route_L_route_soft": ref.new_zeros(()),
            "fpem/env_route_L_expert": ref.new_zeros(()),
            "fpem/env_route_L_router_oracle": ref.new_zeros(()),
            "fpem/env_route_L_balance": ref.new_zeros(()),
            "fpem/env_route_L_diverse": ref.new_zeros(()),
            "fpem/env_route_entropy": ref.new_zeros(()),
            "fpem/env_route_q_max_mean": ref.new_zeros(()),
            "fpem/hyper_alpha_mean": ref.new_zeros(()),
            "fpem/hyper_delta_norm": ref.new_zeros(()),
            "fpem/fallback_q_mean": ref.new_zeros(()),
            "fpem/fallback_q_max": ref.new_zeros(()),
            "fpem/env_q_sum_mean": ref.new_zeros(()),
            "fpem/oracle_fallback_rate": ref.new_zeros(()),
            "fpem/route_count_fallback": ref.new_zeros(()),
            "fpem/env_route_head_mode": ref.new_tensor(1.0 if self._use_hyper_inv_film() else 0.0),
        }
        for idx in range(self.fpem_env_route_k):
            logs[f"fpem/env_route_count_head_{idx}"] = ref.new_zeros(())
            logs[f"fpem/env_route_oracle_count_head_{idx}"] = ref.new_zeros(())
            logs[f"fpem/route_count_env_head_{idx}"] = ref.new_zeros(())
        return logs

    @staticmethod
    def _zero_swap_logs(ref):
        return {
            "fpem/swap_loss": ref.new_zeros(()),
            "fpem/swap_diff_loss": ref.new_zeros(()),
            "fpem/swap_same_loss": ref.new_zeros(()),
            "fpem/swap_gain_mean": ref.new_zeros(()),
            "fpem/swap_s_gain_mean": ref.new_zeros(()),
            "fpem/swap_fallback_router_loss": ref.new_zeros(()),
            "fpem/swap_fallback_q_mean": ref.new_zeros(()),
        }

    def _gc_soft_mask(self, agree):
        tau = float(getattr(self.args, "fpem_gc_tau", 0.5))
        temp = max(float(getattr(self.args, "fpem_gc_temp", 0.1)), 1e-6)
        min_keep = min(max(float(getattr(self.args, "fpem_gc_min_keep", 0.2)), 0.0), 1.0)
        mask = torch.sigmoid((agree - tau) / temp)
        return min_keep + (1.0 - min_keep) * mask

    def prepare_fpem_gc_pred_loss_only(self, primary_loss):
        self._fpem_gc_primary_grads = {}
        if not (self.fpem_use_grad_consensus and self.fpem_gc_pred_loss_only and torch.is_tensor(primary_loss)):
            return False
        outputs = self.latest_fpem_outputs
        names, tensors = [], []
        for name, key in (("inv", "gc_inv_tensor"), ("env", "gc_env_tensor")):
            tensor = outputs.get(key)
            if torch.is_tensor(tensor) and tensor.requires_grad:
                names.append(name)
                tensors.append(tensor)
        if not tensors:
            return False
        grads = torch.autograd.grad(primary_loss, tensors, retain_graph=True, allow_unused=True)
        for name, grad in zip(names, grads):
            if grad is not None:
                self._fpem_gc_primary_grads[name] = grad.detach()
        return bool(self._fpem_gc_primary_grads)

    def clear_fpem_gc_pred_loss_only(self):
        self._fpem_gc_primary_grads = {}

    def register_fpem_grad_consensus_hooks(self, epoch=None):
        if not (self.fpem_use_grad_consensus and self.training):
            return []
        warmup = int(getattr(self.args, "fpem_gc_warmup_epochs", 0))
        if epoch is not None and int(epoch) < warmup:
            return []

        handles = []
        rho_inv = min(max(float(getattr(self.args, "fpem_gc_inv_rho", 0.3)), 0.0), 1.0)
        rho_env = min(max(float(getattr(self.args, "fpem_gc_env_rho", 0.3)), 0.0), 1.0)
        inv_tensor = self.latest_fpem_outputs.get("gc_inv_tensor")
        env_tensor = self.latest_fpem_outputs.get("gc_env_tensor")
        route_q = self.latest_fpem_outputs.get("gc_env_route_q")

        if torch.is_tensor(inv_tensor) and inv_tensor.requires_grad and rho_inv > 0.0:
            def inv_hook(grad):
                primary = self._fpem_gc_primary_grads.get("inv") if self.fpem_gc_pred_loss_only else None
                if self.fpem_gc_pred_loss_only and primary is None:
                    return grad
                base = primary.to(device=grad.device, dtype=grad.dtype) if primary is not None else grad
                agree = base.detach().float().sign().mean(dim=0).abs()
                mask = self._gc_soft_mask(agree).to(device=grad.device, dtype=grad.dtype)
                new_base = (1.0 - rho_inv) * base + rho_inv * (base * mask.unsqueeze(0))
                self.latest_fpem_logs.update({
                    "fpem/gc_inv_agree_mean": tensor_float(agree.mean()),
                    "fpem/gc_inv_mask_mean": tensor_float(mask.mean()),
                })
                return grad + (new_base - base) if primary is not None else new_base
            handles.append(inv_tensor.register_hook(inv_hook))

        if (
            torch.is_tensor(env_tensor)
            and env_tensor.requires_grad
            and torch.is_tensor(route_q)
            and route_q.shape[0] == env_tensor.shape[0]
            and rho_env > 0.0
        ):
            route_id = route_q.detach().argmax(dim=-1).to(device=env_tensor.device)
            min_samples = max(int(getattr(self.args, "fpem_gc_route_min_samples", 2)), 1)

            def env_hook(grad):
                primary = self._fpem_gc_primary_grads.get("env") if self.fpem_gc_pred_loss_only else None
                if self.fpem_gc_pred_loss_only and primary is None:
                    return grad
                base = primary.to(device=grad.device, dtype=grad.dtype) if primary is not None else grad
                new_base = base.clone()
                agree_means, mask_means = [], []
                valid_samples = 0
                for route in torch.unique(route_id):
                    idx = torch.nonzero(route_id == route, as_tuple=False).flatten()
                    if idx.numel() < min_samples:
                        continue
                    g_route = base.detach().float().index_select(0, idx)
                    agree = g_route.sign().mean(dim=0).abs()
                    mask = self._gc_soft_mask(agree).to(device=grad.device, dtype=grad.dtype)
                    new_base[idx] = (1.0 - rho_env) * base[idx] + rho_env * (base[idx] * mask.unsqueeze(0))
                    agree_means.append(agree.mean())
                    mask_means.append(mask.mean())
                    valid_samples += int(idx.numel())
                if agree_means:
                    self.latest_fpem_logs.update({
                        "fpem/gc_env_agree_mean": tensor_float(torch.stack(agree_means).mean()),
                        "fpem/gc_env_mask_mean": tensor_float(torch.stack(mask_means).mean()),
                        "fpem/gc_env_valid_sample_ratio": float(valid_samples) / max(float(grad.shape[0]), 1.0),
                    })
                return grad + (new_base - base) if primary is not None else new_base
            handles.append(env_tensor.register_hook(env_hook))
        return handles


STEVE = StableST
