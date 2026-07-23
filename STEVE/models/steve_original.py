import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import masked_mae_loss
from models.module import ST_encoder, CLUB
from models.layers import RevGradLayer, pca_whitening, MLPAttention


class OriginalStableST(nn.Module):
    """Original STEVE/STGCN implementation with a prediction-only switch.

    ``steve_prediction_mode='full'`` keeps the original STEVE prediction path:

        Y = relu(C @ W) * Y_c + Y_h

    ``steve_prediction_mode='inv_only'`` trains from scratch while using only
    the original invariant representation prediction head:

        Y = Y_h

    The switch is deliberately named without an ``fpem`` prefix and this class
    does not import or instantiate any FPEM modules.
    """

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
        super(OriginalStableST, self).__init__()

        self.args = args
        self.adj = adj
        self.time_labels = 48
        self.mi_w = args.mi_w
        self.embed_size = embed_size
        self.output_dim = output_dim
        self.steve_prediction_mode = str(
            getattr(args, "steve_prediction_mode", "full")
        ).strip().lower()
        if self.steve_prediction_mode in {"inv_only_with_disentangle", "inv_only_disentangle"}:
            self.steve_prediction_mode = "inv_only"
        if self.steve_prediction_mode in {"no_env_inv_only", "inv_only_single_stream"}:
            self.steve_prediction_mode = "inv_only_no_env"
        if self.steve_prediction_mode not in {"full", "inv_only", "inv_only_no_env"}:
            raise ValueError(
                "steve_prediction_mode must be one of: full, inv_only, "
                "inv_only_with_disentangle, inv_only_no_env"
            )
        self.steve_disable_env_stream = self.steve_prediction_mode == "inv_only_no_env"

        T_dim = args.input_length - 4 * (3 - 1)
        self.K = int(args.d_model * args.kw)

        temp_spatial_label = list(range(args.num_nodes))
        self.spatial_label = torch.tensor(temp_spatial_label, device=args.device)

        # Original STEVE STGCN encoders.
        self.st_encoder4variant = ST_encoder(
            args.num_nodes,
            args.d_input,
            args.d_model,
            3,
            3,
            [
                [args.d_model, args.d_model // 2, args.d_model],
                [args.d_model, args.d_model // 2, args.d_model],
            ],
            args.input_length,
            args.dropout,
            args.device,
        )
        self.st_encoder4invariant = ST_encoder(
            args.num_nodes,
            args.d_input,
            args.d_model,
            3,
            3,
            [
                [args.d_model, args.d_model // 2, args.d_model],
                [args.d_model, args.d_model // 2, args.d_model],
            ],
            args.input_length,
            args.dropout,
            args.device,
        )

        # Original dynamic adjacency for the variant/environment encoder.
        self.node_embeddings_1 = nn.Parameter(
            torch.randn(3, args.num_nodes, embed_size), requires_grad=True
        )
        self.node_embeddings_2 = nn.Parameter(
            torch.randn(3, embed_size, args.num_nodes), requires_grad=True
        )

        # Original prediction heads.
        self.tcl4c = nn.Conv2d(T_dim, output_T_dim, 1, bias=True)
        self.tcl4h = nn.Conv2d(T_dim, output_T_dim, 1, bias=True)
        self.variant_predict_conv_1 = nn.Conv2d(T_dim, output_T_dim, 1)
        self.variant_predict_conv_2 = nn.Conv2d(embed_size, output_dim, 1)
        self.invariant_predict_conv_1 = nn.Conv2d(T_dim, output_T_dim, 1)
        self.invariant_predict_conv_2 = nn.Conv2d(embed_size, output_dim, 1)
        self.relu = nn.ReLU()

        # Original variant/environment auxiliary heads.
        self.variant_tconv = nn.Conv2d(
            in_channels=T_dim, out_channels=1, kernel_size=(1, 1), bias=True
        )
        self.variant_end_temproal = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.ReLU(),
            nn.Linear(embed_size * 2, self.time_labels),
        )
        self.variant_end_spacial = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.ReLU(),
            nn.Linear(embed_size * 2, args.num_nodes),
        )
        self.variant_end_congest = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.ReLU(),
            nn.Linear(embed_size // 2, 2),
        )

        # Original invariant auxiliary heads.
        self.invariant_tconv = nn.Conv2d(
            in_channels=T_dim, out_channels=1, kernel_size=(1, 1), bias=True
        )
        self.invariant_end_temporal = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.ReLU(),
            nn.Linear(embed_size * 2, self.time_labels),
        )
        self.invariant_end_spatial = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.ReLU(),
            nn.Linear(embed_size * 2, args.num_nodes),
        )
        self.invariant_end_congest = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.ReLU(),
            nn.Linear(embed_size // 2, 2),
        )

        self.alpha_linear = nn.Linear(2, 2)
        self.beta_linear = nn.Linear(2, 2)
        self.revgrad = RevGradLayer()
        self.mask = torch.zeros(
            [args.batch_size, args.d_input, args.input_length, args.num_nodes],
            dtype=torch.float,
        ).to(device)
        self.receptive_field = args.input_length + 8
        self.mse_loss = torch.nn.MSELoss()

        self.mi_net = CLUB(embed_size, embed_size, embed_size * self.mi_w)
        self.optimizer_mi_net = torch.optim.Adam(self.mi_net.parameters(), lr=0.1)
        self.mae = masked_mae_loss(mask_value=5.0)

        self.generator_conv = nn.Conv2d(
            in_channels=args.input_length,
            out_channels=1,
            kernel_size=(1, 1),
            bias=True,
        )

        bank_temp = np.random.randn(self.K, self.embed_size)
        bank_temp = pca_whitening(bank_temp)
        self.Bank = nn.Parameter(torch.tensor(bank_temp, dtype=torch.float), requires_grad=False)
        self.mlp4bank = nn.Linear(T_dim * args.num_nodes, self.K)
        self.att4bank = MLPAttention(self.embed_size)
        self.bank_gamma = args.bank_gamma
        self.W_weight = nn.Parameter(torch.randn(embed_size, 2), requires_grad=True)

        self.mlp4C = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.ReLU(),
            nn.Linear(embed_size // 2, 2),
        )
        self.mlp4H = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.ReLU(),
            nn.Linear(embed_size // 2, 2),
        )

        self.reset_parameters()
        if self.steve_disable_env_stream:
            self._disable_environment_stream_parameters()

    def _set_requires_grad(self, module_or_param, flag):
        if isinstance(module_or_param, torch.nn.Parameter):
            module_or_param.requires_grad_(flag)
            return
        for p in module_or_param.parameters():
            p.requires_grad_(flag)

    def _disable_environment_stream_parameters(self):
        """Disable modules that belong to the environment/context stream.

        ``inv_only_no_env`` is intended to be a single-stream invariant
        baseline.  These modules are not used in forward/loss and are frozen so
        the optimizer only updates the invariant encoder and invariant
        prediction head.
        """
        env_modules = [
            self.st_encoder4variant,
            self.tcl4c,
            self.variant_predict_conv_1,
            self.variant_predict_conv_2,
            self.variant_tconv,
            self.variant_end_temproal,
            self.variant_end_spacial,
            self.variant_end_congest,
            self.mi_net,
            self.generator_conv,
            self.mlp4bank,
            self.att4bank,
            self.mlp4C,
            self.mlp4H,
            self.invariant_tconv,
            self.invariant_end_temporal,
            self.invariant_end_spatial,
            self.invariant_end_congest,
            self.alpha_linear,
            self.beta_linear,
        ]
        for module in env_modules:
            self._set_requires_grad(module, False)
        for param in [self.node_embeddings_1, self.node_embeddings_2, self.W_weight, self.Bank]:
            param.requires_grad_(False)

    def optimizer_parameters(self):
        if self.steve_disable_env_stream:
            return [p for p in self.parameters() if p.requires_grad]
        return list(self.parameters())

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, x, adj=None):
        # Original STEVE tensor convention: input [B, T, N, C] -> STGCN [B, C, T, N].
        x = x.permute(0, 3, 1, 2)
        if adj is None:
            invariant_output = self.st_encoder4invariant(x, self.adj)
        else:
            invariant_output = self.st_encoder4invariant(x, adj)
        H_tensor = invariant_output.permute(0, 2, 3, 1)

        if self.steve_disable_env_stream:
            Z_tensor = None
        else:
            adaptive_adj = F.softmax(
                F.relu(torch.bmm(self.node_embeddings_1, self.node_embeddings_2)), dim=1
            )
            variant_output = self.st_encoder4variant.variant_encode(x, adaptive_adj)
            Z_tensor = variant_output.permute(0, 2, 3, 1)
        # Canonical order for this cleaned original-STEVE wrapper:
        #   H_tensor = H_inv, invariant representation
        #   Z_tensor = Z_var, variant/environment representation
        # The upstream public code returns this same tuple but later receives it
        # with confusing local variable names.  Downstream code in this repo now
        # keeps the semantic names aligned end to end.
        return H_tensor, Z_tensor

    def _predict_invariant_only(self, H_inv_pooled):
        H_inv_pooled = H_inv_pooled.permute(0, 3, 2, 1)
        Y_h = self.invariant_predict_conv_2(H_inv_pooled)
        return Y_h.permute(0, 3, 2, 1)

    def predict(self, Z_var, C_tensor, H_inv_pooled):
        if self.steve_prediction_mode == "inv_only":
            return self._predict_invariant_only(H_inv_pooled)

        C_tensor = C_tensor.unsqueeze(1)
        out = C_tensor + self.tcl4c(Z_var)
        out = out.permute(0, 3, 2, 1)
        Y_c = self.variant_predict_conv_2(out)
        Y_c = Y_c.permute(0, 3, 2, 1)
        Y_h = self._predict_invariant_only(H_inv_pooled)
        C_weight = torch.relu(torch.matmul(C_tensor, self.W_weight))
        Y = C_weight * Y_c + Y_h
        return Y

    def predict_decomposition(self, H_inv, Z_var):
        """Inference-only decomposition of original STEVE prediction.

        Returns the exact components of the original full prediction:

            y_base      = Y_h
            y_env_delta = relu(C @ W) * Y_c
            y_full      = y_base + y_env_delta

        This helper does not change training behavior; it only exposes the
        already-existing original STEVE computation for sample-wise analysis.
        """
        H_inv_pooled = self.tcl4h(H_inv)
        if self.steve_disable_env_stream or Z_var is None:
            y_base = self._predict_invariant_only(H_inv_pooled)
            y_env_delta = torch.zeros_like(y_base)
            y_context = torch.zeros_like(y_base)
            c_weight = torch.zeros_like(y_base)
            C_tensor_unsqueezed = H_inv_pooled.new_zeros(
                H_inv_pooled.shape[0], 1, H_inv_pooled.shape[2], H_inv_pooled.shape[3]
            )
            att = H_inv_pooled.new_zeros(H_inv_pooled.shape[0], 1)
            return {
                "prediction": y_base,
                "y_base": y_base,
                "y_context": y_context,
                "y_env_delta": y_env_delta,
                "y_full": y_base,
                "c_weight": c_weight,
                "C_tensor": C_tensor_unsqueezed,
                "H_tensor": H_inv_pooled.permute(0, 3, 2, 1),
                "att": att,
            }

        C_tensor, att = self.confounder_ext(Z_var, train=False)
        C_tensor_unsqueezed = C_tensor.unsqueeze(1)
        y_base = self._predict_invariant_only(H_inv_pooled)

        out = C_tensor_unsqueezed + self.tcl4c(Z_var)
        out = out.permute(0, 3, 2, 1)
        y_context = self.variant_predict_conv_2(out)
        y_context = y_context.permute(0, 3, 2, 1)
        c_weight = torch.relu(torch.matmul(C_tensor_unsqueezed, self.W_weight))
        y_env_delta = c_weight * y_context
        y_full = y_base + y_env_delta

        if self.steve_prediction_mode == "inv_only":
            prediction = y_base
        else:
            prediction = y_full

        return {
            "prediction": prediction,
            "y_base": y_base,
            "y_context": y_context,
            "y_env_delta": y_env_delta,
            "y_full": y_full,
            "c_weight": c_weight,
            "C_tensor": C_tensor_unsqueezed,
            "H_tensor": H_inv_pooled.permute(0, 3, 2, 1),
            "att": att,
        }

    def predict_test(self, H_inv, Z_var):
        output = self.predict_decomposition(H_inv, Z_var)
        return output["prediction"], output["att"], output["C_tensor"], output["H_tensor"]

    def confounder_ext(self, Z_tensor, train=True):
        """Original bank-attention confounder/context extractor.

        Args:
            Z_tensor: [B, T, N, D] variant/environment representation.
        Returns:
            C_tensor: [B, N, D]
            att: attention weights from the original bank attention module.
        """
        b, t, n, c = Z_tensor.shape
        Z_tilda = Z_tensor.reshape(b, n * t, c)
        Z_tilda = Z_tilda.permute(0, 2, 1)
        B_tilda = self.mlp4bank(Z_tilda)
        B_tilda = B_tilda.permute(0, 2, 1)

        B_new = []
        for i in range(b):
            _B_new = self.bank_gamma * self.Bank + (1 - self.bank_gamma) * B_tilda[i]
            self.Bank.set_(_B_new.detach())
            B_new.append(_B_new)
        B_new = torch.stack(B_new)

        Q = Z_tensor.mean(1)
        C_tensor, att = self.att4bank(Q, B_new, B_new)
        if self.args.ablation == "bank":
            C_tensor = Z_tensor.mean(1)
        return C_tensor, att

    def variant_loss(self, C_tensor, date, c):
        z_temporal = C_tensor.mean(1).squeeze()
        y_temporal = self.variant_end_temproal(z_temporal)
        loss_temporal = F.cross_entropy(y_temporal, date)

        y_spatial = self.variant_end_spacial(C_tensor)
        y_spatial = y_spatial.mean(0)
        loss_spatial = F.cross_entropy(y_spatial, self.spatial_label)

        y_congest = self.variant_end_congest(C_tensor)
        loss_congest = self.mse_loss(y_congest, c)

        if self.args.ablation == "spatial":
            loss = (loss_congest + loss_temporal) / 2.0
        elif self.args.ablation == "temporal":
            loss = (loss_congest + loss_spatial) / 2.0
        elif self.args.ablation == "traffic":
            loss = (loss_temporal + loss_spatial) / 2.0
        else:
            loss = (loss_spatial + loss_temporal + loss_congest) / 3.0
        return loss

    def invariant_loss(self, H, date, c, p=None, training=True):
        z1_r = H
        if training is True and self.args.ablation != "gr":
            z1_r = self.revgrad(H, p)

        z1_r = z1_r.squeeze(1)
        z1_temporal = z1_r.mean(1).squeeze()
        y_temporal = self.invariant_end_temporal(z1_temporal)
        loss_temporal = F.cross_entropy(y_temporal, date)

        y_spatial = self.invariant_end_spatial(z1_r)
        y_spatial = y_spatial.mean(0)
        loss_spatial = F.cross_entropy(y_spatial, self.spatial_label)

        z1_congest = z1_r.unsqueeze(1)
        y_congest = self.invariant_end_congest(z1_congest)
        loss_congest = self.mse_loss(y_congest, c)

        if self.args.ablation == "spatial":
            loss = (loss_congest + loss_temporal) / 2.0
        elif self.args.ablation == "temporal":
            loss = (loss_congest + loss_spatial) / 2.0
        elif self.args.ablation == "traffic":
            loss = (loss_temporal + loss_spatial) / 2.0
        else:
            loss = (loss_spatial + loss_temporal + loss_congest) / 3.0
        return loss

    def pred_loss(self, Z_var, C_tensor, H_inv_pooled, y_true, scaler):
        y_pred = self.predict(Z_var, C_tensor, H_inv_pooled)
        y_pred = scaler.inverse_transform(y_pred)
        y_true = scaler.inverse_transform(y_true)

        loss = self.args.yita * self.mae(y_pred[..., 0], y_true[..., 0])
        loss += (1 - self.args.yita) * self.mae(y_pred[..., 1], y_true[..., 1])
        return loss

    def calculate_loss(
        self,
        H_inv,
        Z_var,
        target,
        c,
        time_label,
        scaler,
        loss_weights,
        p=None,
        training=False,
        sample_index=None,
    ):
        del sample_index
        H_inv_pooled = self.tcl4h(H_inv)
        if self.steve_disable_env_stream:
            y_pred = self._predict_invariant_only(H_inv_pooled)
            y_pred = scaler.inverse_transform(y_pred)
            y_true = scaler.inverse_transform(target)
            lp = self.args.yita * self.mae(y_pred[..., 0], y_true[..., 0])
            lp += (1 - self.args.yita) * self.mae(y_pred[..., 1], y_true[..., 1])
            # Single-stream invariant baseline: no environment branch and no
            # multi-task balancing.  Train exactly on the invariant prediction
            # loss so the baseline is not affected by DWA weights for removed
            # auxiliary tasks.
            loss = lp
            sep_loss = [lp.item(), 1.0, 1.0]
            return loss, sep_loss, 0

        C_tensor, _att = self.confounder_ext(Z_var)

        lp = self.pred_loss(Z_var, C_tensor, H_inv_pooled, target, scaler)
        loss = 0
        lm = 0
        sep_loss = [lp.item()]

        if training and self.args.ablation != "idp":
            z1_temp = H_inv_pooled.squeeze(1).reshape(-1, H_inv_pooled.shape[-1])
            z2_temp = C_tensor.reshape(-1, H_inv_pooled.shape[-1])
            self.mi_net.train()
            all_len = z1_temp.shape[0]
            sample_len = max(1, int(all_len * 0.1))
            random_choice = np.random.choice(all_len, sample_len, replace=False)
            temp1 = z1_temp[random_choice].detach()
            temp2 = z2_temp[random_choice].detach()
            for _ in range(5):
                self.optimizer_mi_net.zero_grad()
                mi_loss = self.mi_net.learning_loss(temp1, temp2)
                mi_loss.backward()
                self.optimizer_mi_net.step()
            self.mi_net.eval()

            lm = self.mi_net(z1_temp, z2_temp)
            loss += 0.1 * lm

        loss += loss_weights[0] * lp

        lc = self.variant_loss(C_tensor, time_label, c)
        sep_loss.append(lc.item())
        if self.args.ablation != "cd":
            loss += loss_weights[1] * lc

        lh = self.invariant_loss(H_inv_pooled, time_label, c, p, training)
        sep_loss.append(lh.item())
        if self.args.ablation != "cd":
            loss += loss_weights[2] * lh

        if training is False:
            if self.args.lr_mode == "only":
                loss = lp
            elif self.args.lr_mode == "add":
                loss = lp + lc

        return loss, sep_loss, lm


StableST = OriginalStableST
