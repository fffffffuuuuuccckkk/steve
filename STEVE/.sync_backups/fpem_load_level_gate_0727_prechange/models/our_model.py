import copy
import itertools
import math
import os
import time
import warnings

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fpem import (
    AGCRNEncoder,
    ConvexGatedFusion,
    EnvConditionedInvariantHeads,
    EnvConfounderExtractor,
    EnvMask,
    EnvRouteHeads,
    GraphWaveNetEncoder,
    STAEformerEncoder,
)
from models.fpem.losses import (
    future_mi_loss,
    gain_weighted_swap_loss,
    head_prediction_losses,
    route_losses,
    tensor_float,
    weighted_flow_mae,
)
from models.module import CLUB


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)


class ResidualInvariantProjector(nn.Module):
    """Lightweight near-identity projector for frozen invariant representations."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.reset_identity()

    def reset_identity(self):
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x):
        return x + self.net(x)


class StableST(nn.Module):
    """FPEM core with selectable spatio-temporal backbones."""

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

        self.fpem_backbone = str(getattr(args, "fpem_backbone", "agcrn")).lower()
        if self.fpem_backbone not in {"agcrn", "graphwavenet", "staeformer"}:
            raise ValueError("fpem_backbone must be one of agcrn, graphwavenet, staeformer")
        self.encoder_inv = self._build_backbone_encoder(args, embed_size)
        self.encoder_env = self._build_backbone_encoder(args, embed_size)
        self.encoder_env_teacher = copy.deepcopy(self.encoder_env)
        self._set_requires_grad(self.encoder_env_teacher, False)
        requested_pretrained_agcrn = self._as_bool(
            getattr(args, "fpem_use_pretrained_inv_agcrn", False)
        )
        if requested_pretrained_agcrn and self.fpem_backbone != "agcrn":
            print(
                "[StableST] WARNING fpem_use_pretrained_inv_agcrn=true is only compatible with "
                "fpem_backbone=agcrn; ignoring it for fpem_backbone={}".format(self.fpem_backbone),
                flush=True,
            )
        self.fpem_use_pretrained_inv_agcrn = requested_pretrained_agcrn and self.fpem_backbone == "agcrn"
        self.fpem_pretrained_inv_agcrn_path = str(
            getattr(args, "fpem_pretrained_inv_agcrn_path", "")
        )

        pred_t_dim = int(getattr(args, "input_length", T_dim))
        self.tcl4c = nn.Conv2d(pred_t_dim, output_T_dim, 1, bias=True)
        self.tcl4h = nn.Conv2d(pred_t_dim, output_T_dim, 1, bias=True)
        self.variant_predict_conv_2 = nn.Conv2d(embed_size, output_dim, 1)
        self.invariant_predict_conv_2 = nn.Conv2d(embed_size, output_dim, 1)
        self.hyper_concat_predict_conv_2 = nn.Conv2d(embed_size * 2, output_dim, 1)

        self.inv_trunk = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Dropout(args.dropout),
        )
        self.inv_pred = nn.Linear(embed_size, output_dim)

        self.fpem_use_env_supervision = self._as_bool(getattr(args, "fpem_use_env_supervision", False))
        self.fpem_lambda_env_day_cls = float(getattr(args, "fpem_lambda_env_day_cls", 0.0))
        self.fpem_lambda_env_hour_cls = float(getattr(args, "fpem_lambda_env_hour_cls", 0.0))
        self.fpem_lambda_env_rush_cls = float(getattr(args, "fpem_lambda_env_rush_cls", 0.0))
        self.env_day_head = nn.Linear(embed_size, 2)
        self.env_hour_head = nn.Linear(embed_size, 4)
        self.env_rush_head = nn.Linear(embed_size, 2)

        self.fpem_use_env_supcon = self._as_bool(getattr(args, "fpem_use_env_supcon", False))
        self.fpem_lambda_env_supcon = float(getattr(args, "fpem_lambda_env_supcon", 0.0))
        self.fpem_env_supcon_temperature = float(getattr(args, "fpem_env_supcon_temperature", 0.1))

        self.fpem_use_inv_projector = self._as_bool(getattr(args, "fpem_use_inv_projector", False))
        self.inv_projector = ResidualInvariantProjector(embed_size)
        self.fpem_use_inv_env_adversarial = self._as_bool(
            getattr(args, "fpem_use_inv_env_adversarial", False)
        )
        self.fpem_lambda_inv_env_adv = float(getattr(args, "fpem_lambda_inv_env_adv", 0.0))
        self.fpem_grl_alpha = float(getattr(args, "fpem_grl_alpha", 1.0))
        self.inv_day_head = nn.Linear(embed_size, 2)
        self.inv_hour_head = nn.Linear(embed_size, 4)
        self.inv_rush_head = nn.Linear(embed_size, 2)

        self.fpem_use_cross_cov_sep = self._as_bool(getattr(args, "fpem_use_cross_cov_sep", False))
        self.fpem_lambda_cross_cov_sep = float(getattr(args, "fpem_lambda_cross_cov_sep", 0.0))

        self.fpem_env_use_exogenous = self._as_bool(getattr(args, "fpem_env_use_exogenous", True))
        self.fpem_env_exogenous_hour_dim = int(getattr(args, "fpem_env_exogenous_hour_dim", 8))
        self.fpem_env_exogenous_day_dim = int(getattr(args, "fpem_env_exogenous_day_dim", 4))
        self.fpem_env_exogenous_rush_dim = int(getattr(args, "fpem_env_exogenous_rush_dim", 4))
        self.fpem_env_exogenous_load_dim = int(getattr(args, "fpem_env_exogenous_load_dim", 8))
        self.fpem_env_exogenous_load_num_embeddings = int(
            getattr(args, "fpem_env_exogenous_load_num_embeddings", 6)
        )
        self.fpem_env_exogenous_load_channels = int(getattr(args, "d_input", in_channels))
        self.fpem_env_exogenous_scale = float(getattr(args, "fpem_env_exogenous_scale", 0.1))
        self.fpem_env_exogenous_feature_dim = (
            self.fpem_env_exogenous_hour_dim
            + self.fpem_env_exogenous_day_dim
            + self.fpem_env_exogenous_rush_dim
            + self.fpem_env_exogenous_load_channels * self.fpem_env_exogenous_load_dim
        )

        self.fpem_use_env_mask = self._as_bool(getattr(args, "fpem_use_env_mask", False))
        self.fpem_use_env_route = self._as_bool(getattr(args, "fpem_use_env_route", False))
        self.fpem_env_route_k = int(getattr(args, "fpem_env_route_k", 3))
        self.fpem_force_uniform_route = self._as_bool(
            getattr(args, "fpem_force_uniform_route", False)
        )
        self.fpem_env_rep_ablation = str(getattr(args, "fpem_env_rep_ablation", "normal")).lower()
        if self.fpem_env_rep_ablation not in {"normal", "zero", "shuffle_batch"}:
            raise ValueError("fpem_env_rep_ablation must be one of normal, zero, shuffle_batch")
        self.fpem_env_route_head_mode = str(getattr(args, "fpem_env_route_head_mode", "concat_input")).lower()
        if self.fpem_env_route_head_mode not in {
            "concat_input",
            "hyper_inv_film",
            "hyper_inv_film_proto",
            "hyper_inv_film_proto_concat",
            "hyper_inv_film_proto_input_concat",
            "hyper_inv_film_proto_input_add",
        }:
            raise ValueError(
                "fpem_env_route_head_mode must be one of "
                "concat_input, hyper_inv_film, hyper_inv_film_proto, "
                "hyper_inv_film_proto_concat, hyper_inv_film_proto_input_concat, "
                "hyper_inv_film_proto_input_add"
            )
        self.fpem_env_route_use_inv_fallback_expert = self._as_bool(
            getattr(args, "fpem_env_route_use_inv_fallback_expert", True)
        )
        self.fpem_env_route_train_mode = str(getattr(args, "fpem_env_route_train_mode", "soft_oracle")).lower()
        self.fpem_use_env_fusion = self._as_bool(getattr(args, "fpem_use_env_fusion", True))
        self.fpem_use_swap = self._as_bool(getattr(args, "fpem_use_swap", False))
        self.fpem_use_future_mi = self._as_bool(getattr(args, "fpem_use_future_mi", False))
        self.fpem_use_grad_consensus = self._as_bool(getattr(args, "fpem_use_grad_consensus", False))
        self.fpem_gc_pred_loss_only = self._as_bool(getattr(args, "fpem_gc_pred_loss_only", True))

        self.fpem_use_confounder_extractor = self._as_bool(
            getattr(args, "fpem_use_confounder_extractor", True)
        )
        self.fpem_confounder_use_mask = self._as_bool(
            getattr(args, "fpem_confounder_use_mask", False)
        )
        self.confounder_extractor = EnvConfounderExtractor(
            hidden_dim=embed_size,
            num_basis=int(getattr(args, "fpem_confounder_num_basis", 8)),
            num_heads=int(getattr(args, "fpem_confounder_num_heads", 4)),
            dropout=float(getattr(args, "fpem_confounder_dropout", 0.0)),
            use_temporal_attn_pool=self._as_bool(
                getattr(args, "fpem_confounder_use_temporal_attn_pool", True)
            ),
        )

        self.fpem_use_club_mi = self._as_bool(getattr(args, "fpem_use_club_mi", True))
        self.fpem_lambda_club_mi = float(getattr(args, "fpem_lambda_club_mi", 0.01))
        self.fpem_club_steps = int(getattr(args, "fpem_club_steps", 1))
        self.fpem_club_sample_ratio = float(getattr(args, "fpem_club_sample_ratio", 0.1))
        club_hidden = getattr(args, "fpem_club_hidden_dim", None)
        if club_hidden is None:
            club_hidden = embed_size * int(getattr(args, "fpem_club_hidden_mul", 2))
        self.mi_net = CLUB(embed_size, embed_size, int(club_hidden))
        self.optimizer_mi_net = torch.optim.Adam(
            self.mi_net.parameters(),
            lr=float(getattr(args, "fpem_club_lr", 1e-3)),
        )

        mask_hidden = int(getattr(args, "fpem_env_mask_hidden_dim", embed_size))
        self.env_mask = EnvMask(embed_size, mask_hidden, args.dropout)

        route_hidden = int(getattr(args, "fpem_env_route_hidden_dim", embed_size))
        self.route_heads = EnvRouteHeads(embed_size, output_dim, self.fpem_env_route_k, route_hidden, args.dropout)
        self.fusion = ConvexGatedFusion(embed_size, self.fpem_env_route_k, args.dropout)
        self.fpem_use_env_prototype_router = self._as_bool(
            getattr(args, "fpem_use_env_prototype_router", False)
        )
        self.fpem_env_route_target_mode = str(
            getattr(args, "fpem_env_route_target_mode", "prediction_oracle")
        ).lower()
        self.fpem_env_prototype_temperature = float(getattr(args, "fpem_env_prototype_temperature", 1.0))
        self.fpem_env_route_hybrid_alpha = float(getattr(args, "fpem_env_route_hybrid_alpha", 1.0))
        self.fpem_env_route_hybrid_alpha_start = float(
            getattr(args, "fpem_env_route_hybrid_alpha_start", self.fpem_env_route_hybrid_alpha)
        )
        self.fpem_env_route_hybrid_alpha_end = float(
            getattr(args, "fpem_env_route_hybrid_alpha_end", self.fpem_env_route_hybrid_alpha)
        )
        self.fpem_env_route_hybrid_alpha_decay_epochs = int(
            getattr(args, "fpem_env_route_hybrid_alpha_decay_epochs", 0)
        )
        self.fpem_use_sinkhorn_route = self._as_bool(getattr(args, "fpem_use_sinkhorn_route", False))
        self.fpem_sinkhorn_iters = int(getattr(args, "fpem_sinkhorn_iters", 3))
        self.fpem_sinkhorn_epsilon = float(getattr(args, "fpem_sinkhorn_epsilon", 0.05))
        self.fpem_env_route_inference_mode = str(
            getattr(args, "fpem_env_route_inference_mode", "mlp")
        ).lower()
        if self.fpem_env_route_inference_mode not in {
            "mlp",
            "nearest_prototype",
            "gaussian",
            "gaussian_viterbi",
        }:
            raise ValueError(
                "fpem_env_route_inference_mode must be one of "
                "mlp, nearest_prototype, gaussian, gaussian_viterbi"
            )
        self.fpem_env_sinkhorn_prediction_alpha_start = float(
            getattr(args, "fpem_env_sinkhorn_prediction_alpha_start", 0.2)
        )
        self.fpem_env_sinkhorn_prediction_alpha_final = float(
            getattr(args, "fpem_env_sinkhorn_prediction_alpha_final", 1.0)
        )
        self.fpem_env_sinkhorn_environment_beta_start = float(
            getattr(args, "fpem_env_sinkhorn_environment_beta_start", 1.0)
        )
        self.fpem_env_sinkhorn_environment_beta_final = float(
            getattr(args, "fpem_env_sinkhorn_environment_beta_final", 0.2)
        )
        self.fpem_env_sinkhorn_schedule_start_epoch = int(
            getattr(args, "fpem_env_sinkhorn_schedule_start_epoch", 5)
        )
        self.fpem_env_sinkhorn_schedule_end_epoch = int(
            getattr(args, "fpem_env_sinkhorn_schedule_end_epoch", 15)
        )
        self.fpem_env_sinkhorn_temporal_lambda = float(
            getattr(args, "fpem_env_sinkhorn_temporal_lambda", 0.05)
        )
        self.fpem_env_sinkhorn_gaussian_ema = float(
            getattr(args, "fpem_env_sinkhorn_gaussian_ema", 0.05)
        )
        self.fpem_env_sinkhorn_gaussian_min_var = float(
            getattr(args, "fpem_env_sinkhorn_gaussian_min_var", 1e-4)
        )
        self.fpem_sinkhorn_warmup_epochs = int(getattr(args, "fpem_sinkhorn_warmup_epochs", 10))
        self.fpem_sinkhorn_soft_end_epoch = int(getattr(args, "fpem_sinkhorn_soft_end_epoch", 20))
        self.fpem_sinkhorn_temperature_start = float(getattr(args, "fpem_sinkhorn_temperature_start", 1.0))
        self.fpem_sinkhorn_temperature_final = float(getattr(args, "fpem_sinkhorn_temperature_final", 0.3))
        self.fpem_sinkhorn_lambda_common = float(getattr(args, "fpem_sinkhorn_lambda_common", 0.1))
        self.fpem_risk_router_temperature = float(getattr(args, "fpem_risk_router_temperature", 1.0))
        self.fpem_risk_router_lambda = float(getattr(args, "fpem_risk_router_lambda", 0.5))
        self.fpem_risk_router_pairwise_lambda = float(
            getattr(args, "fpem_risk_router_pairwise_lambda", 0.0)
        )
        self.fpem_env_teacher_ema_momentum = float(getattr(args, "fpem_env_teacher_ema_momentum", 0.995))
        self.fpem_env_partition_start_epoch = int(getattr(args, "fpem_env_partition_start_epoch", 5))
        self.fpem_env_partition_update_interval = int(getattr(args, "fpem_env_partition_update_interval", 5))
        self.fpem_env_partition_freeze_last_epochs = int(getattr(args, "fpem_env_partition_freeze_last_epochs", 15))
        self.fpem_env_max_clusters = int(getattr(args, "fpem_env_max_clusters", self.fpem_env_route_k))
        self.fpem_env_max_clusters = max(1, min(self.fpem_env_max_clusters, self.fpem_env_route_k))
        self.fpem_env_min_cluster_ratio = float(getattr(args, "fpem_env_min_cluster_ratio", 0.08))
        self.fpem_env_gmm_n_init = int(getattr(args, "fpem_env_gmm_n_init", 10))
        self.fpem_env_gmm_variance_floor = float(getattr(args, "fpem_env_gmm_variance_floor", 1e-4))
        self.fpem_env_progressive_lambda_common = float(getattr(args, "fpem_env_progressive_lambda_common", 0.2))
        self.fpem_env_cluster_compactness_lambda = float(getattr(args, "fpem_env_cluster_compactness_lambda", 0.01))
        self.fpem_env_cluster_consistency_lambda = float(getattr(args, "fpem_env_cluster_consistency_lambda", 0.05))
        self.fpem_env_split_perturb_std = float(getattr(args, "fpem_env_split_perturb_std", 1e-5))
        self.fpem_route_observable_dim = int(4 * embed_size + 15)
        self.fpem_expert_uniform_warmup_epochs = int(getattr(args, "fpem_expert_uniform_warmup_epochs", 0))
        self.fpem_env_route_balance_warmup_epochs = int(getattr(args, "fpem_env_route_balance_warmup_epochs", 0))
        self.fpem_env_route_initial_temperature = float(getattr(args, "fpem_env_route_initial_temperature", 1.0))
        self.fpem_env_route_final_temperature = float(getattr(args, "fpem_env_route_final_temperature", 0.3))
        self.env_prototypes = nn.Parameter(torch.randn(self.fpem_env_route_k, embed_size))
        self.register_buffer(
            "env_route_gaussian_mu",
            torch.zeros(self.fpem_env_route_k, self.fpem_route_observable_dim),
        )
        self.register_buffer(
            "env_route_gaussian_var",
            torch.ones(self.fpem_env_route_k, self.fpem_route_observable_dim),
        )
        self.register_buffer("env_route_gaussian_mass", torch.ones(self.fpem_env_route_k))
        self.register_buffer("env_route_gaussian_initialized", torch.zeros(1))
        self.register_buffer(
            "env_route_transition_counts",
            torch.ones(self.fpem_env_route_k, self.fpem_env_route_k),
        )
        self.register_buffer("progressive_gmm_initialized", torch.zeros(1))
        self.register_buffer("progressive_feature_mean", torch.zeros(embed_size))
        self.register_buffer("progressive_feature_std", torch.ones(embed_size))
        self.register_buffer("progressive_gmm_log_prior", torch.zeros(self.fpem_env_route_k))
        self.register_buffer("progressive_gmm_mu", torch.zeros(self.fpem_env_route_k, embed_size))
        self.register_buffer("progressive_gmm_var", torch.ones(self.fpem_env_route_k, embed_size))
        self.register_buffer("progressive_cluster_to_expert", torch.arange(self.fpem_env_route_k, dtype=torch.long))
        active_mask = torch.zeros(self.fpem_env_route_k)
        active_mask[: max(1, self.fpem_env_max_clusters)] = 1.0
        self.register_buffer("progressive_active_expert_mask", active_mask)
        self.register_buffer("progressive_active_cluster_count", torch.ones((), dtype=torch.long))
        self.register_buffer("progressive_last_partition_epoch", torch.zeros((), dtype=torch.long))
        self.progressive_fixed_train_assignments = None
        self.progressive_partition_history = []
        self._latest_progressive_partition_logs = {}
        self._latest_progressive_teacher_embedding = None
        self._sinkhorn_warned = False
        self._expert_zero_streak = 0
        self.hyper_inv_heads = EnvConditionedInvariantHeads(
            embed_size,
            self.fpem_env_route_k,
            int(getattr(args, "fpem_hyper_hidden_dim", route_hidden)),
            args.dropout,
            alpha_mode=str(getattr(args, "fpem_hyper_alpha_mode", "sample_gate")),
        )
        self.hyper_concat_input_heads = EnvConditionedInvariantHeads(
            embed_size,
            self.fpem_env_route_k,
            int(getattr(args, "fpem_hyper_hidden_dim", route_hidden)),
            args.dropout,
            alpha_mode=str(getattr(args, "fpem_hyper_alpha_mode", "sample_gate")),
            state_dim=embed_size * 2,
            context_dim=embed_size * 2,
        )
        hyper_router_dim = self.fpem_env_route_k + (1 if self.fpem_env_route_use_inv_fallback_expert else 0)
        self.hyper_router = nn.Sequential(
            nn.Linear(embed_size * 2, route_hidden),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(route_hidden, hyper_router_dim),
        )
        self.risk_router = nn.Sequential(
            nn.Linear(self.fpem_route_observable_dim, route_hidden),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(route_hidden, self.fpem_env_route_k),
        )

        if self.fpem_use_future_mi:
            future_hidden = int(getattr(args, "fpem_future_mi_hidden_dim", embed_size))
            self.fpem_future_mi_target_mode = str(
                getattr(args, "fpem_future_mi_target_mode", "env_encoder")
            ).lower()
            if self.fpem_future_mi_target_mode not in {"env_encoder", "residual_mlp"}:
                raise ValueError("fpem_future_mi_target_mode must be env_encoder or residual_mlp")
            self.future_env_encoder = (
                nn.Sequential(
                    nn.Linear(output_dim, future_hidden),
                    nn.ReLU(),
                    nn.Linear(future_hidden, embed_size),
                )
                if self.fpem_future_mi_target_mode == "residual_mlp"
                else None
            )
            self.future_target_proj = (
                nn.Linear(output_dim, int(getattr(args, "d_input", output_dim)))
                if int(getattr(args, "d_input", output_dim)) != output_dim
                else None
            )
            self.future_env_mu = nn.Linear(embed_size, embed_size)
            self.future_env_logvar = nn.Linear(embed_size, embed_size)
        else:
            self.fpem_future_mi_target_mode = "env_encoder"
            self.future_env_encoder = None
            self.future_target_proj = None
            self.future_env_mu = None
            self.future_env_logvar = None

        self.reset_parameters()
        self._sync_progressive_teacher_from_student()
        self.inv_projector.reset_identity()
        self.hyper_inv_heads.reset_identity()
        self.hyper_concat_input_heads.reset_identity()
        with torch.no_grad():
            self.env_prototypes.copy_(F.normalize(self.env_prototypes, dim=-1, eps=1e-8))
        self.env_hour_embedding = nn.Embedding(24, self.fpem_env_exogenous_hour_dim)
        self.env_day_embedding = nn.Embedding(2, self.fpem_env_exogenous_day_dim)
        self.env_rush_embedding = nn.Embedding(2, self.fpem_env_exogenous_rush_dim)
        self.env_load_embedding = nn.Embedding(
            self.fpem_env_exogenous_load_num_embeddings,
            self.fpem_env_exogenous_load_dim,
        )
        self.env_exog_proj = nn.Sequential(
            nn.Linear(self.fpem_env_exogenous_feature_dim, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, int(getattr(args, "d_input", in_channels))),
        )
        self._init_env_exogenous_parameters()
        self._env_exog_warned_missing = False
        self._latest_env_exog_logs = None
        if self.fpem_use_pretrained_inv_agcrn:
            self._load_pretrained_inv_agcrn()
            self.freeze_invariant_encoder()
        self._configure_trainable_for_selected_mode()

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

    def _init_env_exogenous_parameters(self):
        for module in [
            self.env_hour_embedding,
            self.env_day_embedding,
            self.env_rush_embedding,
            self.env_load_embedding,
            self.env_exog_proj,
        ]:
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.uniform_(p)

    def _torch_load_checkpoint(self, path, map_location="cpu"):
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)

    def _load_pretrained_inv_agcrn(self):
        path = self.fpem_pretrained_inv_agcrn_path
        if not path:
            raise ValueError(
                "fpem_use_pretrained_inv_agcrn=true requires --fpem_pretrained_inv_agcrn_path"
            )
        if not os.path.exists(path):
            raise FileNotFoundError(f"pretrained AGCRN checkpoint not found: {path}")

        ckpt = self._torch_load_checkpoint(path, map_location="cpu")
        state = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
        if not isinstance(state, dict):
            raise ValueError(f"invalid pretrained AGCRN checkpoint format: {path}")

        encoder_state = {}
        for key, value in state.items():
            if key.startswith("encoder."):
                encoder_state[key[len("encoder."):]] = value
            elif key.startswith("module.encoder."):
                encoder_state[key[len("module.encoder."):]] = value
            elif key.startswith("encoder_inv."):
                encoder_state[key[len("encoder_inv."):]] = value
            elif key.startswith("module.encoder_inv."):
                encoder_state[key[len("module.encoder_inv."):]] = value

        if not encoder_state:
            raise ValueError(f"checkpoint has no AGCRN encoder weights: {path}")

        result = self.encoder_inv.load_state_dict(encoder_state, strict=True)
        print(
            "[StableST] loaded pretrained invariant AGCRN encoder "
            f"from {path}; missing={list(result.missing_keys)} unexpected={list(result.unexpected_keys)}",
            flush=True,
        )

    def freeze_invariant_encoder(self):
        for param in self.encoder_inv.parameters():
            param.requires_grad_(False)
        self.encoder_inv.eval()

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "fpem_use_pretrained_inv_agcrn", False):
            self.encoder_inv.eval()
        if hasattr(self, "encoder_env_teacher"):
            self.encoder_env_teacher.eval()
        return self

    def _sync_progressive_teacher_from_student(self):
        if not hasattr(self, "encoder_env_teacher"):
            return
        self.encoder_env_teacher.load_state_dict(self.encoder_env.state_dict(), strict=True)
        self._set_requires_grad(self.encoder_env_teacher, False)
        self.encoder_env_teacher.eval()

    def update_progressive_teacher_ema(self):
        if not self._progressive_gmm_environment_enabled():
            return
        momentum = max(0.0, min(1.0, float(self.fpem_env_teacher_ema_momentum)))
        with torch.no_grad():
            teacher_state = dict(self.encoder_env_teacher.named_parameters())
            for name, student_param in self.encoder_env.named_parameters():
                teacher_param = teacher_state.get(name)
                if teacher_param is not None:
                    teacher_param.mul_(momentum).add_(student_param.detach(), alpha=1.0 - momentum)
            teacher_buffers = dict(self.encoder_env_teacher.named_buffers())
            for name, student_buffer in self.encoder_env.named_buffers():
                teacher_buffer = teacher_buffers.get(name)
                if teacher_buffer is not None and teacher_buffer.shape == student_buffer.shape:
                    if torch.is_floating_point(teacher_buffer):
                        teacher_buffer.mul_(momentum).add_(student_buffer.detach(), alpha=1.0 - momentum)
                    else:
                        teacher_buffer.copy_(student_buffer.detach())
        self.encoder_env_teacher.eval()

    def _build_backbone_encoder(self, args, embed_size):
        backbone = self.fpem_backbone
        if backbone == "agcrn":
            agcrn_embed_dim = int(getattr(args, "agcrn_embed_dim", 10))
            agcrn_layers = int(getattr(args, "agcrn_num_layers", 2))
            agcrn_cheb_k = int(getattr(args, "agcrn_cheb_k", 2))
            return AGCRNEncoder(
                args.num_nodes, args.d_input, embed_size, agcrn_cheb_k, agcrn_embed_dim, agcrn_layers
            )
        if backbone == "graphwavenet":
            return GraphWaveNetEncoder(
                args.num_nodes,
                args.d_input,
                embed_size,
                adj=self.adj,
                num_layers=int(getattr(args, "graphwavenet_layers", 4)),
                kernel_size=int(getattr(args, "graphwavenet_kernel_size", 2)),
                dropout=float(getattr(args, "graphwavenet_dropout", getattr(args, "dropout", 0.1))),
            )
        if backbone == "staeformer":
            return STAEformerEncoder(
                args.num_nodes,
                args.d_input,
                embed_size,
                input_length=int(getattr(args, "staeformer_input_length_max", max(int(getattr(args, "input_length", 12)), 512))),
                num_layers=int(getattr(args, "staeformer_layers", 2)),
                num_heads=int(getattr(args, "staeformer_heads", 4)),
                dropout=float(getattr(args, "staeformer_dropout", getattr(args, "dropout", 0.1))),
                mlp_ratio=float(getattr(args, "staeformer_mlp_ratio", 2.0)),
            )
        raise ValueError("unsupported fpem_backbone={}".format(backbone))

    def _epoch_index(self, p):
        if p is None:
            return 0
        value = float(p)
        if value <= 1.0:
            return int(round(value * float(getattr(self.args, "epochs", 1))))
        return int(round(value))

    def _env_labels_from_time(self, time_label, ref):
        if time_label is None:
            return None
        time_label = time_label.to(device=ref.device, dtype=torch.long).view(-1)
        day = (time_label < 24).long()
        hour = torch.remainder(time_label, 24)
        hour_bin = torch.zeros_like(hour)
        hour_bin = torch.where((hour >= 6) & (hour <= 9), torch.ones_like(hour_bin), hour_bin)
        hour_bin = torch.where((hour >= 10) & (hour <= 15), torch.full_like(hour_bin, 2), hour_bin)
        hour_bin = torch.where(hour >= 16, torch.full_like(hour_bin, 3), hour_bin)
        rush = (
            ((hour >= 7) & (hour <= 9))
            | ((hour >= 17) & (hour <= 19))
        ).long()
        return {"day": day, "hour": hour, "hour_bin": hour_bin, "rush": rush}

    def _zero_env_exog_logs(self, ref, available=0.0):
        return {
            "fpem/fpem_env_use_exogenous": ref.new_tensor(float(self.fpem_env_use_exogenous)),
            "fpem/env_exogenous_available": ref.new_tensor(float(available)),
            "fpem/env_exogenous_time_available": ref.new_tensor(float(available)),
            "fpem/env_exogenous_load_available": ref.new_zeros(()),
            "fpem/env_exogenous_feature_dim": ref.new_tensor(float(self.fpem_env_exogenous_feature_dim)),
            "fpem/env_exogenous_embedding_norm": ref.new_zeros(()),
            "fpem/env_exogenous_load_embedding_norm": ref.new_zeros(()),
        }

    def _route_aux_stats_from_inputs(self, x, time_label=None, exog=None):
        """Small observable route-feature block, never the full input tensor.

        Output shape is fixed to [B, 12]:
          - x/latest traffic stats: mean, std, max, last-minus-first mean
          - time_label stats: day, hour/23, sin(hour), cos(hour), rush
          - c/load-level latest stats: mean, std, max
        These are detached immediately so the route assignment path does not
        create extra autograd retention.
        """
        bsz = x.shape[0]
        flow = x[..., 0].detach().float()
        x_flat = flow.reshape(bsz, -1)
        x_mean = x_flat.mean(dim=1)
        x_std = x_flat.std(dim=1, unbiased=False)
        x_max = x_flat.max(dim=1)[0]
        if flow.shape[1] > 1:
            slope = flow[:, -1].mean(dim=1) - flow[:, 0].mean(dim=1)
        else:
            slope = flow[:, -1].mean(dim=1)
        x_stats = torch.stack([x_mean, x_std, x_max, slope], dim=-1)

        labels = self._env_labels_from_time(time_label, x)
        if labels is None or labels["hour"].numel() != bsz:
            time_stats = x.new_zeros(bsz, 5).float()
        else:
            hour = labels["hour"].float()
            hour_norm = hour / 23.0
            angle = hour * (6.283185307179586 / 24.0)
            time_stats = torch.stack(
                [
                    labels["day"].float(),
                    hour_norm,
                    torch.sin(angle),
                    torch.cos(angle),
                    labels["rush"].float(),
                ],
                dim=-1,
            )

        if exog is not None and torch.is_tensor(exog) and exog.shape[0] == bsz and exog.shape[2] == x.shape[2]:
            c_latest = exog[:, -1].detach().float()
            c_flat = c_latest.reshape(bsz, -1)
            c_stats = torch.stack(
                [
                    c_flat.mean(dim=1),
                    c_flat.std(dim=1, unbiased=False),
                    c_flat.max(dim=1)[0],
                ],
                dim=-1,
            )
        else:
            c_stats = x.new_zeros(bsz, 3).float()
        return torch.cat([x_stats, time_stats.to(device=x.device), c_stats.to(device=x.device)], dim=-1).detach()

    def _load_exogenous_features(self, x, exog=None):
        b, _t, n, _d = x.shape
        if exog is None or not torch.is_tensor(exog):
            return None
        if exog.shape[0] != b or exog.shape[2] != n:
            return None
        c_latest = exog[:, -1]
        if c_latest.dim() != 3:
            return None
        if c_latest.shape[-1] < self.fpem_env_exogenous_load_channels:
            pad = c_latest.new_zeros(
                b,
                n,
                self.fpem_env_exogenous_load_channels - c_latest.shape[-1],
            )
            c_latest = torch.cat([c_latest, pad], dim=-1)
        c_latest = c_latest[..., : self.fpem_env_exogenous_load_channels]
        idx = torch.round(c_latest).long().clamp(
            0,
            max(self.fpem_env_exogenous_load_num_embeddings - 1, 0),
        )
        load_emb = self.env_load_embedding(idx)
        return load_emb.reshape(b, n, self.fpem_env_exogenous_load_channels * self.fpem_env_exogenous_load_dim)

    def _apply_env_exogenous(self, x, time_label=None, exog=None):
        logs = self._zero_env_exog_logs(x, available=0.0)
        if not self.fpem_env_use_exogenous:
            return x, logs
        labels = self._env_labels_from_time(time_label, x)
        if labels is None or labels["hour"].numel() != x.shape[0]:
            if not self._env_exog_warned_missing:
                print(
                    "[StableST] WARNING fpem_env_use_exogenous=true but time_label is unavailable "
                    "or has mismatched batch size; falling back to original encoder_env input.",
                    flush=True,
                )
                self._env_exog_warned_missing = True
            return x, logs

        hour_emb = self.env_hour_embedding(labels["hour"].clamp(0, 23))
        day_emb = self.env_day_embedding(labels["day"].clamp(0, 1))
        rush_emb = self.env_rush_embedding(labels["rush"].clamp(0, 1))
        time_exog = torch.cat([hour_emb, day_emb, rush_emb], dim=-1).to(dtype=x.dtype)
        time_exog = time_exog[:, None, :].expand(-1, x.shape[2], -1)
        load_exog = self._load_exogenous_features(x, exog=exog)
        if load_exog is None:
            load_exog = x.new_zeros(
                x.shape[0],
                x.shape[2],
                self.fpem_env_exogenous_load_channels * self.fpem_env_exogenous_load_dim,
            )
            load_available = 0.0
            if not self._env_exog_warned_missing:
                print(
                    "[StableST] WARNING fpem_env_use_exogenous=true but load-level c is unavailable "
                    "or has mismatched shape; using only time/day/rush exogenous features.",
                    flush=True,
                )
                self._env_exog_warned_missing = True
        else:
            load_exog = load_exog.to(dtype=x.dtype)
            load_available = 1.0
        exog_features = torch.cat([time_exog, load_exog], dim=-1)
        delta = self.env_exog_proj(exog_features).to(dtype=x.dtype)
        delta = self.fpem_env_exogenous_scale * delta
        delta = delta[:, None, :, :].expand(-1, x.shape[1], -1, -1)
        logs.update({
            "fpem/env_exogenous_available": x.new_tensor(1.0),
            "fpem/env_exogenous_time_available": x.new_tensor(1.0),
            "fpem/env_exogenous_load_available": x.new_tensor(float(load_available)),
            "fpem/env_exogenous_embedding_norm": exog_features.detach().float().norm(dim=-1).mean().to(dtype=x.dtype),
            "fpem/env_exogenous_load_embedding_norm": load_exog.detach().float().norm(dim=-1).mean().to(dtype=x.dtype),
        })
        return x + delta, logs

    @staticmethod
    def _pool_nodes(x):
        return x.mean(dim=1)

    @staticmethod
    def _acc(logits, labels):
        if logits is None or labels is None or labels.numel() == 0:
            return logits.new_zeros(()) if torch.is_tensor(logits) else torch.tensor(0.0)
        return (logits.detach().argmax(dim=-1) == labels).to(dtype=logits.dtype).mean()

    def _env_supervision_loss(self, e_env, labels, training):
        zero = e_env.new_zeros(())
        logs = {
            "fpem/loss_env_day": zero,
            "fpem/loss_env_hour": zero,
            "fpem/loss_env_rush": zero,
            "fpem/env_day_acc": zero,
            "fpem/env_hour_acc": zero,
            "fpem/env_rush_acc": zero,
        }
        if labels is None or not (training and self.fpem_use_env_supervision):
            return zero, logs
        pooled = self._pool_nodes(e_env)
        day_logits = self.env_day_head(pooled)
        hour_logits = self.env_hour_head(pooled)
        rush_logits = self.env_rush_head(pooled)
        day_loss = F.cross_entropy(day_logits.float(), labels["day"])
        hour_loss = F.cross_entropy(hour_logits.float(), labels["hour_bin"])
        rush_loss = F.cross_entropy(rush_logits.float(), labels["rush"])
        total = (
            self.fpem_lambda_env_day_cls * day_loss
            + self.fpem_lambda_env_hour_cls * hour_loss
            + self.fpem_lambda_env_rush_cls * rush_loss
        )
        logs.update({
            "fpem/loss_env_day": day_loss.detach(),
            "fpem/loss_env_hour": hour_loss.detach(),
            "fpem/loss_env_rush": rush_loss.detach(),
            "fpem/env_day_acc": self._acc(day_logits, labels["day"]),
            "fpem/env_hour_acc": self._acc(hour_logits, labels["hour_bin"]),
            "fpem/env_rush_acc": self._acc(rush_logits, labels["rush"]),
        })
        return total, logs

    def _inv_env_adversarial_loss(self, h_inv, labels, training):
        zero = h_inv.new_zeros(())
        logs = {
            "fpem/loss_inv_env_adv": zero,
            "fpem/inv_day_acc": zero,
            "fpem/inv_hour_acc": zero,
            "fpem/inv_rush_acc": zero,
        }
        if labels is None or not (training and self.fpem_use_inv_env_adversarial):
            return zero, logs
        pooled = self._pool_nodes(h_inv)
        pooled = grad_reverse(pooled, self.fpem_grl_alpha)
        day_logits = self.inv_day_head(pooled)
        hour_logits = self.inv_hour_head(pooled)
        rush_logits = self.inv_rush_head(pooled)
        day_loss = F.cross_entropy(day_logits.float(), labels["day"])
        hour_loss = F.cross_entropy(hour_logits.float(), labels["hour_bin"])
        rush_loss = F.cross_entropy(rush_logits.float(), labels["rush"])
        raw = day_loss + hour_loss + rush_loss
        logs.update({
            "fpem/loss_inv_env_adv": raw.detach(),
            "fpem/inv_day_acc": self._acc(day_logits, labels["day"]),
            "fpem/inv_hour_acc": self._acc(hour_logits, labels["hour_bin"]),
            "fpem/inv_rush_acc": self._acc(rush_logits, labels["rush"]),
        })
        return self.fpem_lambda_inv_env_adv * raw, logs

    def _env_supcon_loss(self, e_env, labels, training):
        zero = e_env.new_zeros(())
        logs = {
            "fpem/loss_env_supcon": zero,
            "fpem/env_supcon_valid_pairs": zero,
        }
        if labels is None or not (training and self.fpem_use_env_supcon):
            return zero, logs
        z = F.normalize(self._pool_nodes(e_env).float(), dim=-1, eps=1e-8)
        b = z.shape[0]
        if b <= 1:
            return zero, logs
        day = labels["day"]
        hour_bin = labels["hour_bin"]
        eye = torch.eye(b, dtype=torch.bool, device=z.device)
        positive = (day[:, None] == day[None, :]) & ((hour_bin[:, None] - hour_bin[None, :]).abs() <= 1) & (~eye)
        if not bool(positive.any()):
            return zero, logs
        strong_negative = (hour_bin[:, None] == hour_bin[None, :]) & (day[:, None] != day[None, :]) & (~eye)
        logits = z.matmul(z.t()) / max(self.fpem_env_supcon_temperature, 1e-6)
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits).masked_fill(eye, 0.0)
        weights = torch.ones_like(exp_logits)
        weights = torch.where(strong_negative, weights.new_tensor(2.0), weights)
        denom = (exp_logits * weights).sum(dim=1).clamp_min(1e-8)
        pos_exp = (exp_logits * positive.to(dtype=exp_logits.dtype)).sum(dim=1)
        valid = pos_exp > 0
        if not bool(valid.any()):
            return zero, logs
        loss = -torch.log(pos_exp[valid].clamp_min(1e-8) / denom[valid]).mean()
        weighted = self.fpem_lambda_env_supcon * loss.to(dtype=e_env.dtype)
        logs.update({
            "fpem/loss_env_supcon": loss.detach().to(dtype=e_env.dtype),
            "fpem/env_supcon_valid_pairs": positive.to(dtype=e_env.dtype).sum().detach(),
        })
        return weighted, logs

    def _cross_cov_sep_loss(self, h_inv, e_env, training):
        zero = h_inv.new_zeros(())
        logs = {"fpem/loss_cross_cov_sep": zero}
        if not (training and self.fpem_use_cross_cov_sep):
            return zero, logs
        h = h_inv.reshape(-1, h_inv.shape[-1]).float()
        e = e_env.reshape(-1, e_env.shape[-1]).float()
        if h.shape[0] <= 1:
            return zero, logs
        h = F.normalize(h - h.mean(dim=0, keepdim=True), dim=0, eps=1e-6)
        e = F.normalize(e - e.mean(dim=0, keepdim=True), dim=0, eps=1e-6)
        cov = h.t().matmul(e) / float(max(h.shape[0] - 1, 1))
        raw = cov.pow(2).sum()
        logs["fpem/loss_cross_cov_sep"] = raw.detach().to(dtype=h_inv.dtype)
        return (self.fpem_lambda_cross_cov_sep * raw).to(dtype=h_inv.dtype), logs

    def _route_temperature(self, epoch):
        start = max(self.fpem_env_route_initial_temperature, 1e-6)
        end = max(self.fpem_env_route_final_temperature, 1e-6)
        warm = max(self.fpem_env_route_balance_warmup_epochs, 0)
        if epoch is None or warm <= 0:
            return max(self.fpem_env_prototype_temperature, 1e-6)
        progress = min(max(float(epoch) / float(warm), 0.0), 1.0)
        return start + (end - start) * progress

    def _hybrid_alpha(self, epoch):
        if self.fpem_env_route_hybrid_alpha_decay_epochs <= 0 or epoch is None:
            return self.fpem_env_route_hybrid_alpha
        progress = min(max(float(epoch) / float(self.fpem_env_route_hybrid_alpha_decay_epochs), 0.0), 1.0)
        return self.fpem_env_route_hybrid_alpha_start + (
            self.fpem_env_route_hybrid_alpha_end - self.fpem_env_route_hybrid_alpha_start
        ) * progress

    def _sinkhorn(self, logits):
        eps = max(self.fpem_sinkhorn_epsilon, 1e-6)
        q = torch.exp((logits.float() / eps).clamp(-50.0, 50.0))
        if not torch.isfinite(q).all() or q.numel() == 0:
            raise RuntimeError("non-finite sinkhorn input")
        b, k = q.shape
        if b <= 1 or k <= 1:
            return torch.softmax(logits.float(), dim=-1).to(dtype=logits.dtype)
        for _ in range(max(self.fpem_sinkhorn_iters, 1)):
            q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-8)
            q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-8) * (float(b) / float(k))
        q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-8)
        if not torch.isfinite(q).all():
            raise RuntimeError("non-finite sinkhorn output")
        return q.to(dtype=logits.dtype)

    def _prototype_route(self, e_env, training, epoch):
        pooled = F.normalize(self._pool_nodes(e_env).float(), dim=-1, eps=1e-8)
        prototypes = F.normalize(self.env_prototypes.float(), dim=-1, eps=1e-8)
        temp = max(self._route_temperature(epoch), 1e-6)
        logits = pooled.matmul(prototypes.t()) / temp
        if self.fpem_force_uniform_route:
            q = logits.new_full(logits.shape, 1.0 / max(logits.shape[1], 1))
            mode = "uniform_fixed"
        elif training and epoch is not None and int(epoch) <= self.fpem_expert_uniform_warmup_epochs:
            q = logits.new_full(logits.shape, 1.0 / max(logits.shape[1], 1))
            mode = "uniform_warmup"
        elif self.fpem_use_sinkhorn_route and training:
            try:
                q = self._sinkhorn(logits)
                mode = "sinkhorn"
            except Exception as exc:
                if not self._sinkhorn_warned:
                    print(f"[StableST] WARNING sinkhorn failed, fallback softmax: {exc}", flush=True)
                    self._sinkhorn_warned = True
                q = torch.softmax(logits, dim=-1)
                mode = "softmax_fallback"
        else:
            q = torch.softmax(logits, dim=-1)
            mode = "softmax"
        return q.to(dtype=e_env.dtype), logits.to(dtype=e_env.dtype), mode

    def _uniform_route(self, ref):
        q = ref.new_full((ref.shape[0], self.fpem_env_route_k), 1.0 / max(self.fpem_env_route_k, 1))
        row_sum = q.sum(dim=-1)
        if not torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-6, rtol=1e-6):
            raise AssertionError("fixed uniform route rows must sum to one")
        logits = ref.new_zeros(ref.shape[0], self.fpem_env_route_k)
        return q, logits

    def _hard_prediction_environment_sinkhorn_enabled(self):
        return self.fpem_env_route_train_mode == "hard_prediction_environment_sinkhorn"

    def _hard_prediction_sinkhorn_enabled(self):
        return self.fpem_env_route_train_mode == "hard_prediction_sinkhorn"

    def _warmup_risk_sinkhorn_enabled(self):
        return self.fpem_env_route_train_mode == "warmup_risk_sinkhorn"

    def _progressive_gmm_environment_enabled(self):
        return self.fpem_env_route_train_mode == "progressive_gmm_environment"

    def _hard_sinkhorn_family_enabled(self):
        return (
            self._hard_prediction_sinkhorn_enabled()
            or self._hard_prediction_environment_sinkhorn_enabled()
            or self._warmup_risk_sinkhorn_enabled()
            or self._progressive_gmm_environment_enabled()
        )

    def _warmup_risk_stage(self, epoch):
        epoch = int(epoch or 0)
        if epoch <= max(int(self.fpem_sinkhorn_warmup_epochs), 0):
            return "warmup"
        if epoch <= max(int(self.fpem_sinkhorn_soft_end_epoch), int(self.fpem_sinkhorn_warmup_epochs)):
            return "soft"
        return "hard"

    def _warmup_risk_sinkhorn_temperature(self, epoch, ref):
        warm = max(int(self.fpem_sinkhorn_warmup_epochs), 0)
        end = max(int(self.fpem_sinkhorn_soft_end_epoch), warm)
        if epoch is None or end <= warm:
            progress = 1.0
        else:
            progress = min(max((float(epoch) - float(warm)) / float(max(end - warm, 1)), 0.0), 1.0)
        temp = self.fpem_sinkhorn_temperature_start + (
            self.fpem_sinkhorn_temperature_final - self.fpem_sinkhorn_temperature_start
        ) * progress
        return ref.new_tensor(max(float(temp), 1e-6)), ref.new_tensor(progress)

    def _risk_router_scores(self, route_features, ref_dtype=None):
        # Detach route features so risk loss updates only the risk router, not
        # the invariant/environment encoders or expert heads.
        risk = self.risk_router(route_features.detach().float())
        if ref_dtype is not None:
            risk = risk.to(dtype=ref_dtype)
        return risk

    def _risk_route_q_from_scores(self, predicted_risk, training=False, epoch=None):
        q = torch.softmax(
            -predicted_risk.float() / max(float(self.fpem_risk_router_temperature), 1e-6),
            dim=-1,
        ).to(dtype=predicted_risk.dtype)
        if training and self._warmup_risk_stage(epoch) == "warmup":
            q = predicted_risk.new_full(q.shape, 1.0 / max(q.shape[1], 1))
            mode = "risk_uniform_warmup"
        else:
            mode = "risk_softmax"
        logits = -predicted_risk
        return q, logits, mode

    def _risk_ranking_loss(self, predicted_risk, target_risk):
        zero = predicted_risk.new_zeros(())
        pair_losses = []
        correct = []
        for i in range(predicted_risk.shape[1]):
            for j in range(i + 1, predicted_risk.shape[1]):
                true_diff = (target_risk[:, j] - target_risk[:, i]).detach()
                pred_diff = predicted_risk[:, j] - predicted_risk[:, i]
                valid = true_diff.abs() > 1e-6
                if bool(valid.any()):
                    sign = true_diff[valid].sign()
                    pair_losses.append(F.softplus(-sign * pred_diff[valid]).mean())
                    correct.append(((pred_diff[valid] * true_diff[valid]) > 0).to(dtype=predicted_risk.dtype).mean())
        if not pair_losses:
            return zero, zero
        return torch.stack(pair_losses).mean(), torch.stack(correct).mean()

    def _progressive_partition_ready(self):
        return bool(self.progressive_gmm_initialized.detach().view(-1)[0].item() > 0.5)

    def _progressive_active_slots(self, device=None):
        if self._progressive_partition_ready():
            mask = self.progressive_active_expert_mask > 0.5
            slots = torch.nonzero(mask, as_tuple=False).flatten()
            if slots.numel() == 0:
                slots = torch.arange(max(1, self.fpem_env_max_clusters), device=self.progressive_active_expert_mask.device)
        else:
            slots = torch.arange(max(1, self.fpem_env_max_clusters), device=self.progressive_active_expert_mask.device)
        if device is not None:
            slots = slots.to(device=device)
        return slots.long()

    def _progressive_uniform_q(self, ref):
        q = ref.new_zeros(ref.shape[0], self.fpem_env_route_k)
        slots = self._progressive_active_slots(device=ref.device)
        q[:, slots] = 1.0 / float(max(int(slots.numel()), 1))
        return q

    def _progressive_normalize_embedding(self, embedding):
        mean = self.progressive_feature_mean.to(device=embedding.device, dtype=embedding.dtype)
        std = self.progressive_feature_std.to(device=embedding.device, dtype=embedding.dtype).clamp_min(1e-6)
        return (embedding - mean) / std

    def _progressive_gmm_logits(self, embedding):
        z = self._progressive_normalize_embedding(embedding).float()
        active_k = int(self.progressive_active_cluster_count.detach().item())
        active_k = max(1, min(active_k, self.fpem_env_max_clusters))
        mu = self.progressive_gmm_mu[:active_k].to(device=z.device, dtype=torch.float32)
        var = self.progressive_gmm_var[:active_k].to(device=z.device, dtype=torch.float32).clamp_min(
            float(self.fpem_env_gmm_variance_floor)
        )
        log_prior = self.progressive_gmm_log_prior[:active_k].to(device=z.device, dtype=torch.float32)
        diff = z[:, None, :] - mu[None, :, :]
        log_prob = -0.5 * (
            (diff.pow(2) / var[None, :, :]).sum(dim=-1)
            + var.log().sum(dim=-1).view(1, -1)
            + float(z.shape[-1]) * math.log(2.0 * math.pi)
        )
        return log_prob + log_prior.view(1, -1)

    def _progressive_route_from_embedding(self, embedding, ref, hard=True):
        if not self._progressive_partition_ready():
            q = self._progressive_uniform_q(ref)
            logits = ref.new_zeros(ref.shape[0], self.fpem_env_route_k)
            cluster_id = torch.zeros(ref.shape[0], dtype=torch.long, device=ref.device)
            return q, logits, cluster_id, "uniform_no_partition"
        logits_active = self._progressive_gmm_logits(embedding)
        prob_active = torch.softmax(logits_active, dim=-1).to(dtype=ref.dtype)
        cluster_id = logits_active.detach().argmax(dim=-1)
        mapping = self.progressive_cluster_to_expert.to(device=ref.device, dtype=torch.long)
        expert_id = mapping.index_select(0, cluster_id).clamp_min(0).clamp_max(self.fpem_env_route_k - 1)
        q = ref.new_zeros(ref.shape[0], self.fpem_env_route_k)
        if hard:
            q.scatter_(1, expert_id.view(-1, 1), 1.0)
        else:
            for cluster_idx in range(prob_active.shape[1]):
                slot = int(mapping[cluster_idx].detach().item())
                if 0 <= slot < self.fpem_env_route_k:
                    q[:, slot] = q[:, slot] + prob_active[:, cluster_idx]
            q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        logits = ref.new_full((ref.shape[0], self.fpem_env_route_k), -1.0e9)
        for cluster_idx in range(logits_active.shape[1]):
            slot = int(mapping[cluster_idx].detach().item())
            if 0 <= slot < self.fpem_env_route_k:
                logits[:, slot] = logits_active[:, cluster_idx].to(dtype=ref.dtype)
        return q, logits, cluster_id.to(device=ref.device), "progressive_gmm_teacher_hard"

    def _progressive_route_q(
        self,
        e_useful,
        training=False,
        epoch=None,
        sample_index=None,
        teacher_embedding=None,
    ):
        ref = e_useful.mean(dim=1)
        if training:
            if (
                sample_index is not None
                and self._progressive_partition_ready()
                and torch.is_tensor(getattr(self, "progressive_fixed_train_assignments", None))
            ):
                idx = sample_index.detach().long()
                assignment_cpu = self.progressive_fixed_train_assignments
                assignment = torch.full(idx.shape, -1, dtype=torch.long, device=idx.device)
                valid = (idx >= 0) & (idx < int(assignment_cpu.numel()))
                if bool(valid.any()):
                    gathered = assignment_cpu.to(device=idx.device).index_select(0, idx[valid])
                    assignment[valid] = gathered
                valid = assignment >= 0
                if bool(valid.any()):
                    mapping = self.progressive_cluster_to_expert.to(device=idx.device, dtype=torch.long)
                    expert_id = torch.zeros_like(assignment).clamp_min(0)
                    expert_id[valid] = mapping.index_select(0, assignment[valid]).clamp_min(0).clamp_max(
                        self.fpem_env_route_k - 1
                    )
                    q = self._progressive_uniform_q(ref)
                    q_valid = ref.new_zeros(int(valid.sum().item()), self.fpem_env_route_k)
                    q_valid.scatter_(1, expert_id[valid].view(-1, 1), 1.0)
                    q[valid] = q_valid
                    logits = ref.new_zeros(ref.shape[0], self.fpem_env_route_k)
                    return q.detach(), logits, assignment.to(device=ref.device), "progressive_fixed_train_assignment"
            q = self._progressive_uniform_q(ref)
            logits = ref.new_zeros(ref.shape[0], self.fpem_env_route_k)
            cluster_id = torch.zeros(ref.shape[0], dtype=torch.long, device=ref.device)
            return q.detach(), logits, cluster_id, "progressive_uniform_warmup"
        emb = teacher_embedding
        if emb is None:
            emb = self._progressive_pool_embedding(e_useful.detach())
        return self._progressive_route_from_embedding(emb, ref, hard=True)

    @staticmethod
    def _numpy_logsumexp(a, axis=1):
        m = np.max(a, axis=axis, keepdims=True)
        return m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True).clip(min=1e-300))

    def _fit_diag_gmm_numpy(self, x, k, rng):
        n, d = x.shape
        var_floor = max(float(self.fpem_env_gmm_variance_floor), 1e-8)
        if k <= 1:
            mu = x.mean(axis=0, keepdims=True)
            var = np.maximum(x.var(axis=0, keepdims=True), var_floor)
            log_prob = -0.5 * (
                ((x[:, None, :] - mu[None, :, :]) ** 2 / var[None, :, :]).sum(axis=-1)
                + np.log(var).sum(axis=-1)[None, :]
                + d * np.log(2.0 * np.pi)
            )
            ll = float(log_prob.sum())
            labels = np.zeros(n, dtype=np.int64)
            bic = -2.0 * ll + float((2 * d + 0) * np.log(max(n, 1)))
            return {"k": 1, "weights": np.ones(1), "means": mu, "vars": var, "labels": labels, "bic": bic, "loglik": ll}

        best = None
        n_init = max(1, int(self.fpem_env_gmm_n_init))
        for _ in range(n_init):
            replace = n < k
            init_idx = rng.choice(n, size=k, replace=replace)
            mu = x[init_idx].copy()
            overall_var = np.maximum(x.var(axis=0, keepdims=True), var_floor)
            var = np.repeat(overall_var, k, axis=0).copy()
            weights = np.full(k, 1.0 / float(k), dtype=np.float64)
            prev_ll = None
            for _em_iter in range(60):
                diff = x[:, None, :] - mu[None, :, :]
                log_prob = -0.5 * (
                    (diff * diff / var[None, :, :]).sum(axis=-1)
                    + np.log(var).sum(axis=-1)[None, :]
                    + d * np.log(2.0 * np.pi)
                ) + np.log(weights.clip(min=1e-12))[None, :]
                log_norm = self._numpy_logsumexp(log_prob, axis=1)
                ll = float(log_norm.sum())
                resp = np.exp(log_prob - log_norm)
                nk = resp.sum(axis=0).clip(min=1e-8)
                weights = nk / float(n)
                mu = (resp.T @ x) / nk[:, None]
                diff = x[:, None, :] - mu[None, :, :]
                var = (resp[:, :, None] * diff * diff).sum(axis=0) / nk[:, None]
                var = np.maximum(var, var_floor)
                if prev_ll is not None and abs(ll - prev_ll) < 1e-4 * (1.0 + abs(prev_ll)):
                    break
                prev_ll = ll
            labels = resp.argmax(axis=1).astype(np.int64)
            num_params = (k - 1) + 2 * k * d
            bic = -2.0 * ll + float(num_params * np.log(max(n, 1)))
            candidate = {
                "k": k,
                "weights": weights.copy(),
                "means": mu.copy(),
                "vars": var.copy(),
                "labels": labels.copy(),
                "bic": float(bic),
                "loglik": float(ll),
            }
            if best is None or candidate["bic"] < best["bic"]:
                best = candidate
        return best

    def _best_overlap_match(self, overlap):
        old_k, new_k = overlap.shape
        new_to_old = [-1 for _ in range(new_k)]
        best_score = 0
        if old_k <= 0 or new_k <= 0:
            return new_to_old, best_score
        if new_k <= old_k:
            best = None
            for old_perm in itertools.permutations(range(old_k), new_k):
                score = sum(int(overlap[old_perm[j], j]) for j in range(new_k))
                if best is None or score > best[0]:
                    best = (score, old_perm)
            if best is not None:
                best_score, old_perm = best
                for j, old_idx in enumerate(old_perm):
                    if overlap[old_idx, j] > 0:
                        new_to_old[j] = int(old_idx)
        else:
            best = None
            for new_perm in itertools.permutations(range(new_k), old_k):
                score = sum(int(overlap[i, new_perm[i]]) for i in range(old_k))
                if best is None or score > best[0]:
                    best = (score, new_perm)
            if best is not None:
                best_score, new_perm = best
                for old_idx, new_idx in enumerate(new_perm):
                    if overlap[old_idx, new_idx] > 0:
                        new_to_old[int(new_idx)] = int(old_idx)
        return new_to_old, int(best_score)

    def _copy_expert_parameters(self, parent_expert, child_expert):
        parent = int(parent_expert)
        child = int(child_expert)
        if parent == child or parent < 0 or child < 0:
            return
        copied = 0
        module_pairs = [
            (getattr(self.hyper_inv_heads, "hypernets", None), getattr(self.hyper_inv_heads, "alpha_gates", None)),
            (getattr(self.hyper_concat_input_heads, "hypernets", None), getattr(self.hyper_concat_input_heads, "alpha_gates", None)),
        ]
        with torch.no_grad():
            for hypernets, gates in module_pairs:
                for modules in (hypernets, gates):
                    if modules is None or parent >= len(modules) or child >= len(modules):
                        continue
                    modules[child].load_state_dict(copy.deepcopy(modules[parent].state_dict()))
                    for param in modules[child].parameters():
                        param.add_(torch.randn_like(param) * float(self.fpem_env_split_perturb_std))
                    copied += 1
            heads = getattr(self.route_heads, "heads", None)
            if heads is not None and parent < len(heads) and child < len(heads):
                heads[child].load_state_dict(copy.deepcopy(heads[parent].state_dict()))
                for param in heads[child].parameters():
                    param.add_(torch.randn_like(param) * float(self.fpem_env_split_perturb_std))
                copied += 1
        return copied

    def _align_and_apply_progressive_partition(self, labels, selected, indices, epoch, raw_candidate_logs, start_time):
        labels = np.asarray(labels, dtype=np.int64)
        indices = np.asarray(indices, dtype=np.int64)
        selected_k = int(selected["k"])
        previous = None
        if torch.is_tensor(getattr(self, "progressive_fixed_train_assignments", None)):
            previous = self.progressive_fixed_train_assignments.detach().cpu().numpy()
        old_k = int(self.progressive_active_cluster_count.detach().item()) if self._progressive_partition_ready() else 0
        old_mapping = self.progressive_cluster_to_expert.detach().cpu().numpy().astype(np.int64)
        new_to_old = [-1 for _ in range(selected_k)]
        overlap_score = 0
        change_ratio = 1.0
        overlap = np.zeros((max(old_k, 1), max(selected_k, 1)), dtype=np.int64)
        max_index = int(indices.max()) if indices.size > 0 else 0
        if previous is not None and old_k > 0 and indices.size > 0 and previous.size > max_index:
            old_labels = previous[indices]
            valid = (old_labels >= 0) & (old_labels < old_k)
            if valid.any():
                overlap = np.zeros((old_k, selected_k), dtype=np.int64)
                for old_label, new_label in zip(old_labels[valid], labels[valid]):
                    overlap[int(old_label), int(new_label)] += 1
                new_to_old, overlap_score = self._best_overlap_match(overlap)

        new_mapping = np.full(self.fpem_env_route_k, -1, dtype=np.int64)
        used_experts = set()
        for new_cluster, old_cluster in enumerate(new_to_old):
            if old_cluster >= 0:
                expert = int(old_mapping[old_cluster])
                if 0 <= expert < self.fpem_env_route_k and expert not in used_experts:
                    new_mapping[new_cluster] = expert
                    used_experts.add(expert)
        unused_experts = [idx for idx in range(self.fpem_env_route_k) if idx not in used_experts]
        copied_splits = []
        for new_cluster in range(selected_k):
            if new_mapping[new_cluster] >= 0:
                continue
            expert = unused_experts.pop(0) if unused_experts else new_cluster
            parent_old = -1
            parent_expert = -1
            if overlap.shape[0] > 0 and new_cluster < overlap.shape[1] and overlap[:, new_cluster].sum() > 0:
                parent_old = int(overlap[:, new_cluster].argmax())
                parent_expert = int(old_mapping[parent_old]) if 0 <= parent_old < len(old_mapping) else -1
                if 0 <= parent_expert < self.fpem_env_route_k:
                    copied = self._copy_expert_parameters(parent_expert, expert)
                    copied_splits.append({"new_cluster": new_cluster, "parent_cluster": parent_old, "parent_expert": parent_expert, "new_expert": int(expert), "copied_modules": int(copied or 0)})
            new_mapping[new_cluster] = expert
            used_experts.add(int(expert))

        assignment_len = max_index + 1 if indices.size else 0
        if previous is not None:
            assignment_len = max(assignment_len, int(previous.shape[0]))
        assignments = np.full(assignment_len, -1, dtype=np.int64)
        assignments[indices] = labels
        if previous is not None and previous.size == assignments.size:
            valid = (previous >= 0) & (assignments >= 0)
            if valid.any():
                change_ratio = float((previous[valid] != assignments[valid]).mean())

        mean = selected["feature_mean"].astype(np.float32)
        std = selected["feature_std"].astype(np.float32)
        means = selected["means"].astype(np.float32)
        variances = selected["vars"].astype(np.float32)
        weights = selected["weights"].astype(np.float32).clip(min=1e-12)
        active_mask = np.zeros(self.fpem_env_route_k, dtype=np.float32)
        for cluster_idx in range(selected_k):
            expert = int(new_mapping[cluster_idx])
            if 0 <= expert < self.fpem_env_route_k:
                active_mask[expert] = 1.0

        device = self.progressive_gmm_mu.device
        with torch.no_grad():
            self.progressive_feature_mean.copy_(torch.as_tensor(mean, device=device, dtype=self.progressive_feature_mean.dtype))
            self.progressive_feature_std.copy_(torch.as_tensor(std, device=device, dtype=self.progressive_feature_std.dtype))
            self.progressive_gmm_mu.zero_()
            self.progressive_gmm_var.fill_(1.0)
            self.progressive_gmm_log_prior.fill_(-1.0e9)
            self.progressive_gmm_mu[:selected_k].copy_(torch.as_tensor(means, device=device, dtype=self.progressive_gmm_mu.dtype))
            self.progressive_gmm_var[:selected_k].copy_(torch.as_tensor(variances, device=device, dtype=self.progressive_gmm_var.dtype))
            self.progressive_gmm_log_prior[:selected_k].copy_(torch.as_tensor(np.log(weights), device=device, dtype=self.progressive_gmm_log_prior.dtype))
            self.progressive_cluster_to_expert.fill_(-1)
            self.progressive_cluster_to_expert[:selected_k].copy_(torch.as_tensor(new_mapping[:selected_k], device=device, dtype=torch.long))
            self.progressive_active_expert_mask.copy_(torch.as_tensor(active_mask, device=device, dtype=self.progressive_active_expert_mask.dtype))
            self.progressive_active_cluster_count.fill_(selected_k)
            self.progressive_last_partition_epoch.fill_(int(epoch))
            self.progressive_gmm_initialized.fill_(1.0)
        self.progressive_fixed_train_assignments = torch.as_tensor(assignments, dtype=torch.long)

        counts = np.bincount(labels, minlength=selected_k).astype(np.int64)
        ratios = counts / max(float(len(labels)), 1.0)
        entropy = float(-(ratios[ratios > 0] * np.log(ratios[ratios > 0])).sum())
        info = {
            "epoch": int(epoch),
            "selected_k": selected_k,
            "bic_by_k": raw_candidate_logs,
            "cluster_sizes": counts.tolist(),
            "cluster_ratios": ratios.tolist(),
            "cluster_assignment_entropy": entropy,
            "assignment_change_ratio": float(change_ratio),
            "hungarian_overlap_score": int(overlap_score),
            "active_expert_count": int(active_mask.sum()),
            "cluster_to_expert_mapping": new_mapping[:selected_k].astype(int).tolist(),
            "copied_splits": copied_splits,
            "gmm_means": means.tolist(),
            "gmm_variances": variances.tolist(),
            "partition_update_time_sec": float(time.time() - start_time),
            "assignment_source": "ema_teacher_encoder_only_no_target_no_prediction_error",
        }
        self._latest_progressive_partition_logs = info
        self.progressive_partition_history.append(info)
        return info

    def _select_progressive_gmm(self, embeddings, epoch):
        x_raw = np.asarray(embeddings, dtype=np.float64)
        feature_mean = x_raw.mean(axis=0)
        feature_std = x_raw.std(axis=0)
        feature_std = np.maximum(feature_std, 1e-6)
        x = (x_raw - feature_mean[None, :]) / feature_std[None, :]
        rng = np.random.default_rng(int(getattr(self.args, "seed", 2024)) + int(epoch) * 9973)
        candidates = []
        best = None
        n = x.shape[0]
        max_k = max(1, min(int(self.fpem_env_max_clusters), int(self.fpem_env_route_k), n))
        for k in range(1, max_k + 1):
            candidate = self._fit_diag_gmm_numpy(x, k, rng)
            counts = np.bincount(candidate["labels"], minlength=k)
            ratios = counts / max(float(n), 1.0)
            min_ratio = float(ratios.min()) if ratios.size else 1.0
            rejected = bool(k > 1 and min_ratio < float(self.fpem_env_min_cluster_ratio))
            bic_raw = float(candidate["bic"])
            bic_score = float("inf") if rejected else bic_raw
            candidates.append({
                "k": int(k),
                "bic": bic_raw,
                "bic_score": bic_score,
                "rejected_min_cluster": rejected,
                "cluster_sizes": counts.astype(int).tolist(),
                "cluster_ratios": ratios.tolist(),
            })
            if not rejected and (best is None or bic_score < best["bic_score"]):
                candidate["bic_score"] = bic_score
                best = candidate
        if best is None:
            best = self._fit_diag_gmm_numpy(x, 1, rng)
            best["bic_score"] = float(best["bic"])
        best["feature_mean"] = feature_mean
        best["feature_std"] = feature_std
        return best, candidates

    def progressive_should_update_partition(self, epoch, total_epochs):
        if not self._progressive_gmm_environment_enabled():
            return False
        epoch = int(epoch)
        total_epochs = int(total_epochs)
        if epoch < int(self.fpem_env_partition_start_epoch):
            return False
        if int(self.fpem_env_partition_freeze_last_epochs) > 0:
            first_frozen_epoch = total_epochs - int(self.fpem_env_partition_freeze_last_epochs) + 1
            if epoch >= first_frozen_epoch:
                return False
        last = int(self.progressive_last_partition_epoch.detach().item())
        if last <= 0:
            return True
        interval = max(1, int(self.fpem_env_partition_update_interval))
        return (epoch - last) >= interval

    def progressive_extract_teacher_embeddings(self, loader, device, max_batches=None):
        was_training = self.training
        self.eval()
        embs = []
        indices = []
        with torch.no_grad():
            for batch_idx, raw_batch in enumerate(loader):
                if max_batches is not None and int(max_batches) >= 0 and batch_idx >= int(max_batches):
                    break
                batch = tuple(t.to(device, non_blocking=True) for t in raw_batch)
                if len(batch) == 5:
                    x, _target, time_label, c, sample_index = batch
                elif len(batch) == 4:
                    x, _target, time_label, c = batch
                    sample_index = torch.arange(x.shape[0], device=x.device)
                else:
                    x, _target, c = batch
                    time_label = None
                    sample_index = torch.arange(x.shape[0], device=x.device)
                x_env, _logs = self._apply_env_exogenous(x, time_label=time_label, exog=c)
                seq = self.encoder_env_teacher(x_env.detach())
                nodes = self._temporal_pool_env(seq)
                emb = self._progressive_pool_embedding(nodes)
                embs.append(emb.detach().cpu())
                indices.append(sample_index.detach().long().cpu())
        if was_training:
            self.train(True)
        if not embs:
            return None, None
        return torch.cat(embs, dim=0).numpy(), torch.cat(indices, dim=0).numpy()

    def maybe_update_progressive_gmm_partition(self, loader, epoch, total_epochs, device, max_batches=None):
        if not self.progressive_should_update_partition(epoch, total_epochs):
            return None
        start_time = time.time()
        embeddings, indices = self.progressive_extract_teacher_embeddings(loader, device, max_batches=max_batches)
        if embeddings is None or indices is None or embeddings.shape[0] == 0:
            return None
        selected, candidate_logs = self._select_progressive_gmm(embeddings, epoch)
        return self._align_and_apply_progressive_partition(
            selected["labels"],
            selected,
            indices,
            epoch,
            candidate_logs,
            start_time,
        )

    def get_progressive_gmm_state_for_checkpoint(self):
        return {
            "fixed_train_assignments": (
                self.progressive_fixed_train_assignments.detach().cpu()
                if torch.is_tensor(getattr(self, "progressive_fixed_train_assignments", None))
                else None
            ),
            "partition_history": list(getattr(self, "progressive_partition_history", [])),
            "latest_partition_logs": dict(getattr(self, "_latest_progressive_partition_logs", {}) or {}),
        }

    def load_progressive_gmm_state_from_checkpoint(self, state):
        if not isinstance(state, dict):
            return
        assignments = state.get("fixed_train_assignments")
        if assignments is not None:
            self.progressive_fixed_train_assignments = torch.as_tensor(assignments, dtype=torch.long).detach().cpu()
        self.progressive_partition_history = list(state.get("partition_history", []) or [])
        self._latest_progressive_partition_logs = dict(state.get("latest_partition_logs", {}) or {})

    @staticmethod
    def _row_normalize_cost(cost):
        if cost.shape[-1] <= 1:
            return torch.zeros_like(cost)
        centered = cost - cost.mean(dim=1, keepdim=True)
        return centered / centered.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _sinkhorn_from_scores(scores, num_iters=5):
        with torch.no_grad():
            bsz, num_experts = scores.shape
            scores = scores.float()
            scores = scores - scores.max(dim=1, keepdim=True)[0]
            q = torch.exp(scores).clamp_min(1e-12)
            q = q / q.sum().clamp_min(1e-12)
            for _ in range(max(1, int(num_iters))):
                q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)
                q = q / float(max(bsz, 1))
                q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
                q = q / float(max(num_experts, 1))
            q = q * float(max(bsz, 1))
            q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return q

    @staticmethod
    def _balanced_hard_idx_from_soft(soft_q):
        with torch.no_grad():
            bsz, num_experts = soft_q.shape
            hard_idx = soft_q.argmax(dim=-1)
            if num_experts <= 1 or bsz < num_experts:
                return hard_idx, soft_q.new_zeros(())

            target_min = bsz // num_experts
            counts = torch.bincount(hard_idx, minlength=num_experts)
            if bool((counts >= target_min).all()):
                return hard_idx, soft_q.new_zeros(())

            hard_idx = hard_idx.clone()
            order = torch.sort(soft_q.reshape(-1), descending=True)[1]
            used = torch.zeros(bsz, dtype=torch.bool, device=soft_q.device)
            counts = torch.zeros(num_experts, dtype=torch.long, device=soft_q.device)
            for flat_idx in order:
                sample = flat_idx // num_experts
                expert = flat_idx - sample * num_experts
                if bool(used[sample]):
                    continue
                if counts[expert] >= target_min and int(counts.sum().item()) < target_min * num_experts:
                    continue
                hard_idx[sample] = expert
                used[sample] = True
                counts[expert] += 1
                if bool(used.all()):
                    break
            if not bool(used.all()):
                hard_idx[~used] = soft_q[~used].argmax(dim=-1)
            repaired = (hard_idx != soft_q.argmax(dim=-1)).to(dtype=soft_q.dtype).mean()
        return hard_idx, repaired

    def _hard_sinkhorn_from_cost(self, cost):
        with torch.no_grad():
            scores = -cost.detach().float() / max(float(getattr(self.args, "fpem_env_route_sinkhorn_tau", 1.0)), 1e-6)
            soft_q = self._sinkhorn_from_scores(
                scores,
                num_iters=int(getattr(self.args, "fpem_env_route_sinkhorn_iters", 5)),
            ).to(dtype=cost.dtype)
            hard_idx, repaired = self._balanced_hard_idx_from_soft(soft_q)
            hard_q = F.one_hot(hard_idx, num_classes=cost.shape[1]).to(device=cost.device, dtype=cost.dtype)
        return soft_q.detach(), hard_idx.detach(), hard_q.detach(), repaired.detach()

    def _observable_route_features(self, z_inv, e_useful, y_heads):
        bsz = z_inv.shape[0]
        z_float = z_inv.detach().float()
        e_float = e_useful.detach().float()
        z_mean = z_float.mean(dim=1)
        z_std = z_float.std(dim=1, unbiased=False)
        e_mean = e_float.mean(dim=1)
        e_std = e_float.std(dim=1, unbiased=False)

        aux = getattr(self, "_latest_route_aux_stats", None)
        if not torch.is_tensor(aux) or aux.shape[0] != bsz:
            aux = z_inv.new_zeros(bsz, 12).float()
        else:
            aux = aux.to(device=z_inv.device).float()
        if aux.shape[1] < 12:
            aux = torch.cat([aux, aux.new_zeros(bsz, 12 - aux.shape[1])], dim=-1)
        elif aux.shape[1] > 12:
            aux = aux[:, :12]

        heads = y_heads.detach().float().reshape(bsz, y_heads.shape[1], -1)
        if heads.shape[1] > 1:
            expert_var = heads.var(dim=1, unbiased=False).mean(dim=1)
            expert_range = (heads.max(dim=1)[0] - heads.min(dim=1)[0]).mean(dim=1)
            pairwise = []
            for i in range(heads.shape[1]):
                for j in range(i + 1, heads.shape[1]):
                    pairwise.append((heads[:, i] - heads[:, j]).abs().mean(dim=1))
            expert_pairwise = torch.stack(pairwise, dim=0).mean(dim=0) if pairwise else expert_var.new_zeros(bsz)
        else:
            expert_var = heads.new_zeros(bsz)
            expert_range = heads.new_zeros(bsz)
            expert_pairwise = heads.new_zeros(bsz)
        disagreement = torch.stack([expert_var, expert_range, expert_pairwise], dim=-1)

        features = torch.cat([z_mean, z_std, e_mean, e_std, aux, disagreement], dim=-1)
        if features.shape[1] < self.fpem_route_observable_dim:
            pad = features.new_zeros(bsz, self.fpem_route_observable_dim - features.shape[1])
            features = torch.cat([features, pad], dim=-1)
        elif features.shape[1] > self.fpem_route_observable_dim:
            features = features[:, : self.fpem_route_observable_dim]
        return features.detach().to(dtype=z_inv.dtype)

    def _gaussian_ready(self):
        ready = getattr(self, "env_route_gaussian_initialized", None)
        return torch.is_tensor(ready) and bool((ready.detach().float() > 0.5).any())

    def _init_env_route_gaussian_from_features(self, route_features):
        with torch.no_grad():
            features = route_features.detach().float()
            bsz, dim = features.shape
            if dim != self.fpem_route_observable_dim:
                return
            if bsz <= 0:
                return
            order = torch.sort(features[:, 0])[1]
            labels_sorted = torch.arange(bsz, device=features.device, dtype=torch.long)
            labels_sorted = torch.clamp(labels_sorted * self.fpem_env_route_k // max(bsz, 1), max=self.fpem_env_route_k - 1)
            labels = torch.zeros_like(labels_sorted)
            labels[order] = labels_sorted
            mu = self.env_route_gaussian_mu.detach().to(device=features.device, dtype=torch.float32).clone()
            var = self.env_route_gaussian_var.detach().to(device=features.device, dtype=torch.float32).clone()
            mass = self.env_route_gaussian_mass.detach().to(device=features.device, dtype=torch.float32).clone()
            global_mean = features.mean(dim=0)
            global_var = features.var(dim=0, unbiased=False).clamp_min(self.fpem_env_sinkhorn_gaussian_min_var)
            for k in range(self.fpem_env_route_k):
                mask = labels == k
                if bool(mask.any()):
                    part = features[mask]
                    mu[k] = part.mean(dim=0)
                    var[k] = part.var(dim=0, unbiased=False).clamp_min(self.fpem_env_sinkhorn_gaussian_min_var)
                    mass[k] = float(part.shape[0])
                else:
                    mu[k] = global_mean
                    var[k] = global_var
                    mass[k] = 1.0
            self.env_route_gaussian_mu.copy_(mu.to(device=self.env_route_gaussian_mu.device, dtype=self.env_route_gaussian_mu.dtype))
            self.env_route_gaussian_var.copy_(var.to(device=self.env_route_gaussian_var.device, dtype=self.env_route_gaussian_var.dtype))
            self.env_route_gaussian_mass.copy_(mass.to(device=self.env_route_gaussian_mass.device, dtype=self.env_route_gaussian_mass.dtype))
            self.env_route_gaussian_initialized.fill_(1.0)

    def _gaussian_cost_logits_q(self, route_features):
        features = route_features.detach().float()
        mu = self.env_route_gaussian_mu.to(device=features.device, dtype=torch.float32)
        var = self.env_route_gaussian_var.to(device=features.device, dtype=torch.float32).clamp_min(
            self.fpem_env_sinkhorn_gaussian_min_var
        )
        mass = self.env_route_gaussian_mass.to(device=features.device, dtype=torch.float32).clamp_min(1e-6)
        diff = features[:, None, :] - mu[None, :, :]
        cost = (diff.pow(2) / var[None, :, :]).mean(dim=-1)
        prior = torch.log(mass / mass.sum().clamp_min(1e-6)).view(1, -1)
        logits = -cost + prior
        q = torch.softmax(logits, dim=-1).to(dtype=route_features.dtype)
        return cost.to(dtype=route_features.dtype), logits.to(dtype=route_features.dtype), q

    def _update_env_route_gaussian_ema(self, route_features, hard_q):
        if route_features.numel() == 0:
            return
        with torch.no_grad():
            if not self._gaussian_ready():
                self._init_env_route_gaussian_from_features(route_features)
            features = route_features.detach().float()
            q = hard_q.detach().float()
            momentum = max(0.0, min(1.0, self.fpem_env_sinkhorn_gaussian_ema))
            if momentum <= 0.0:
                return
            mu = self.env_route_gaussian_mu.detach().to(device=features.device, dtype=torch.float32).clone()
            var = self.env_route_gaussian_var.detach().to(device=features.device, dtype=torch.float32).clone()
            mass = self.env_route_gaussian_mass.detach().to(device=features.device, dtype=torch.float32).clone()
            hard_idx = q.argmax(dim=-1)
            for k in range(self.fpem_env_route_k):
                mask = hard_idx == k
                if not bool(mask.any()):
                    continue
                part = features[mask]
                batch_mu = part.mean(dim=0)
                batch_var = part.var(dim=0, unbiased=False).clamp_min(self.fpem_env_sinkhorn_gaussian_min_var)
                mu[k] = (1.0 - momentum) * mu[k] + momentum * batch_mu
                var[k] = (1.0 - momentum) * var[k] + momentum * batch_var
                mass[k] = (1.0 - momentum) * mass[k] + momentum * float(part.shape[0])
            if hard_idx.numel() > 1:
                counts = self.env_route_transition_counts.detach().to(device=features.device, dtype=torch.float32).clone()
                for prev, cur in zip(hard_idx[:-1], hard_idx[1:]):
                    counts[prev, cur] += 1.0
                self.env_route_transition_counts.copy_(
                    counts.to(device=self.env_route_transition_counts.device, dtype=self.env_route_transition_counts.dtype)
                )
            self.env_route_gaussian_mu.copy_(mu.to(device=self.env_route_gaussian_mu.device, dtype=self.env_route_gaussian_mu.dtype))
            self.env_route_gaussian_var.copy_(var.to(device=self.env_route_gaussian_var.device, dtype=self.env_route_gaussian_var.dtype))
            self.env_route_gaussian_mass.copy_(mass.to(device=self.env_route_gaussian_mass.device, dtype=self.env_route_gaussian_mass.dtype))

    def _batch_temporal_cost(self, gaussian_idx, ref):
        cost = ref.new_zeros(ref.shape)
        if self.fpem_env_sinkhorn_temporal_lambda <= 0.0 or gaussian_idx.numel() <= 1:
            return cost, ref.new_zeros(())
        prev_idx = torch.cat([gaussian_idx[:1], gaussian_idx[:-1]], dim=0)
        cost = ref.new_ones(ref.shape)
        cost.scatter_(1, prev_idx.view(-1, 1), 0.0)
        cost[0].zero_()
        valid = ref.new_tensor(float(max(int(gaussian_idx.numel()) - 1, 0)) / float(max(int(gaussian_idx.numel()), 1)))
        return cost, valid

    def _env_route_inference_q(self, q_prediction, logits_prediction, route_features, e_useful, training, epoch):
        mode = self.fpem_env_route_inference_mode
        if mode in {"gaussian", "gaussian_viterbi"}:
            if self._gaussian_ready():
                _cost, logits, q_soft = self._gaussian_cost_logits_q(route_features)
                hard_idx = logits.detach().argmax(dim=-1)
                q = F.one_hot(hard_idx, num_classes=self.fpem_env_route_k).to(
                    device=route_features.device,
                    dtype=route_features.dtype,
                )
                return q, logits, q_soft.detach(), "gaussian" if mode == "gaussian" else "gaussian_viterbi_batchlocal"
            return q_prediction, logits_prediction, q_prediction.detach(), "mlp_fallback_gaussian_not_ready"
        if mode == "nearest_prototype":
            proto_q, proto_logits, proto_mode = self._prototype_route(e_useful, training=False, epoch=epoch)
            hard_idx = proto_q.detach().argmax(dim=-1)
            q = F.one_hot(hard_idx, num_classes=self.fpem_env_route_k).to(device=proto_q.device, dtype=proto_q.dtype)
            return q, proto_logits, proto_q.detach(), "nearest_prototype_" + proto_mode
        return q_prediction, logits_prediction, q_prediction.detach(), "mlp"

    def _env_sinkhorn_schedule(self, epoch, ref):
        start = float(self.fpem_env_sinkhorn_schedule_start_epoch)
        end = float(max(self.fpem_env_sinkhorn_schedule_end_epoch, self.fpem_env_sinkhorn_schedule_start_epoch))
        if epoch is None:
            progress = 1.0
        elif end <= start:
            progress = 1.0 if float(epoch) >= start else 0.0
        else:
            progress = min(max((float(epoch) - start) / (end - start), 0.0), 1.0)
        alpha = self.fpem_env_sinkhorn_prediction_alpha_start + (
            self.fpem_env_sinkhorn_prediction_alpha_final - self.fpem_env_sinkhorn_prediction_alpha_start
        ) * progress
        beta = self.fpem_env_sinkhorn_environment_beta_start + (
            self.fpem_env_sinkhorn_environment_beta_final - self.fpem_env_sinkhorn_environment_beta_start
        ) * progress
        return ref.new_tensor(alpha), ref.new_tensor(beta), ref.new_tensor(progress)

    def _prediction_eval_tensor(self, tensor, scaler):
        if scaler is None:
            return tensor
        return scaler.inverse_transform(tensor)

    def _per_sample_oracle_from_heads(self, expert_preds, target, scaler=None):
        """True per-sample hard-routing oracle over complete expert predictions.

        expert_preds: [B, K, ...]
        target:       [B, ...]

        The expert id is chosen once per sample after reducing all prediction
        dimensions.  This avoids the invalid element-wise/node-wise oracle that
        can make oracle diagnostics incomparable with hard-routing predictions.
        """
        preds_eval = self._prediction_eval_tensor(expert_preds, scaler)
        target_eval = self._prediction_eval_tensor(target, scaler)
        err = torch.abs(preds_eval - target_eval.unsqueeze(1))
        reduce_dims = tuple(range(2, err.ndim))
        per_sample_expert_mae = err.mean(dim=reduce_dims) if reduce_dims else err
        oracle_expert_id = per_sample_expert_mae.argmin(dim=1)
        batch_indices = torch.arange(expert_preds.shape[0], device=expert_preds.device)
        oracle_pred = preds_eval[batch_indices, oracle_expert_id]
        oracle_hard_mae = torch.abs(oracle_pred - target_eval).mean()
        return oracle_hard_mae, oracle_expert_id.detach(), oracle_pred, per_sample_expert_mae

    def _route_prediction_mae(self, prediction, target, scaler=None):
        pred_eval = self._prediction_eval_tensor(prediction, scaler)
        target_eval = self._prediction_eval_tensor(target, scaler)
        return torch.abs(pred_eval - target_eval).mean()

    def _warn_if_invalid_oracle(self, oracle_hard_mae, routed_mae, context):
        try:
            if bool((oracle_hard_mae.detach() > routed_mae.detach() + 1e-5).item()):
                warnings.warn(
                    "Invalid oracle result in {}: oracle hard MAE is larger than "
                    "the evaluated hard-routing MAE.".format(context),
                    RuntimeWarning,
                )
        except Exception:
            pass

    def _hard_prediction_environment_sinkhorn_loss(self, output, target, scaler, training, epoch):
        y_heads = output["y_hyper_heads"]
        route_features = output.get("route_features")
        if route_features is None:
            route_features = self._observable_route_features(output["Z_inv"], output["E_useful"], y_heads)
        loss_head = head_prediction_losses(y_heads, target, scaler, getattr(self.args, "yita", 0.5))
        prediction_cost = self._row_normalize_cost(loss_head.detach().float()).to(dtype=loss_head.dtype)
        if not self._gaussian_ready():
            self._init_env_route_gaussian_from_features(route_features)
        gaussian_cost_raw, gaussian_logits, gaussian_q_soft = self._gaussian_cost_logits_q(route_features)
        environment_cost = self._row_normalize_cost(gaussian_cost_raw.detach().float()).to(dtype=loss_head.dtype)
        gaussian_idx = gaussian_logits.detach().argmax(dim=-1)
        temporal_cost, temporal_valid = self._batch_temporal_cost(gaussian_idx, prediction_cost)
        alpha, beta, schedule_progress = self._env_sinkhorn_schedule(epoch, prediction_cost)
        combined_cost = alpha * prediction_cost + beta * environment_cost
        combined_cost = combined_cost + float(self.fpem_env_sinkhorn_temporal_lambda) * temporal_cost
        soft_q, hard_idx, hard_q, repaired = self._hard_sinkhorn_from_cost(combined_cost)
        selected_prediction_loss = (hard_q * loss_head).sum(dim=1).mean()
        if training:
            self._update_env_route_gaussian_ema(route_features, hard_q)

        router_q = output.get("env_route_q_prediction")
        if router_q is not None and torch.is_tensor(router_q) and router_q.shape == hard_q.shape:
            router_idx = router_q.detach().argmax(dim=-1)
            router_assignment_accuracy = (router_idx == hard_idx).to(dtype=loss_head.dtype).mean()
        else:
            router_assignment_accuracy = loss_head.new_zeros(())

        q_for_usage = hard_q.detach().float()
        usage = q_for_usage.mean(dim=0)
        entropy = -(soft_q.float().clamp(1e-8, 1.0) * soft_q.float().clamp(1e-8, 1.0).log()).sum(dim=1).mean()
        eff = torch.exp(-(usage.clamp_min(1e-8) * usage.clamp_min(1e-8).log()).sum())
        oracle_hard_mae, _oracle_idx, _oracle_pred, _oracle_per_head = self._per_sample_oracle_from_heads(
            y_heads.detach(), target.detach(), scaler
        )
        hard_routed_pred = (hard_q.view(hard_q.shape[0], hard_q.shape[1], 1, 1, 1) * y_heads.detach()).sum(dim=1)
        hard_routed_mae = self._route_prediction_mae(hard_routed_pred, target.detach(), scaler)
        self._warn_if_invalid_oracle(
            oracle_hard_mae,
            hard_routed_mae,
            "hard_prediction_environment_sinkhorn",
        )
        logs = {
            "fpem/env_route_loss": selected_prediction_loss.detach(),
            "fpem/env_route_L_final": selected_prediction_loss.detach(),
            "fpem/env_route_L_global": loss_head.min(dim=1)[0].mean().detach(),
            "fpem/env_route_L_route_soft": loss_head.new_zeros(()),
            "fpem/env_route_L_expert": selected_prediction_loss.detach(),
            "fpem/env_route_L_router_oracle": loss_head.new_zeros(()),
            "fpem/env_route_L_balance": loss_head.new_zeros(()),
            "fpem/env_route_L_diverse": loss_head.new_zeros(()),
            "fpem/env_route_L_proto_align": loss_head.new_zeros(()),
            "fpem/env_route_entropy": entropy.detach().to(dtype=loss_head.dtype),
            "fpem/env_route_train_mode_hard_prediction_environment_sinkhorn": loss_head.new_tensor(1.0),
            "fpem/hard_sinkhorn_enabled": loss_head.new_tensor(float(training)),
            "fpem/sinkhorn_soft_entropy": entropy.detach().to(dtype=loss_head.dtype),
            "fpem/sinkhorn_selected_loss": selected_prediction_loss.detach(),
            "fpem/hard_sinkhorn_balance_repaired": repaired.detach().to(dtype=loss_head.dtype),
            "fpem/sinkhorn_soft_row_sum_mean": soft_q.sum(dim=1).mean().detach().to(dtype=loss_head.dtype),
            "fpem/router_assignment_accuracy": router_assignment_accuracy.detach(),
            "fpem/router_assignment_agreement": router_assignment_accuracy.detach(),
            "fpem/oracle_hard_mae": oracle_hard_mae.detach().to(dtype=loss_head.dtype),
            "fpem/router_hard_mae": hard_routed_mae.detach().to(dtype=loss_head.dtype),
            "fpem/router_regret": (hard_routed_mae.detach() - oracle_hard_mae.detach()).to(dtype=loss_head.dtype),
            "fpem/env_sinkhorn_prediction_alpha": alpha.detach(),
            "fpem/env_sinkhorn_environment_beta": beta.detach(),
            "fpem/env_sinkhorn_schedule_progress": schedule_progress.detach(),
            "fpem/env_sinkhorn_temporal_lambda": loss_head.new_tensor(float(self.fpem_env_sinkhorn_temporal_lambda)),
            "fpem/env_sinkhorn_temporal_valid_fraction": temporal_valid.detach(),
            "fpem/env_sinkhorn_prediction_cost_mean": prediction_cost.detach().mean(),
            "fpem/env_sinkhorn_environment_cost_mean": environment_cost.detach().mean(),
            "fpem/env_sinkhorn_temporal_cost_mean": temporal_cost.detach().mean(),
            "fpem/env_sinkhorn_gaussian_initialized": loss_head.new_tensor(float(self._gaussian_ready())),
            "fpem/env_sinkhorn_gaussian_feature_dim": loss_head.new_tensor(float(self.fpem_route_observable_dim)),
            "fpem/env_route_inference_mode_gaussian": loss_head.new_tensor(
                float(self.fpem_env_route_inference_mode in {"gaussian", "gaussian_viterbi"})
            ),
            "fpem/effective_expert_number": eff.detach().to(dtype=loss_head.dtype),
            "fpem/max_expert_usage_ratio": usage.max().detach().to(dtype=loss_head.dtype),
            "fpem/min_expert_usage_ratio": usage.min().detach().to(dtype=loss_head.dtype),
        }
        for idx in range(self.fpem_env_route_k):
            logs[f"fpem/route_soft_mean_expert_{idx}"] = usage[idx].detach().to(dtype=loss_head.dtype)
            logs[f"fpem/route_hard_count_expert_{idx}"] = hard_q[:, idx].detach().sum().to(dtype=loss_head.dtype)
            logs[f"fpem/hard_count_expert_{idx}"] = hard_q[:, idx].detach().sum().to(dtype=loss_head.dtype)
            logs[f"fpem/sinkhorn_soft_col_mass_expert_{idx}"] = soft_q[:, idx].detach().sum().to(dtype=loss_head.dtype)
            logs[f"fpem/env_sinkhorn_gaussian_mass_expert_{idx}"] = (
                self.env_route_gaussian_mass[idx].detach().to(device=loss_head.device, dtype=loss_head.dtype)
            )
        return selected_prediction_loss, logs, hard_q

    def _progressive_gmm_environment_loss(self, output, target, scaler, training, epoch, sample_index=None):
        y_heads = output["y_hyper_heads"]
        loss_head = head_prediction_losses(y_heads, target, scaler, getattr(self.args, "yita", 0.5))
        q_train = output.get("env_route_q")
        if q_train is None:
            q_train = self._progressive_uniform_q(loss_head)
        active_slots = self._progressive_active_slots(device=loss_head.device)
        active_heads = y_heads.index_select(1, active_slots)
        uniform_pred = active_heads.mean(dim=1)
        uniform_loss = weighted_flow_mae(uniform_pred, target, scaler, getattr(self.args, "yita", 0.5))
        partition_ready = self._progressive_partition_ready()
        if training and partition_ready:
            selected_loss = (q_train.detach() * loss_head).sum(dim=1).mean()
            common_term = float(self.fpem_env_progressive_lambda_common) * uniform_loss
        else:
            selected_loss = uniform_loss
            common_term = loss_head.new_zeros(())

        routed_mae = weighted_flow_mae(output["prediction"], target, scaler, getattr(self.args, "yita", 0.5))
        oracle_hard_mae, _oracle_idx, _oracle_pred, per_sample_expert_mae = self._per_sample_oracle_from_heads(
            y_heads.detach(), target.detach(), scaler
        )
        hard_route_idx = q_train.detach().argmax(dim=1)
        batch_indices = torch.arange(y_heads.shape[0], device=y_heads.device)
        hard_route_pred = y_heads.detach()[batch_indices, hard_route_idx]
        routed_hard_mae = self._route_prediction_mae(hard_route_pred, target.detach(), scaler)
        self._warn_if_invalid_oracle(
            oracle_hard_mae,
            routed_hard_mae,
            "progressive_gmm_environment",
        )
        route_q = q_train.detach().float()
        usage = route_q.mean(dim=0)
        entropy = -(route_q.clamp(1e-8, 1.0) * route_q.clamp(1e-8, 1.0).log()).sum(dim=1).mean()
        eff = torch.exp(-(usage.clamp_min(1e-8) * usage.clamp_min(1e-8).log()).sum())

        compactness_loss = loss_head.new_zeros(())
        consistency_loss = loss_head.new_zeros(())
        student_embedding = output.get("progressive_student_embedding")
        teacher_embedding = output.get("progressive_teacher_embedding")
        cluster_id = output.get("progressive_cluster_id")
        if partition_ready and torch.is_tensor(student_embedding) and torch.is_tensor(cluster_id):
            valid = cluster_id >= 0
            if bool(valid.any()):
                z_student = student_embedding[valid].float()
                mean = self.progressive_feature_mean.to(device=z_student.device, dtype=z_student.dtype)
                std = self.progressive_feature_std.to(device=z_student.device, dtype=z_student.dtype).clamp_min(1e-6)
                centers = self.progressive_gmm_mu.to(device=z_student.device, dtype=z_student.dtype) * std.view(1, -1)
                centers = centers + mean.view(1, -1)
                assigned_center = centers.index_select(0, cluster_id[valid].long()).detach()
                compactness_loss = (z_student - assigned_center).pow(2).mean().to(dtype=loss_head.dtype)
        if (
            partition_ready
            and torch.is_tensor(student_embedding)
            and torch.is_tensor(teacher_embedding)
            and int(self.progressive_active_cluster_count.detach().item()) > 1
        ):
            logit_scale = float(max(int(student_embedding.shape[-1]), 1))
            teacher_logits = self._progressive_gmm_logits(teacher_embedding.detach()).clamp(-50.0, 50.0) / logit_scale
            student_logits = self._progressive_gmm_logits(student_embedding).clamp(-50.0, 50.0) / logit_scale
            teacher_prob = torch.softmax(teacher_logits, dim=-1).detach()
            log_student = torch.log_softmax(student_logits, dim=-1)
            consistency_loss = F.kl_div(log_student, teacher_prob, reduction="batchmean").to(dtype=loss_head.dtype)

        total = selected_loss + common_term
        if training:
            total = total + float(self.fpem_env_cluster_compactness_lambda) * compactness_loss
            total = total + float(self.fpem_env_cluster_consistency_lambda) * consistency_loss

        latest = getattr(self, "_latest_progressive_partition_logs", {}) or {}
        selected_k = int(self.progressive_active_cluster_count.detach().item())
        partition_epoch = int(self.progressive_last_partition_epoch.detach().item())
        logs = {
            "fpem/env_route_loss": total.detach(),
            "fpem/env_route_L_final": selected_loss.detach(),
            "fpem/env_route_L_global": uniform_loss.detach(),
            "fpem/env_route_L_route_soft": loss_head.new_zeros(()),
            "fpem/env_route_L_expert": selected_loss.detach(),
            "fpem/env_route_L_router_oracle": loss_head.new_zeros(()),
            "fpem/env_route_L_balance": loss_head.new_zeros(()),
            "fpem/env_route_L_diverse": loss_head.new_zeros(()),
            "fpem/env_route_L_proto_align": loss_head.new_zeros(()),
            "fpem/env_route_entropy": entropy.detach().to(dtype=loss_head.dtype),
            "fpem/env_route_train_mode_progressive_gmm_environment": loss_head.new_tensor(1.0),
            "fpem/progressive_assignment_uses_target_or_prediction_error": loss_head.new_tensor(0.0),
            "fpem/top_level_prediction_source_progressive_gmm_teacher_hard": loss_head.new_tensor(1.0),
            "fpem/progressive_partition_ready": loss_head.new_tensor(float(partition_ready)),
            "fpem/progressive_selected_k": loss_head.new_tensor(float(selected_k)),
            "fpem/progressive_active_expert_count": loss_head.new_tensor(float(active_slots.numel())),
            "fpem/progressive_last_partition_epoch": loss_head.new_tensor(float(partition_epoch)),
            "fpem/progressive_lambda_common": loss_head.new_tensor(float(self.fpem_env_progressive_lambda_common)),
            "fpem/progressive_common_term": common_term.detach(),
            "fpem/progressive_compactness_loss": compactness_loss.detach(),
            "fpem/progressive_consistency_loss": consistency_loss.detach(),
            "fpem/progressive_cluster_entropy": loss_head.new_tensor(float(latest.get("cluster_assignment_entropy", 0.0))),
            "fpem/progressive_assignment_change_ratio": loss_head.new_tensor(float(latest.get("assignment_change_ratio", 0.0))),
            "fpem/progressive_hungarian_overlap_score": loss_head.new_tensor(float(latest.get("hungarian_overlap_score", 0.0))),
            "fpem/progressive_partition_update_time_sec": loss_head.new_tensor(float(latest.get("partition_update_time_sec", 0.0))),
            "fpem/progressive_routed_mae": routed_mae.detach(),
            "fpem/progressive_routed_hard_eval_mae": routed_hard_mae.detach().to(dtype=loss_head.dtype),
            "fpem/uniform_expert_mae": uniform_loss.detach(),
            "fpem/oracle_hard_mae": oracle_hard_mae.detach().to(dtype=loss_head.dtype),
            "fpem/router_regret": (routed_hard_mae.detach() - oracle_hard_mae.detach()).to(dtype=loss_head.dtype),
            "fpem/effective_expert_number": eff.detach().to(dtype=loss_head.dtype),
            "fpem/max_expert_usage_ratio": usage.max().detach().to(dtype=loss_head.dtype),
            "fpem/min_expert_usage_ratio": usage.min().detach().to(dtype=loss_head.dtype),
        }
        bic_by_k = latest.get("bic_by_k", []) if isinstance(latest, dict) else []
        for item in bic_by_k:
            try:
                k = int(item.get("k", 0))
                if 1 <= k <= self.fpem_env_route_k:
                    logs[f"fpem/progressive_bic_k{k}"] = loss_head.new_tensor(float(item.get("bic", 0.0)))
                    logs[f"fpem/progressive_bic_score_k{k}"] = loss_head.new_tensor(float(item.get("bic_score", item.get("bic", 0.0))))
                    logs[f"fpem/progressive_bic_rejected_k{k}"] = loss_head.new_tensor(float(bool(item.get("rejected_min_cluster", False))))
            except Exception:
                continue
        cluster_sizes = latest.get("cluster_sizes", []) if isinstance(latest, dict) else []
        cluster_ratios = latest.get("cluster_ratios", []) if isinstance(latest, dict) else []
        mapping = self.progressive_cluster_to_expert.detach().to(device=loss_head.device)
        for idx in range(self.fpem_env_route_k):
            logs[f"fpem/route_soft_mean_expert_{idx}"] = usage[idx].detach().to(dtype=loss_head.dtype)
            logs[f"fpem/route_hard_count_expert_{idx}"] = q_train[:, idx].detach().sum().to(dtype=loss_head.dtype)
            logs[f"fpem/hard_count_expert_{idx}"] = q_train[:, idx].detach().sum().to(dtype=loss_head.dtype)
            logs[f"fpem/progressive_active_expert_{idx}"] = self.progressive_active_expert_mask[idx].detach().to(
                device=loss_head.device, dtype=loss_head.dtype
            )
            if idx < len(cluster_sizes):
                logs[f"fpem/progressive_cluster_size_{idx}"] = loss_head.new_tensor(float(cluster_sizes[idx]))
            if idx < len(cluster_ratios):
                logs[f"fpem/progressive_cluster_ratio_{idx}"] = loss_head.new_tensor(float(cluster_ratios[idx]))
            if idx < mapping.numel():
                logs[f"fpem/progressive_cluster_to_expert_{idx}"] = mapping[idx].to(dtype=loss_head.dtype)
            mask = q_train.detach().argmax(dim=1) == idx
            for expert_idx in range(self.fpem_env_route_k):
                if bool(mask.any()):
                    logs[f"fpem/expert_cross_mae_group_{idx}_expert_{expert_idx}"] = (
                        loss_head[mask, expert_idx].detach().mean().to(dtype=loss_head.dtype)
                    )
                else:
                    logs[f"fpem/expert_cross_mae_group_{idx}_expert_{expert_idx}"] = loss_head.new_zeros(())
        return total, logs, q_train.detach()

    def _warmup_risk_sinkhorn_loss(self, output, target, scaler, training, epoch):
        y_heads = output["y_hyper_heads"]
        predicted_risk = output.get("predicted_risk")
        if predicted_risk is None:
            predicted_risk = self._risk_router_scores(output["route_features"], ref_dtype=y_heads.dtype)
        loss_head = head_prediction_losses(y_heads, target, scaler, getattr(self.args, "yita", 0.5))
        target_risk = loss_head.detach()
        uniform_pred = y_heads.mean(dim=1)
        uniform_loss = weighted_flow_mae(uniform_pred, target, scaler, getattr(self.args, "yita", 0.5))
        oracle_mae, oracle_idx, _oracle_pred, _oracle_per_head = self._per_sample_oracle_from_heads(
            y_heads.detach(), target.detach(), scaler
        )

        stage = self._warmup_risk_stage(epoch)
        if stage == "warmup":
            q_train = loss_head.new_full(loss_head.shape, 1.0 / max(loss_head.shape[1], 1))
            selected_loss = uniform_loss
            sinkhorn_soft_q = q_train
            hard_q = q_train
            repaired = loss_head.new_zeros(())
            sinkhorn_temperature = loss_head.new_tensor(self.fpem_sinkhorn_temperature_start)
            sinkhorn_progress = loss_head.new_zeros(())
        else:
            sinkhorn_temperature, sinkhorn_progress = self._warmup_risk_sinkhorn_temperature(epoch, loss_head)
            cost = self._row_normalize_cost(loss_head.detach().float()).to(dtype=loss_head.dtype)
            scores = -cost / sinkhorn_temperature.clamp_min(1e-6)
            sinkhorn_soft_q = self._sinkhorn_from_scores(
                scores,
                num_iters=int(getattr(self.args, "fpem_env_route_sinkhorn_iters", 5)),
            ).to(dtype=loss_head.dtype)
            hard_idx, repaired = self._balanced_hard_idx_from_soft(sinkhorn_soft_q)
            hard_q = F.one_hot(hard_idx, num_classes=loss_head.shape[1]).to(
                device=loss_head.device, dtype=loss_head.dtype
            )
            if stage == "soft":
                q_train = sinkhorn_soft_q.detach()
            else:
                q_train = hard_q.detach()
            selected_loss = (q_train * loss_head).sum(dim=1).mean()

        common_loss = self.fpem_sinkhorn_lambda_common * uniform_loss if stage != "warmup" else loss_head.new_zeros(())
        risk_loss_raw = F.smooth_l1_loss(predicted_risk.float(), target_risk.float())
        ranking_loss_raw, ranking_acc = self._risk_ranking_loss(predicted_risk.float(), target_risk.float())
        total = selected_loss + common_loss
        if training:
            total = total + self.fpem_risk_router_lambda * risk_loss_raw.to(dtype=loss_head.dtype)
            total = total + self.fpem_risk_router_pairwise_lambda * ranking_loss_raw.to(dtype=loss_head.dtype)

        risk_q = output.get("risk_route_q")
        if risk_q is None:
            risk_q, _risk_logits, _mode = self._risk_route_q_from_scores(predicted_risk, training=False, epoch=epoch)
        risk_pred = (risk_q.view(risk_q.shape[0], risk_q.shape[1], 1, 1, 1) * y_heads).sum(dim=1)
        risk_mae = weighted_flow_mae(risk_pred, target, scaler, getattr(self.args, "yita", 0.5))
        risk_eval_mae = self._route_prediction_mae(risk_pred.detach(), target.detach(), scaler)
        usage = risk_q.detach().float().mean(dim=0)
        hard_usage_q = q_train.detach().float()
        sinkhorn_entropy = (
            -(sinkhorn_soft_q.detach().float().clamp(1e-8, 1.0)
              * sinkhorn_soft_q.detach().float().clamp(1e-8, 1.0).log()).sum(dim=1).mean()
        ).to(dtype=loss_head.dtype)
        eff = torch.exp(-(usage.clamp_min(1e-8) * usage.clamp_min(1e-8).log()).sum())
        risk_abs = (predicted_risk.detach().float() - target_risk.float()).abs().mean().to(dtype=loss_head.dtype)
        risk_argmin = predicted_risk.detach().argmin(dim=1)
        oracle_q = F.one_hot(oracle_idx, num_classes=loss_head.shape[1]).to(dtype=loss_head.dtype, device=loss_head.device)
        risk_oracle_acc = (risk_argmin == oracle_idx).to(dtype=loss_head.dtype).mean()
        logs = {
            "fpem/env_route_loss": total.detach(),
            "fpem/env_route_L_final": selected_loss.detach(),
            "fpem/env_route_L_global": uniform_loss.detach(),
            "fpem/env_route_L_route_soft": loss_head.new_zeros(()),
            "fpem/env_route_L_expert": selected_loss.detach(),
            "fpem/env_route_L_router_oracle": risk_loss_raw.detach().to(dtype=loss_head.dtype),
            "fpem/env_route_L_balance": loss_head.new_zeros(()),
            "fpem/env_route_L_diverse": loss_head.new_zeros(()),
            "fpem/env_route_L_proto_align": loss_head.new_zeros(()),
            "fpem/env_route_entropy": (-(risk_q.float().clamp(1e-8, 1.0) * risk_q.float().clamp(1e-8, 1.0).log()).sum(dim=1).mean()).detach().to(dtype=loss_head.dtype),
            "fpem/env_route_train_mode_warmup_risk_sinkhorn": loss_head.new_tensor(1.0),
            "fpem/top_level_prediction_source_risk_router_soft": loss_head.new_tensor(1.0),
            "fpem/warmup_risk_stage_warmup": loss_head.new_tensor(float(stage == "warmup")),
            "fpem/warmup_risk_stage_soft": loss_head.new_tensor(float(stage == "soft")),
            "fpem/warmup_risk_stage_hard": loss_head.new_tensor(float(stage == "hard")),
            "fpem/warmup_risk_selected_loss": selected_loss.detach(),
            "fpem/warmup_risk_uniform_common_loss": uniform_loss.detach(),
            "fpem/warmup_risk_lambda_common": loss_head.new_tensor(float(self.fpem_sinkhorn_lambda_common)),
            "fpem/warmup_risk_total_common_term": common_loss.detach(),
            "fpem/warmup_risk_temperature": sinkhorn_temperature.detach(),
            "fpem/warmup_risk_temperature_progress": sinkhorn_progress.detach(),
            "fpem/sinkhorn_soft_entropy": sinkhorn_entropy.detach(),
            "fpem/sinkhorn_selected_loss": selected_loss.detach(),
            "fpem/hard_sinkhorn_enabled": loss_head.new_tensor(float(stage == "hard" and training)),
            "fpem/hard_sinkhorn_balance_repaired": repaired.detach().to(dtype=loss_head.dtype),
            "fpem/sinkhorn_soft_row_sum_mean": sinkhorn_soft_q.sum(dim=1).mean().detach().to(dtype=loss_head.dtype),
            "fpem/oracle_hard_mae": oracle_mae.detach().to(dtype=loss_head.dtype),
            "fpem/uniform_expert_mae": uniform_loss.detach(),
            "fpem/risk_routed_mae": risk_mae.detach(),
            "fpem/risk_routed_eval_mae": risk_eval_mae.detach().to(dtype=loss_head.dtype),
            "fpem/router_regret": (risk_eval_mae.detach() - oracle_mae.detach()).to(dtype=loss_head.dtype),
            "fpem/risk_prediction_mae": risk_abs.detach(),
            "fpem/risk_loss": risk_loss_raw.detach().to(dtype=loss_head.dtype),
            "fpem/risk_ranking_loss": ranking_loss_raw.detach().to(dtype=loss_head.dtype),
            "fpem/risk_ranking_accuracy": ranking_acc.detach().to(dtype=loss_head.dtype),
            "fpem/risk_oracle_argmin_accuracy": risk_oracle_acc.detach(),
            "fpem/risk_router_temperature": loss_head.new_tensor(float(self.fpem_risk_router_temperature)),
            "fpem/risk_router_lambda": loss_head.new_tensor(float(self.fpem_risk_router_lambda)),
            "fpem/risk_router_pairwise_lambda": loss_head.new_tensor(float(self.fpem_risk_router_pairwise_lambda)),
            "fpem/effective_expert_number": eff.detach().to(dtype=loss_head.dtype),
            "fpem/max_expert_usage_ratio": usage.max().detach().to(dtype=loss_head.dtype),
            "fpem/min_expert_usage_ratio": usage.min().detach().to(dtype=loss_head.dtype),
        }
        for idx in range(self.fpem_env_route_k):
            logs[f"fpem/route_soft_mean_expert_{idx}"] = usage[idx].detach().to(dtype=loss_head.dtype)
            logs[f"fpem/route_hard_count_expert_{idx}"] = q_train[:, idx].detach().sum().to(dtype=loss_head.dtype)
            logs[f"fpem/hard_count_expert_{idx}"] = q_train[:, idx].detach().sum().to(dtype=loss_head.dtype)
            logs[f"fpem/sinkhorn_soft_col_mass_expert_{idx}"] = sinkhorn_soft_q[:, idx].detach().sum().to(dtype=loss_head.dtype)
            mask = q_train.detach().argmax(dim=1) == idx
            logs[f"fpem/env_route_count_head_{idx}"] = mask.to(dtype=loss_head.dtype).sum().detach()
            for expert_idx in range(self.fpem_env_route_k):
                if bool(mask.any()):
                    logs[f"fpem/expert_cross_mae_group_{idx}_expert_{expert_idx}"] = (
                        loss_head[mask, expert_idx].detach().mean().to(dtype=loss_head.dtype)
                    )
                else:
                    logs[f"fpem/expert_cross_mae_group_{idx}_expert_{expert_idx}"] = loss_head.new_zeros(())
        # The third return value is stored in latest_fpem_outputs for diagnostics.
        # It must reflect the responsibility used to train experts in this batch,
        # not the oracle argmin label computed from the current target.
        return total, logs, q_train.detach()

    def _apply_env_rep_ablation(self, e_useful):
        mode = self.fpem_env_rep_ablation
        logs = {
            "fpem/env_rep_ablation_normal": e_useful.new_tensor(float(mode == "normal")),
            "fpem/env_rep_ablation_zero": e_useful.new_tensor(float(mode == "zero")),
            "fpem/env_rep_ablation_shuffle_batch": e_useful.new_tensor(float(mode == "shuffle_batch")),
            "fpem/env_rep_ablation_mode_id": e_useful.new_tensor(
                {"normal": 0.0, "zero": 1.0, "shuffle_batch": 2.0}[mode]
            ),
        }
        if mode == "normal":
            out = e_useful
        elif mode == "zero":
            out = torch.zeros_like(e_useful)
        else:
            batch_size = int(e_useful.shape[0])
            if batch_size <= 1:
                out = e_useful.clone()
            else:
                perm = torch.randperm(batch_size, device=e_useful.device)
                out = e_useful.index_select(0, perm)
        if out.shape != e_useful.shape:
            raise AssertionError(
                f"env representation ablation changed shape: {tuple(e_useful.shape)} -> {tuple(out.shape)}"
            )
        return out, logs

    def forward_features(self, x, time_label=None, exog=None):
        # Keep the invariant encoder strictly x-only. Exogenous time/day/rush
        # and load-level information is fused only into the environment branch input.
        self._latest_route_aux_stats = self._route_aux_stats_from_inputs(x, time_label=time_label, exog=exog)
        if self.fpem_use_pretrained_inv_agcrn:
            with torch.no_grad():
                z_inv_raw_seq = self.encoder_inv(x)
        else:
            z_inv_raw_seq = self.encoder_inv(x)
        self._latest_z_inv_raw_seq = z_inv_raw_seq.detach()
        z_inv_seq = self.inv_projector(z_inv_raw_seq) if self.fpem_use_inv_projector else z_inv_raw_seq
        x_env, env_exog_logs = self._apply_env_exogenous(x, time_label=time_label, exog=exog)
        self._latest_env_exog_logs = env_exog_logs
        if self._progressive_gmm_environment_enabled():
            with torch.no_grad():
                teacher_seq = self.encoder_env_teacher(x_env.detach())
                teacher_nodes = self._temporal_pool_env(teacher_seq)
                self._latest_progressive_teacher_embedding = self._progressive_pool_embedding(teacher_nodes).detach()
        else:
            self._latest_progressive_teacher_embedding = None
        z_env_seq = self.encoder_env(x_env)
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
        sample_index=None,
        return_loss=False,
        return_output=False,
    ):
        z_inv_seq, z_env_seq = self.forward_features(x, time_label=time_label, exog=c)
        if return_loss:
            return self.calculate_loss(
                z_inv_seq, z_env_seq, target, c, time_label, scaler, loss_weights, p, bool(training_loss),
                sample_index=sample_index,
            )
        if return_output:
            return self.forward_output_from_features(
                z_inv_seq, z_env_seq, exog=c, training=bool(training_loss), epoch=self._epoch_index(p),
                sample_index=sample_index,
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
        return self.fpem_env_route_head_mode in {
            "hyper_inv_film",
            "hyper_inv_film_proto",
            "hyper_inv_film_proto_concat",
            "hyper_inv_film_proto_input_concat",
            "hyper_inv_film_proto_input_add",
        }

    def _use_hyper_proto_film(self):
        return self.fpem_env_route_head_mode in {
            "hyper_inv_film_proto",
            "hyper_inv_film_proto_concat",
            "hyper_inv_film_proto_input_concat",
            "hyper_inv_film_proto_input_add",
        }

    def _use_hyper_concat_fusion(self):
        return self.fpem_env_route_head_mode == "hyper_inv_film_proto_concat"

    def _use_hyper_input_concat(self):
        return self.fpem_env_route_head_mode == "hyper_inv_film_proto_input_concat"

    def _use_hyper_input_add(self):
        return self.fpem_env_route_head_mode == "hyper_inv_film_proto_input_add"

    def _hyper_router_logits(self, z_inv, e_useful):
        ctx = torch.cat([z_inv.mean(dim=1), e_useful.mean(dim=1)], dim=-1)
        return self.hyper_router(ctx)

    def _env_prediction_q_from_logits(self, logits):
        tau = max(float(getattr(self.args, "fpem_env_route_tau", 1.0)), 1e-6)
        if logits.shape[-1] == self.fpem_env_route_k + 1:
            logits_env = logits[:, 1:]
            fallback_q = torch.softmax(logits / tau, dim=-1)[:, 0]
        else:
            logits_env = logits
            fallback_q = logits.new_zeros(logits.shape[0])
        q_env = torch.softmax(logits_env / tau, dim=-1)
        return q_env, logits_env, fallback_q

    def _prediction_q_from_logits_with_optional_fallback(self, logits):
        """Return router probabilities for normal prediction-oracle inference.

        For the hyper/proto input-add family, the older helper intentionally
        drops the invariant fallback column and returns only environment-expert
        probabilities.  The explicit fallback experiment needs a real
        candidate set [invariant head, env head 0, ...], so this helper keeps
        the full K+1 distribution only when the experiment explicitly enables
        fpem_env_route_use_inv_fallback_expert.
        """
        tau = max(float(getattr(self.args, "fpem_env_route_tau", 1.0)), 1e-6)
        if self.fpem_env_route_use_inv_fallback_expert and logits.shape[-1] == self.fpem_env_route_k + 1:
            q = torch.softmax(logits / tau, dim=-1)
            return q, logits, q[:, 0]
        return self._env_prediction_q_from_logits(logits)

    def _temporal_pool_inv(self, z_inv_seq):
        # Align AGCRN FPEM with STEVE_CODE_shit: keep [B,T,N,H],
        # then collapse T with Conv2d(T_dim, 1, 1).
        return self.tcl4h(z_inv_seq).squeeze(1)

    def _temporal_pool_env(self, z_env_seq):
        return self.tcl4c(z_env_seq).squeeze(1)

    @staticmethod
    def _progressive_pool_embedding(e_nodes):
        # Generic observable environment embedding: temporal-pooled encoder
        # features collapsed over nodes.  No target, expert loss, Sinkhorn label,
        # or hand-crafted time-series statistic is involved.
        if e_nodes.dim() != 3:
            raise ValueError(f"progressive environment nodes must be [B,N,H], got {tuple(e_nodes.shape)}")
        return e_nodes.mean(dim=1)

    @staticmethod
    def _flatten_bn_for_mi(x):
        if x is None:
            return None
        if x.dim() != 3:
            raise ValueError(f"MI tensor must be [B,N,H], got {tuple(x.shape)}")
        return x.reshape(-1, x.shape[-1])

    @staticmethod
    def _set_requires_grad(module, flag):
        for param in module.parameters():
            param.requires_grad_(flag)

    def _configure_trainable_for_selected_mode(self):
        """Disable gradients for modules that are unreachable in the selected mode.

        This keeps old forward behavior unchanged while making the optimizer
        parameter set reflect the active architecture more closely.
        """
        if not self.fpem_use_env_supervision:
            self._set_requires_grad(self.env_day_head, False)
            self._set_requires_grad(self.env_hour_head, False)
            self._set_requires_grad(self.env_rush_head, False)
        if not self.fpem_use_inv_projector:
            self._set_requires_grad(self.inv_projector, False)
        if not self.fpem_use_inv_env_adversarial:
            self._set_requires_grad(self.inv_day_head, False)
            self._set_requires_grad(self.inv_hour_head, False)
            self._set_requires_grad(self.inv_rush_head, False)
        if not self.fpem_use_club_mi:
            self._set_requires_grad(self.mi_net, False)
        if not self.fpem_env_use_exogenous:
            self._set_requires_grad(self.env_hour_embedding, False)
            self._set_requires_grad(self.env_day_embedding, False)
            self._set_requires_grad(self.env_rush_embedding, False)
            self._set_requires_grad(self.env_load_embedding, False)
            self._set_requires_grad(self.env_exog_proj, False)
        if self.fpem_force_uniform_route:
            self.env_prototypes.requires_grad_(False)
            self._set_requires_grad(self.hyper_router, False)
            if hasattr(self.route_heads, "router"):
                self._set_requires_grad(self.route_heads.router, False)
        if self.fpem_env_route_train_mode in {
            "hard_prediction_sinkhorn",
            "hard_prediction_environment_sinkhorn",
            "warmup_risk_sinkhorn",
            "progressive_gmm_environment",
        }:
            self.env_prototypes.requires_grad_(False)
        if self.fpem_env_route_train_mode != "warmup_risk_sinkhorn":
            self._set_requires_grad(self.risk_router, False)
        else:
            self._set_requires_grad(self.hyper_router, False)
            if hasattr(self.route_heads, "router"):
                self._set_requires_grad(self.route_heads.router, False)
        if self.fpem_env_route_train_mode == "progressive_gmm_environment":
            self._set_requires_grad(self.hyper_router, False)
            if hasattr(self.route_heads, "router"):
                self._set_requires_grad(self.route_heads.router, False)
            self._set_requires_grad(self.encoder_env_teacher, False)
        if not self.fpem_use_env_fusion or (
            self._use_hyper_inv_film() and not self._use_hyper_concat_fusion()
        ):
            self._set_requires_grad(self.fusion, False)
        if not self._use_hyper_input_concat():
            self._set_requires_grad(self.hyper_concat_input_heads, False)
            self._set_requires_grad(self.hyper_concat_predict_conv_2, False)
        if self.fpem_env_route_head_mode == "concat_input":
            self._set_requires_grad(self.hyper_inv_heads, False)
            self._set_requires_grad(self.hyper_concat_input_heads, False)
            self._set_requires_grad(self.hyper_concat_predict_conv_2, False)
            self._set_requires_grad(self.hyper_router, False)
        elif self.fpem_env_route_head_mode == "hyper_inv_film":
            self._set_requires_grad(self.route_heads, False)
            self.env_prototypes.requires_grad_(False)
        elif self.fpem_env_route_head_mode in {
            "hyper_inv_film_proto",
            "hyper_inv_film_proto_concat",
            "hyper_inv_film_proto_input_add",
        }:
            self._set_requires_grad(self.route_heads, False)
            if (
                self.fpem_env_route_target_mode == "env_prototype"
                and self.fpem_env_route_train_mode not in {
                    "hard_prediction_sinkhorn",
                    "hard_prediction_environment_sinkhorn",
                    "warmup_risk_sinkhorn",
                    "progressive_gmm_environment",
                }
            ):
                self._set_requires_grad(self.hyper_router, False)
        elif self._use_hyper_input_concat():
            self._set_requires_grad(self.route_heads, False)
            self._set_requires_grad(self.hyper_inv_heads, False)
            if (
                self.fpem_env_route_target_mode == "env_prototype"
                and self.fpem_env_route_train_mode not in {
                    "hard_prediction_sinkhorn",
                    "hard_prediction_environment_sinkhorn",
                    "warmup_risk_sinkhorn",
                    "progressive_gmm_environment",
                }
            ):
                self._set_requires_grad(self.hyper_router, False)

    def _fit_club_estimator(self, z_inv, c_cur, training):
        logs = {}
        if (not training) or (not self.fpem_use_club_mi):
            return logs

        z_flat = self._flatten_bn_for_mi(z_inv).detach()
        c_flat = self._flatten_bn_for_mi(c_cur).detach()
        total = z_flat.shape[0]
        if total <= 1:
            return logs

        ratio = max(0.0, min(1.0, self.fpem_club_sample_ratio))
        sample_size = min(max(2, int(total * ratio)), total)
        indices = torch.randperm(total, device=z_flat.device)[:sample_size]
        z_sub = z_flat[indices]
        c_sub = c_flat[indices]

        self._set_requires_grad(self.mi_net, True)
        last_loss = None
        for _ in range(max(1, self.fpem_club_steps)):
            self.optimizer_mi_net.zero_grad(set_to_none=True)
            fit_loss = self.mi_net.learning_loss(z_sub, c_sub)
            fit_loss.backward()
            self.optimizer_mi_net.step()
            last_loss = fit_loss.detach()
        self.optimizer_mi_net.zero_grad(set_to_none=True)
        if last_loss is not None:
            logs["fpem/club_fit_loss"] = last_loss
        return logs

    def _club_upper_bound_loss(self, z_inv, c_cur):
        if not self.fpem_use_club_mi:
            return z_inv.new_zeros(()), {}

        z_flat = self._flatten_bn_for_mi(z_inv)
        c_flat = self._flatten_bn_for_mi(c_cur)
        if z_flat.shape[0] <= 1:
            return z_inv.new_zeros(()), {}

        self._set_requires_grad(self.mi_net, False)
        try:
            club_upper = self.mi_net(z_flat, c_flat)
        finally:
            self._set_requires_grad(self.mi_net, True)
        return club_upper, {"fpem/club_upper": club_upper.detach()}

    def _predict_invariant_from_nodes(self, h_inv):
        h = h_inv.unsqueeze(1).permute(0, 3, 2, 1)
        return self.invariant_predict_conv_2(h).permute(0, 3, 2, 1)

    def _predict_variant_from_nodes(self, env_feat):
        h = env_feat.unsqueeze(1).permute(0, 3, 2, 1)
        return self.variant_predict_conv_2(h).permute(0, 3, 2, 1)

    def _predict_hyper_concat_from_nodes(self, feat):
        h = feat.unsqueeze(1).permute(0, 3, 2, 1)
        return self.hyper_concat_predict_conv_2(h).permute(0, 3, 2, 1)

    def _predict_from_nodes(self, z_inv, e_useful, training=False, epoch=None, sample_index=None, teacher_embedding=None):
        h_inv = z_inv
        y_inv = self._predict_invariant_from_nodes(h_inv)
        y_env = self._predict_variant_from_nodes(h_inv + e_useful)
        output = {
            "Z_inv": z_inv,
            "E_useful": e_useful,
            "h_inv": h_inv,
            "y_inv": y_inv,
            "y_env": y_env,
            "y_global": y_inv,
        }

        route_active = self._route_enabled_for_epoch(training, epoch)
        hyper_mode = route_active and self._use_hyper_inv_film()
        fusion_active = route_active and self.fpem_use_env_fusion and (
            (not hyper_mode) or self._use_hyper_concat_fusion()
        )
        fusion_logs = self.fusion.zero_logs(y_inv)
        if hyper_mode:
            hyper_heads = self.hyper_inv_heads
            if self._use_hyper_input_concat():
                # Requested variant: concatenate invariant and environment
                # representations before the FiLM-modulated prediction head.
                hyper_h = torch.cat([z_inv, e_useful], dim=-1)
                hyper_pred = self._predict_hyper_concat_from_nodes
                hyper_heads = self.hyper_concat_input_heads
            elif self._use_hyper_input_add():
                # Requested variant: use the same additive representation as
                # the environment-enhanced branch before FiLM modulation.
                hyper_h = z_inv + e_useful
                hyper_pred = self._predict_variant_from_nodes
            elif self._use_hyper_proto_film():
                # FPEM-B default: FiLM modulates the invariant forecasting
                # state itself.  With identity FiLM initialization, every
                # hyper head reconstructs y_inv exactly before learning
                # non-zero gamma/beta.
                hyper_h = z_inv
                hyper_pred = self._predict_invariant_from_nodes
            else:
                # Backward-compatible old hyper path.
                hyper_h = self.inv_trunk(z_inv)
                hyper_pred = self.inv_pred
            hyper_out = hyper_heads(z_inv, e_useful, hyper_h, hyper_pred)
            y_hyper_heads = hyper_out["y_hyper_heads"]
            route_features = self._observable_route_features(z_inv, e_useful, y_hyper_heads)
            output["route_features"] = route_features
            hard_prediction_only_mode = self._hard_prediction_sinkhorn_enabled()
            hard_prediction_environment_mode = self._hard_prediction_environment_sinkhorn_enabled()
            warmup_risk_mode = self._warmup_risk_sinkhorn_enabled()
            progressive_gmm_mode = self._progressive_gmm_environment_enabled()
            hard_prediction_mode = hard_prediction_only_mode or hard_prediction_environment_mode
            if warmup_risk_mode or progressive_gmm_mode:
                logits = z_inv.new_zeros(z_inv.shape[0], self.fpem_env_route_k)
            elif hard_prediction_mode:
                logits = self._hyper_router_logits(z_inv.detach(), e_useful.detach())
            else:
                logits = self._hyper_router_logits(z_inv, e_useful)
            if self._use_hyper_proto_film():
                if self.fpem_force_uniform_route:
                    q, final_logits = self._uniform_route(z_inv)
                    q_prediction = q
                    logits_prediction = final_logits
                    fallback_q_prediction = q.new_zeros(q.shape[0])
                    proto_q = q
                    proto_logits = final_logits
                    proto_mode = "uniform_fixed"
                    target_mode = "uniform_fixed"
                elif warmup_risk_mode:
                    predicted_risk = self._risk_router_scores(route_features, ref_dtype=z_inv.dtype)
                    q, final_logits, proto_mode = self._risk_route_q_from_scores(
                        predicted_risk,
                        training=training,
                        epoch=epoch,
                    )
                    if training:
                        q = q.detach()
                    q_prediction = q
                    logits_prediction = final_logits
                    fallback_q_prediction = q.new_zeros(q.shape[0])
                    proto_q = q.detach()
                    proto_logits = final_logits.detach()
                    target_mode = "warmup_risk_sinkhorn"
                    output["predicted_risk"] = predicted_risk
                    output["risk_route_q"] = q
                    output["top_level_prediction_source"] = "risk_router_soft"
                elif progressive_gmm_mode:
                    q, final_logits, cluster_id, proto_mode = self._progressive_route_q(
                        e_useful,
                        training=training,
                        epoch=epoch,
                        sample_index=sample_index,
                        teacher_embedding=teacher_embedding,
                    )
                    q_prediction = q
                    logits_prediction = final_logits
                    fallback_q_prediction = q.new_zeros(q.shape[0])
                    proto_q = q.detach()
                    proto_logits = final_logits.detach()
                    target_mode = "progressive_gmm_environment"
                    output["progressive_cluster_id"] = cluster_id
                    output["top_level_prediction_source"] = "progressive_gmm_teacher_hard"
                elif hard_prediction_only_mode:
                    q_prediction, logits_prediction, fallback_q_prediction = self._env_prediction_q_from_logits(logits)
                    hard_idx = q_prediction.detach().argmax(dim=-1)
                    q = F.one_hot(hard_idx, num_classes=q_prediction.shape[1]).to(
                        device=q_prediction.device, dtype=q_prediction.dtype
                    )
                    final_logits = logits_prediction
                    proto_q = q_prediction.detach()
                    proto_logits = logits_prediction.detach()
                    proto_mode = "router_hard"
                    target_mode = "router_hard"
                elif hard_prediction_environment_mode:
                    q_prediction, logits_prediction, fallback_q_prediction = self._env_prediction_q_from_logits(logits)
                    q, final_logits, proto_q, proto_mode = self._env_route_inference_q(
                        q_prediction,
                        logits_prediction,
                        route_features,
                        e_useful,
                        training=training,
                        epoch=epoch,
                    )
                    proto_logits = final_logits.detach()
                    target_mode = "hard_prediction_environment_sinkhorn"
                else:
                    q_prediction, logits_prediction, fallback_q_prediction = self._prediction_q_from_logits_with_optional_fallback(logits)
                    proto_q, proto_logits, proto_mode = self._prototype_route(e_useful, training=training, epoch=epoch)
                    target_mode = self.fpem_env_route_target_mode
                    if target_mode == "prediction_oracle":
                        q = q_prediction
                        final_logits = logits_prediction
                    elif target_mode == "hybrid":
                        alpha_env = max(0.0, min(1.0, self._hybrid_alpha(epoch)))
                        if q_prediction.shape[-1] == self.fpem_env_route_k + 1 and proto_q.shape[-1] == self.fpem_env_route_k:
                            proto_q = torch.cat([proto_q.new_zeros(proto_q.shape[0], 1), proto_q], dim=1)
                        q = alpha_env * proto_q + (1.0 - alpha_env) * q_prediction
                        q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                        final_logits = torch.log(q.float().clamp_min(1e-8)).to(dtype=q.dtype)
                    else:
                        q = proto_q
                        final_logits = proto_logits
                        target_mode = "env_prototype"
                q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                if self.fpem_env_route_use_inv_fallback_expert and q.shape[-1] == self.fpem_env_route_k + 1:
                    y_candidates = torch.cat([y_inv.unsqueeze(1), y_hyper_heads], dim=1)
                    q_env = q[:, 1:]
                    q_env_sum = q_env.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    y_route = (
                        (q_env / q_env_sum).view(q.shape[0], self.fpem_env_route_k, 1, 1, 1)
                        * y_hyper_heads
                    ).sum(dim=1)
                    y_final = (
                        q.view(q.shape[0], q.shape[1], 1, 1, 1)
                        * y_candidates
                    ).sum(dim=1)
                    fallback_q = q[:, 0]
                    alpha = fallback_q.view(-1, 1, 1, 1)
                else:
                    y_candidates = y_hyper_heads
                    y_route = (
                        q.view(q.shape[0], q.shape[1], 1, 1, 1)
                        * y_hyper_heads
                    ).sum(dim=1)
                    fallback_q = q.new_zeros(q.shape[0])
                    if self._use_hyper_concat_fusion() and self.fpem_use_env_fusion:
                        y_final, alpha, fusion_logs = self.fusion(y_inv, y_route, z_inv, e_useful, q)
                    else:
                        y_final = y_route
                        alpha = q.new_zeros(q.shape[0], 1, 1, 1)
                q_prob = q.float().clamp(1e-8, 1.0)
                output.update({
                    "y_route_heads": y_hyper_heads,
                    "y_hyper_heads": y_hyper_heads,
                    "y_candidates": y_candidates,
                    "route_q": q,
                    "env_route_q": q,
                    "env_route_logits": final_logits,
                    "env_route_q_router_soft": q_prediction,
                    "env_route_q_prediction": q_prediction,
                    "env_route_logits_prediction": logits_prediction,
                    "env_route_q_prototype": proto_q,
                    "env_route_logits_prototype": proto_logits,
                    "env_route_target_mode": target_mode,
                    "env_route_hybrid_alpha": q.new_tensor(self._hybrid_alpha(epoch)),
                    "env_route_proto_mode": proto_mode,
                    "env_prototypes": self.env_prototypes,
                    "env_route_entropy_per_sample": (-(q_prob * q_prob.log()).sum(dim=-1)).to(dtype=q.dtype),
                    "env_route_q_max": q.max(dim=-1).values,
                    "env_route_selected_head": q.argmax(dim=-1),
                    "fallback_q": fallback_q,
                    "fallback_q_prediction": fallback_q_prediction,
                    "y_route": y_route,
                    "hyper_alpha": hyper_out["hyper_alpha"],
                    "hyper_delta_norm": hyper_out["hyper_delta_norm"],
                    "hyper_gamma_norm_per_head": hyper_out["hyper_gamma_norm_per_head"],
                    "hyper_beta_norm_per_head": hyper_out["hyper_beta_norm_per_head"],
                    "route_head_mode": self.fpem_env_route_head_mode,
                })
            else:
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
                    "hyper_gamma_norm_per_head": hyper_out["hyper_gamma_norm_per_head"],
                    "hyper_beta_norm_per_head": hyper_out["hyper_beta_norm_per_head"],
                    "route_head_mode": "hyper_inv_film",
                })
        elif route_active:
            route_out = self.route_heads(
                z_inv,
                e_useful,
                tau=float(getattr(self.args, "fpem_env_route_tau", 1.0)),
            )
            if self.fpem_force_uniform_route:
                uniform_q, uniform_logits = self._uniform_route(z_inv)
                route_out["env_route_q_prediction"] = route_out["env_route_q"]
                route_out["env_route_logits_prediction"] = route_out["env_route_logits"]
                route_out["env_route_q_prototype"] = uniform_q
                route_out["env_route_logits_prototype"] = uniform_logits
                route_out["env_route_target_mode"] = "uniform_fixed"
                route_out["env_route_hybrid_alpha"] = z_inv.new_zeros(())
                route_out["env_route_proto_mode"] = "uniform_fixed"
                route_out["env_prototypes"] = self.env_prototypes
                route_out["env_route_q"] = uniform_q
                route_out["route_q"] = uniform_q
                route_out["env_route_logits"] = uniform_logits
                route_out["env_route_entropy_per_sample"] = (
                    -(uniform_q.float().clamp(1e-8, 1.0) * uniform_q.float().clamp(1e-8, 1.0).log()).sum(dim=-1)
                ).to(dtype=uniform_q.dtype)
                route_out["env_route_q_max"] = uniform_q.max(dim=-1).values
                route_out["env_route_selected_head"] = uniform_q.argmax(dim=-1)
                route_out["y_route"] = (
                    uniform_q.view(uniform_q.shape[0], uniform_q.shape[1], 1, 1, 1)
                    * route_out["y_route_heads"]
                ).sum(dim=1)
            elif self.fpem_use_env_prototype_router:
                proto_q, proto_logits, proto_mode = self._prototype_route(e_useful, training=training, epoch=epoch)
                route_out["env_route_q_prediction"] = route_out["env_route_q"]
                route_out["env_route_logits_prediction"] = route_out["env_route_logits"]
                route_out["env_route_q_prototype"] = proto_q
                route_out["env_route_logits_prototype"] = proto_logits
                route_out["env_route_target_mode"] = self.fpem_env_route_target_mode
                route_out["env_route_hybrid_alpha"] = z_inv.new_tensor(self._hybrid_alpha(epoch))
                route_out["env_route_proto_mode"] = proto_mode
                route_out["env_prototypes"] = self.env_prototypes
                route_out["env_route_q"] = proto_q
                route_out["route_q"] = proto_q
                route_out["env_route_logits"] = proto_logits
                route_out["env_route_entropy_per_sample"] = (
                    -(proto_q.float().clamp(1e-8, 1.0) * proto_q.float().clamp(1e-8, 1.0).log()).sum(dim=-1)
                ).to(dtype=proto_q.dtype)
                route_out["env_route_q_max"] = proto_q.max(dim=-1).values
                route_out["env_route_selected_head"] = proto_q.argmax(dim=-1)
                route_out["y_route"] = (
                    proto_q.view(proto_q.shape[0], proto_q.shape[1], 1, 1, 1)
                    * route_out["y_route_heads"]
                ).sum(dim=1)
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
            y_route = y_env
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
            "training": bool(training),
            "fusion_logs": fusion_logs,
        })
        return output

    def forward_output(self, x, exog=None, time_label=None, training=False, epoch=None, sample_index=None):
        z_inv_seq, z_env_seq = self.forward_features(x, time_label=time_label, exog=exog)
        return self.forward_output_from_features(
            z_inv_seq, z_env_seq, exog=exog, training=training, epoch=epoch, sample_index=sample_index
        )

    def forward_output_from_features(self, z_inv_seq, z_env_seq, exog=None, training=False, epoch=None, sample_index=None):
        z_inv_raw_seq = getattr(self, "_latest_z_inv_raw_seq", None)
        if not torch.is_tensor(z_inv_raw_seq) or z_inv_raw_seq.shape[:3] != z_inv_seq.shape[:3]:
            z_inv_raw_seq = z_inv_seq.detach()
        z_inv_raw = self._temporal_pool_inv(z_inv_raw_seq)
        z_inv = self._temporal_pool_inv(z_inv_seq)
        if self.fpem_use_confounder_extractor:
            e_raw, conf_logs = self.confounder_extractor(z_env_seq)
            if self.fpem_confounder_use_mask:
                use_mask = self._mask_enabled_for_epoch(training, epoch)
                e_useful, e_discard, env_mask, mask_loss, mask_logs = self.env_mask(
                    e_raw,
                    use_mask=use_mask,
                    lambda_sparse=float(getattr(self.args, "fpem_lambda_mask_sparse", 0.0)),
                    lambda_entropy=float(getattr(self.args, "fpem_lambda_mask_entropy", 0.0)),
                    temperature=float(getattr(self.args, "fpem_env_mask_temperature", 1.0)),
                )
            else:
                e_useful = e_raw
                e_discard = torch.zeros_like(e_raw)
                env_mask = torch.ones_like(e_raw)
                mask_loss = e_raw.new_zeros(())
                mask_logs = {
                    "fpem/mask_loss": e_raw.new_zeros(()),
                    "fpem/mask_sparse_loss": e_raw.new_zeros(()),
                    "fpem/mask_entropy_loss": e_raw.new_zeros(()),
                    "fpem/mask_entropy": e_raw.new_zeros(()),
                    "fpem/mask_mean": e_raw.new_tensor(1.0),
                    "fpem/mask_active_ratio": e_raw.new_tensor(1.0),
                }
        else:
            conf_logs = {}
            e_raw = self._temporal_pool_env(z_env_seq)
            use_mask = self._mask_enabled_for_epoch(training, epoch)
            e_useful, e_discard, env_mask, mask_loss, mask_logs = self.env_mask(
                e_raw,
                use_mask=use_mask,
                lambda_sparse=float(getattr(self.args, "fpem_lambda_mask_sparse", 0.0)),
                lambda_entropy=float(getattr(self.args, "fpem_lambda_mask_entropy", 0.0)),
                temperature=float(getattr(self.args, "fpem_env_mask_temperature", 1.0)),
            )
        e_useful, env_rep_ablation_logs = self._apply_env_rep_ablation(e_useful)
        c_cur = e_useful
        progressive_student_embedding = (
            self._progressive_pool_embedding(e_useful) if self._progressive_gmm_environment_enabled() else None
        )
        progressive_teacher_embedding = getattr(self, "_latest_progressive_teacher_embedding", None)
        output = self._predict_from_nodes(
            z_inv,
            e_useful,
            training=training,
            epoch=epoch,
            sample_index=sample_index,
            teacher_embedding=progressive_teacher_embedding,
        )
        output.update({
            "Z_inv_seq": z_inv_seq,
            "Z_inv_raw_seq": z_inv_raw_seq,
            "Z_inv_raw": z_inv_raw,
            "Z_env_seq": z_env_seq,
            "Z_inv_temporal": z_inv,
            "Z_env_temporal": e_raw,
            "C_raw": e_raw,
            "C_cur": c_cur,
            "E_raw": e_raw,
            "E_useful": e_useful,
            "E_discard": e_discard,
            "progressive_student_embedding": progressive_student_embedding,
            "progressive_teacher_embedding": progressive_teacher_embedding,
            "env_mask": env_mask,
            "mask_loss": mask_loss,
            "mask_logs": mask_logs,
            "env_exog_logs": getattr(self, "_latest_env_exog_logs", self._zero_env_exog_logs(z_env_seq)),
            "env_rep_ablation_logs": env_rep_ablation_logs,
        })
        output.update(conf_logs)
        return output

    def _future_env_from_target(self, target, y_inv):
        if not self.fpem_use_future_mi:
            return None
        mode = str(getattr(self.args, "fpem_future_mi_target_mode", self.fpem_future_mi_target_mode)).lower()
        if mode == "residual_mlp":
            if self.future_env_encoder is None:
                return None
            residual = (target[:, : y_inv.shape[1]] - y_inv.detach()).squeeze(1)
            return self.future_env_encoder(residual)
        if mode != "env_encoder":
            raise ValueError("fpem_future_mi_target_mode must be env_encoder or residual_mlp")

        target_future = target
        expected_in = int(getattr(self.args, "d_input", target_future.shape[-1]))
        if target_future.shape[-1] != expected_in:
            if self.future_target_proj is None:
                raise ValueError(
                    "target feature dim does not match d_input and future_target_proj is unavailable: "
                    f"{target_future.shape[-1]} vs {expected_in}"
                )
            target_future = self.future_target_proj(target_future)
        z_future_seq = self.encoder_env(target_future)
        pool = str(getattr(self.args, "fpem_future_mi_future_pool", "last")).lower()
        if pool == "mean":
            return z_future_seq.mean(dim=1)
        if pool != "last":
            raise ValueError("fpem_future_mi_future_pool must be last or mean")
        return z_future_seq[:, -1, :, :]

    def _future_confounder_from_target(self, target, y_inv=None):
        """Build a node-wise future confounder target ``[B,N,H]``."""
        if not self.fpem_use_future_mi:
            return None
        mode = str(
            getattr(self.args, "fpem_future_mi_target_mode", self.fpem_future_mi_target_mode)
        ).lower()
        if mode == "residual_mlp":
            if self.future_env_encoder is None or y_inv is None:
                return None
            residual = (target[:, : y_inv.shape[1]] - y_inv.detach()).squeeze(1)
            return self.future_env_encoder(residual)
        if mode != "env_encoder":
            raise ValueError("fpem_future_mi_target_mode must be env_encoder or residual_mlp")

        target_future = target
        expected_in = int(getattr(self.args, "d_input", target_future.shape[-1]))
        if target_future.shape[-1] != expected_in:
            if self.future_target_proj is None:
                raise ValueError(
                    "target feature dim does not match d_input and future_target_proj is unavailable: "
                    f"{target_future.shape[-1]} vs {expected_in}"
                )
            target_future = self.future_target_proj(target_future)
        z_future_seq = self.encoder_env(target_future)
        c_future, _future_conf_logs = self.confounder_extractor(z_future_seq)
        return c_future

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
            "fpem/swap_prediction_delta": zero,
            "fpem/swap_route_delta": zero,
            "fpem/swap_hyper_alpha_delta": zero,
            "fpem/swap_hyper_delta_norm": zero,
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
        logs["fpem/swap_prediction_delta"] = (
            swap_output["prediction"].detach() - output["prediction"].detach()
        ).abs().mean()
        if torch.is_tensor(output.get("env_route_q")) and torch.is_tensor(swap_output.get("env_route_q")):
            logs["fpem/swap_route_delta"] = (
                swap_output["env_route_q"].detach() - output["env_route_q"].detach()
            ).abs().mean()
        if torch.is_tensor(output.get("hyper_alpha")) and torch.is_tensor(swap_output.get("hyper_alpha")):
            logs["fpem/swap_hyper_alpha_delta"] = (
                swap_output["hyper_alpha"].detach() - output["hyper_alpha"].detach()
            ).abs().mean()
        if torch.is_tensor(swap_output.get("hyper_delta_norm")):
            logs["fpem/swap_hyper_delta_norm"] = swap_output["hyper_delta_norm"].detach()
        use_fallback_router = (
            self._use_hyper_inv_film()
            and not self._use_hyper_proto_film()
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

    def calculate_loss(
        self,
        Z_tensor,
        H_tensor,
        target,
        c,
        time_label,
        scaler,
        loss_weights,
        p=None,
        training=False,
        sample_index=None,
    ):
        epoch = self._epoch_index(p)
        output = self.forward_output_from_features(
            Z_tensor, H_tensor, exog=c, training=training, epoch=epoch, sample_index=sample_index
        )
        env_labels = self._env_labels_from_time(time_label, output["Z_inv"])
        club_fit_logs = self._fit_club_estimator(
            output["Z_inv"],
            output["C_cur"],
            training=training,
        )
        club_upper, club_upper_logs = self._club_upper_bound_loss(
            output["Z_inv"],
            output["C_cur"],
        )
        primary_loss = weighted_flow_mae(output["prediction"], target, scaler, getattr(self.args, "yita", 0.5))
        inv_pred_loss = weighted_flow_mae(output["y_inv"], target, scaler, getattr(self.args, "yita", 0.5))
        primary_weight = float(loss_weights[0]) if loss_weights is not None else 1.0
        hard_prediction_sinkhorn_mode = (
            self._hard_prediction_sinkhorn_enabled()
            and self.fpem_use_env_route
            and self._route_enabled_for_epoch(training, epoch)
        )
        hard_prediction_environment_sinkhorn_mode = (
            self._hard_prediction_environment_sinkhorn_enabled()
            and self.fpem_use_env_route
            and self._route_enabled_for_epoch(training, epoch)
        )
        warmup_risk_sinkhorn_mode = (
            self._warmup_risk_sinkhorn_enabled()
            and self.fpem_use_env_route
            and self._route_enabled_for_epoch(training, epoch)
        )
        progressive_gmm_environment_mode = (
            self._progressive_gmm_environment_enabled()
            and self.fpem_use_env_route
            and self._route_enabled_for_epoch(training, epoch)
        )
        hard_sinkhorn_family_mode = (
            hard_prediction_sinkhorn_mode
            or hard_prediction_environment_sinkhorn_mode
            or warmup_risk_sinkhorn_mode
            or progressive_gmm_environment_mode
        )
        if training and hard_sinkhorn_family_mode:
            loss = output["mask_loss"]
        else:
            loss = primary_weight * primary_loss + output["mask_loss"]
        if training:
            inv_lambda = float(getattr(
                self.args,
                "fpem_lambda_inv_pred",
                getattr(self.args, "fpem_env_route_lambda_global", 0.2),
            ))
            if (self._use_hyper_proto_film() and not self.fpem_env_route_use_inv_fallback_expert) or hard_sinkhorn_family_mode:
                inv_lambda = 0.0
            loss = loss + inv_lambda * inv_pred_loss
            if self._use_hyper_inv_film() and not hard_sinkhorn_family_mode:
                loss = loss + float(getattr(self.args, "fpem_lambda_hyper_delta_norm", 0.0)) * output["hyper_delta_norm"]
            loss = loss + self.fpem_lambda_club_mi * club_upper

        route_loss = primary_loss.new_zeros(())
        route_logs = self._zero_route_logs(primary_loss)
        q_route_diag = None
        if self.fpem_use_env_route and self._route_enabled_for_epoch(training, epoch):
            if warmup_risk_sinkhorn_mode:
                route_loss, route_logs, q_route_diag = self._warmup_risk_sinkhorn_loss(
                    output,
                    target,
                    scaler,
                    training=training,
                    epoch=epoch,
                )
            elif progressive_gmm_environment_mode:
                route_loss, route_logs, q_route_diag = self._progressive_gmm_environment_loss(
                    output,
                    target,
                    scaler,
                    training=training,
                    epoch=epoch,
                    sample_index=sample_index,
                )
            elif hard_prediction_environment_sinkhorn_mode:
                route_loss, route_logs, q_route_diag = self._hard_prediction_environment_sinkhorn_loss(
                    output,
                    target,
                    scaler,
                    training=training,
                    epoch=epoch,
                )
            else:
                route_loss, route_logs, q_route_diag = route_losses(output, target, scaler, self.args)
            if training:
                loss = loss + route_loss
                hard_counts = [
                    tensor_float(route_logs.get(f"fpem/route_hard_count_expert_{idx}", primary_loss.new_zeros(())))
                    for idx in range(self.fpem_env_route_k)
                ]
                if hard_counts and min(hard_counts) <= 0.0:
                    self._expert_zero_streak += 1
                else:
                    self._expert_zero_streak = 0
                if self._expert_zero_streak >= 3:
                    print(
                        f"EXPERT_COLLAPSE_WARNING epoch={epoch} zero_assignment_streak={self._expert_zero_streak}",
                        flush=True,
                    )
                    route_logs["fpem/expert_collapse_warning"] = primary_loss.new_tensor(1.0)

        if training and self.fpem_use_future_mi:
            if self.fpem_use_confounder_extractor:
                future_target = self._future_confounder_from_target(target, output.get("y_inv"))
            else:
                future_target = self._future_env_from_target(target, output.get("y_inv"))
        else:
            future_target = None
        future_loss, future_logs = future_mi_loss(
            output.get("C_cur", output["E_useful"]),
            future_target,
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

        env_cls_loss, env_cls_logs = self._env_supervision_loss(output["C_cur"], env_labels, training)
        env_supcon_loss, env_supcon_logs = self._env_supcon_loss(output["C_cur"], env_labels, training)
        inv_adv_loss, inv_adv_logs = self._inv_env_adversarial_loss(output["Z_inv"], env_labels, training)
        xcov_loss, xcov_logs = self._cross_cov_sep_loss(output["Z_inv"], output["C_cur"], training)
        if training:
            loss = loss + env_cls_loss + env_supcon_loss + inv_adv_loss + xcov_loss

        logs = {}
        logs.update(output["env_exog_logs"])
        logs.update(output["env_rep_ablation_logs"])
        logs.update(output["mask_logs"])
        logs.update(output["fusion_logs"])
        logs.update(route_logs)
        logs.update(future_logs)
        logs.update(swap_logs)
        logs.update(env_cls_logs)
        logs.update(env_supcon_logs)
        logs.update(inv_adv_logs)
        logs.update(xcov_logs)
        logs.update(club_fit_logs)
        logs.update(club_upper_logs)
        for key, value in output.items():
            if key.startswith("conf_") and torch.is_tensor(value):
                logs[f"fpem/{key}"] = value
        logs.update({
            "fpem/primary_loss": primary_loss.detach(),
            "fpem/inv_pred_loss": inv_pred_loss.detach(),
            "fpem/primary_uses_route": output["primary_uses_route"].detach(),
            "fpem/primary_uses_env_fusion": output["primary_uses_env_fusion"].detach(),
            "fpem/backbone_agcrn": primary_loss.new_tensor(float(self.fpem_backbone == "agcrn")),
            "fpem/backbone_graphwavenet": primary_loss.new_tensor(float(self.fpem_backbone == "graphwavenet")),
            "fpem/backbone_staeformer": primary_loss.new_tensor(float(self.fpem_backbone == "staeformer")),
            "fpem/gc_pred_loss_only": primary_loss.new_tensor(float(self.fpem_gc_pred_loss_only)),
            "fpem/env_route_head_mode": primary_loss.new_tensor(1.0 if self._use_hyper_inv_film() else 0.0),
            "fpem/env_route_head_mode_hyper_proto": primary_loss.new_tensor(float(self._use_hyper_proto_film())),
            "fpem/fpem_force_uniform_route": primary_loss.new_tensor(float(self.fpem_force_uniform_route)),
            "fpem/hard_prediction_sinkhorn_total_replaces_primary": primary_loss.new_tensor(
                float(training and hard_prediction_sinkhorn_mode)
            ),
            "fpem/hard_prediction_environment_sinkhorn_total_replaces_primary": primary_loss.new_tensor(
                float(training and hard_prediction_environment_sinkhorn_mode)
            ),
            "fpem/warmup_risk_sinkhorn_total_replaces_primary": primary_loss.new_tensor(
                float(training and warmup_risk_sinkhorn_mode)
            ),
            "fpem/progressive_gmm_environment_total_replaces_primary": primary_loss.new_tensor(
                float(training and progressive_gmm_environment_mode)
            ),
            "fpem/lambda_club_mi": primary_loss.new_tensor(self.fpem_lambda_club_mi),
        })
        self.latest_fpem_logs = {key: tensor_float(value) for key, value in logs.items()}

        self.latest_fpem_outputs = {
            "primary_loss": primary_loss,
            "prediction": output["prediction"],
            "y_inv": output["y_inv"],
            "y_route": output["y_route"],
            "y_route_heads": output.get("y_route_heads"),
            "y_hyper_heads": output.get("y_hyper_heads"),
            "Z_inv_raw": output.get("Z_inv_raw"),
            "Z_inv": output["Z_inv"],
            "C_cur": output["C_cur"],
            "env_route_q": output["env_route_q"],
            "env_route_q_prototype": output.get("env_route_q_prototype"),
            "env_route_q_prediction": output.get("env_route_q_prediction"),
            "env_route_q_oracle": None if (warmup_risk_sinkhorn_mode or progressive_gmm_environment_mode) else q_route_diag,
            "env_route_q_train": q_route_diag if (warmup_risk_sinkhorn_mode or progressive_gmm_environment_mode) else None,
            "env_route_q_diagnostic": q_route_diag,
            "progressive_cluster_id": output.get("progressive_cluster_id"),
            "predicted_risk": output.get("predicted_risk"),
            "risk_route_q": output.get("risk_route_q"),
            "route_features": output.get("route_features"),
            "fallback_q": output.get("fallback_q"),
            "primary_uses_env_fusion": output.get("primary_uses_env_fusion"),
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
            "fpem/env_route_L_proto_align": ref.new_zeros(()),
            "fpem/env_route_entropy": ref.new_zeros(()),
            "fpem/env_route_q_max_mean": ref.new_zeros(()),
            "fpem/hyper_alpha_mean": ref.new_zeros(()),
            "fpem/hyper_delta_norm": ref.new_zeros(()),
            "fpem/hyper_route_proto_mode_uniform_warmup": ref.new_zeros(()),
            "fpem/hyper_route_proto_mode_uniform_fixed": ref.new_zeros(()),
            "fpem/hyper_route_proto_mode_sinkhorn": ref.new_zeros(()),
            "fpem/hyper_route_proto_mode_softmax": ref.new_zeros(()),
            "fpem/fallback_q_mean": ref.new_zeros(()),
            "fpem/fallback_q_max": ref.new_zeros(()),
            "fpem/env_q_sum_mean": ref.new_zeros(()),
            "fpem/oracle_fallback_rate": ref.new_zeros(()),
            "fpem/route_count_fallback": ref.new_zeros(()),
            "fpem/env_route_head_mode": ref.new_tensor(1.0 if self._use_hyper_inv_film() else 0.0),
            "fpem/route_entropy_mean": ref.new_zeros(()),
            "fpem/route_mean_distribution_entropy": ref.new_zeros(()),
            "fpem/effective_expert_number": ref.new_zeros(()),
            "fpem/max_expert_usage_ratio": ref.new_zeros(()),
            "fpem/min_expert_usage_ratio": ref.new_zeros(()),
            "fpem/prototype_pairwise_cosine": ref.new_zeros(()),
            "fpem/expert_prediction_pairwise_cosine": ref.new_zeros(()),
            "fpem/expert_collapse_warning": ref.new_zeros(()),
            "fpem/env_route_target_mode_env_prototype": ref.new_zeros(()),
            "fpem/env_route_target_mode_hybrid": ref.new_zeros(()),
            "fpem/env_route_hybrid_alpha": ref.new_zeros(()),
            "fpem/env_route_train_mode_warmup_risk_sinkhorn": ref.new_zeros(()),
            "fpem/warmup_risk_stage_warmup": ref.new_zeros(()),
            "fpem/warmup_risk_stage_soft": ref.new_zeros(()),
            "fpem/warmup_risk_stage_hard": ref.new_zeros(()),
            "fpem/warmup_risk_selected_loss": ref.new_zeros(()),
            "fpem/warmup_risk_uniform_common_loss": ref.new_zeros(()),
            "fpem/warmup_risk_lambda_common": ref.new_zeros(()),
            "fpem/warmup_risk_total_common_term": ref.new_zeros(()),
            "fpem/warmup_risk_temperature": ref.new_zeros(()),
            "fpem/warmup_risk_temperature_progress": ref.new_zeros(()),
            "fpem/uniform_expert_mae": ref.new_zeros(()),
            "fpem/risk_routed_mae": ref.new_zeros(()),
            "fpem/risk_prediction_mae": ref.new_zeros(()),
            "fpem/risk_loss": ref.new_zeros(()),
            "fpem/risk_ranking_loss": ref.new_zeros(()),
            "fpem/risk_ranking_accuracy": ref.new_zeros(()),
            "fpem/risk_oracle_argmin_accuracy": ref.new_zeros(()),
            "fpem/risk_router_temperature": ref.new_zeros(()),
            "fpem/risk_router_lambda": ref.new_zeros(()),
            "fpem/risk_router_pairwise_lambda": ref.new_zeros(()),
            "fpem/env_route_train_mode_progressive_gmm_environment": ref.new_zeros(()),
            "fpem/top_level_prediction_source_progressive_gmm_teacher_hard": ref.new_zeros(()),
            "fpem/progressive_assignment_uses_target_or_prediction_error": ref.new_zeros(()),
            "fpem/progressive_partition_ready": ref.new_zeros(()),
            "fpem/progressive_selected_k": ref.new_zeros(()),
            "fpem/progressive_active_expert_count": ref.new_zeros(()),
            "fpem/progressive_last_partition_epoch": ref.new_zeros(()),
            "fpem/progressive_lambda_common": ref.new_zeros(()),
            "fpem/progressive_common_term": ref.new_zeros(()),
            "fpem/progressive_compactness_loss": ref.new_zeros(()),
            "fpem/progressive_consistency_loss": ref.new_zeros(()),
            "fpem/progressive_cluster_entropy": ref.new_zeros(()),
            "fpem/progressive_assignment_change_ratio": ref.new_zeros(()),
            "fpem/progressive_hungarian_overlap_score": ref.new_zeros(()),
            "fpem/progressive_partition_update_time_sec": ref.new_zeros(()),
            "fpem/progressive_routed_mae": ref.new_zeros(()),
            "fpem/progressive_gmm_environment_total_replaces_primary": ref.new_zeros(()),
        }
        for idx in range(self.fpem_env_route_k):
            logs[f"fpem/env_route_count_head_{idx}"] = ref.new_zeros(())
            logs[f"fpem/env_route_oracle_count_head_{idx}"] = ref.new_zeros(())
            logs[f"fpem/route_count_env_head_{idx}"] = ref.new_zeros(())
            logs[f"fpem/route_soft_mean_expert_{idx}"] = ref.new_zeros(())
            logs[f"fpem/route_hard_count_expert_{idx}"] = ref.new_zeros(())
            logs[f"fpem/hard_count_expert_{idx}"] = ref.new_zeros(())
            logs[f"fpem/sinkhorn_soft_col_mass_expert_{idx}"] = ref.new_zeros(())
            logs[f"fpem/progressive_active_expert_{idx}"] = ref.new_zeros(())
            logs[f"fpem/progressive_cluster_size_{idx}"] = ref.new_zeros(())
            logs[f"fpem/progressive_cluster_ratio_{idx}"] = ref.new_zeros(())
            logs[f"fpem/progressive_cluster_to_expert_{idx}"] = ref.new_zeros(())
            logs[f"fpem/progressive_bic_k{idx + 1}"] = ref.new_zeros(())
            logs[f"fpem/progressive_bic_score_k{idx + 1}"] = ref.new_zeros(())
            logs[f"fpem/progressive_bic_rejected_k{idx + 1}"] = ref.new_zeros(())
            logs[f"fpem/hyper_alpha_head_{idx}"] = ref.new_zeros(())
            logs[f"fpem/hyper_gamma_norm_head_{idx}"] = ref.new_zeros(())
            logs[f"fpem/hyper_beta_norm_head_{idx}"] = ref.new_zeros(())
            for expert_idx in range(self.fpem_env_route_k):
                logs[f"fpem/expert_cross_mae_group_{idx}_expert_{expert_idx}"] = ref.new_zeros(())
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
            "fpem/swap_prediction_delta": ref.new_zeros(()),
            "fpem/swap_route_delta": ref.new_zeros(()),
            "fpem/swap_hyper_alpha_delta": ref.new_zeros(()),
            "fpem/swap_hyper_delta_norm": ref.new_zeros(()),
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

    def clear_fpem_runtime_cache(self):
        """Release per-batch tensors that may still reference an autograd graph."""
        self.latest_fpem_outputs = {}
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
