import torch
import torch.nn.functional as F
from contextlib import nullcontext


def autocast_disabled():
    if torch.cuda.is_available():
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda", enabled=False)
        return torch.cuda.amp.autocast(enabled=False)
    return nullcontext()


def tensor_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def masked_channel_mae(pred, target, mask_value=5.0):
    err = (pred - target).abs()
    mask = target > mask_value
    if bool(mask.any()):
        return err.masked_select(mask).mean()
    return err.mean()


def weighted_flow_mae(pred, target, scaler, yita=0.5, mask_value=5.0):
    pred_raw = scaler.inverse_transform(pred)
    target_raw = scaler.inverse_transform(target)
    loss = float(yita) * masked_channel_mae(pred_raw[..., 0], target_raw[..., 0], mask_value)
    loss = loss + (1.0 - float(yita)) * masked_channel_mae(pred_raw[..., 1], target_raw[..., 1], mask_value)
    return loss


def target_mask(target_raw):
    return target_raw[..., 0] > 5.0


def flow_error_view(pred, target, scaler, yita=0.5):
    pred_raw = scaler.inverse_transform(pred)
    target_raw = scaler.inverse_transform(target)
    err = float(yita) * (pred_raw[..., 0] - target_raw[..., 0]).abs()
    err = err + (1.0 - float(yita)) * (pred_raw[..., 1] - target_raw[..., 1]).abs()
    return err, target_mask(target_raw)


def masked_mean(value, mask):
    mask_f = mask.to(dtype=value.dtype)
    while mask_f.dim() < value.dim():
        mask_f = mask_f.unsqueeze(1)
    mask_f = mask_f.expand_as(value)
    return (value * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def head_prediction_losses(y_heads, target, scaler, yita=0.5):
    pred_raw = scaler.inverse_transform(y_heads)
    target_raw = scaler.inverse_transform(target).unsqueeze(1)
    err = float(yita) * (pred_raw[..., 0] - target_raw[..., 0]).abs()
    err = err + (1.0 - float(yita)) * (pred_raw[..., 1] - target_raw[..., 1]).abs()
    mask = target_mask(target_raw.squeeze(1)).unsqueeze(1).to(dtype=err.dtype)
    return (err * mask).sum(dim=(2, 3)) / mask.sum(dim=(2, 3)).clamp_min(1.0)


def gradient_compat_assignment(y_heads, target, scaler, norm_mode="batch_head"):
    with torch.no_grad():
        pred_raw = scaler.inverse_transform(y_heads.detach())
        target_raw = scaler.inverse_transform(target).detach().unsqueeze(1)
        score = ((pred_raw - target_raw) * pred_raw).mean(dim=tuple(range(2, pred_raw.dim())))
        if norm_mode == "none":
            score_norm = score
        elif norm_mode == "sample_head":
            score_norm = (score - score.mean(dim=1, keepdim=True)) / score.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
        else:
            score_norm = score / score.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        cost = score_norm.abs()
        pseudo_head = cost.argmin(dim=1)
    return pseudo_head.detach(), cost.detach(), score.detach()


def _sinkhorn_balanced(scores, num_iters=5):
    """Return a [B,K] matrix with rows ~= 1 and columns ~= B/K."""
    with torch.no_grad():
        bsz, num_experts = scores.shape
        scores = scores.float()
        scores = scores - scores.max(dim=1, keepdim=True).values
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


def _balanced_hard_idx_from_soft(soft_q):
    """Greedy balanced rounding for the Sinkhorn matrix.

    Plain argmax is kept as the primary preference, but exactly uniform
    probabilities can otherwise send every sample to expert 0.  This deterministic
    repair keeps the prediction-error Sinkhorn mode useful from the first batch.
    """
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
        _, order = torch.sort(soft_q.reshape(-1), descending=True)
        used = torch.zeros(bsz, dtype=torch.bool, device=soft_q.device)
        counts = torch.zeros(num_experts, dtype=torch.long, device=soft_q.device)
        for flat_idx in order:
            sample = flat_idx // num_experts
            expert = flat_idx - sample * num_experts
            if used[sample]:
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


def hard_prediction_sinkhorn_assignment(loss_head, temperature=1.0, num_iters=5):
    """Assign samples to prediction experts using detached loss + Sinkhorn."""
    with torch.no_grad():
        cost = loss_head.detach().float()
        cost = (cost - cost.mean(dim=1, keepdim=True)) / cost.std(
            dim=1, unbiased=False, keepdim=True
        ).clamp_min(1e-6)
        scores = -cost / max(float(temperature), 1e-6)
        soft_q = _sinkhorn_balanced(scores, num_iters=num_iters).to(dtype=loss_head.dtype)
        hard_idx, repaired = _balanced_hard_idx_from_soft(soft_q)
        hard_q = F.one_hot(hard_idx, num_classes=loss_head.shape[1]).to(
            device=loss_head.device, dtype=loss_head.dtype
        )
    return soft_q.detach(), hard_idx.detach(), hard_q.detach(), repaired.detach()


def route_losses(route_out, target, scaler, args):
    q = route_out["env_route_q"]
    logits = route_out["env_route_logits"]
    mode = str(route_out.get("route_head_mode", getattr(args, "fpem_env_route_head_mode", "concat_input"))).lower()
    is_hyper = mode in {
        "hyper_inv_film",
        "hyper_inv_film_proto",
        "hyper_inv_film_proto_concat",
        "hyper_inv_film_proto_input_concat",
        "hyper_inv_film_proto_input_add",
    }
    y_heads = route_out["y_hyper_heads"] if is_hyper else route_out["y_route_heads"]
    k_env = y_heads.shape[1]
    k = q.shape[-1]
    zero = target.new_zeros(())
    logs = {
        "fpem/env_route_loss": zero,
        "fpem/env_route_L_final": zero,
        "fpem/env_route_L_global": zero,
        "fpem/env_route_L_route_soft": zero,
        "fpem/env_route_L_expert": zero,
        "fpem/env_route_L_router_oracle": zero,
        "fpem/env_route_L_balance": zero,
        "fpem/env_route_L_diverse": zero,
        "fpem/env_route_L_proto_align": zero,
        "fpem/env_route_entropy": zero,
        "fpem/env_route_q_max_mean": zero,
        "fpem/env_route_train_mode_gradient_compat": zero,
        "fpem/env_route_train_mode_hard_prediction_sinkhorn": zero,
        "fpem/env_route_gradient_compat_aux": zero,
        "fpem/hard_sinkhorn_enabled": zero,
        "fpem/sinkhorn_soft_entropy": zero,
        "fpem/sinkhorn_selected_loss": zero,
        "fpem/router_assignment_accuracy": zero,
        "fpem/router_assignment_agreement": zero,
        "fpem/oracle_hard_mae": zero,
        "fpem/router_hard_mae": zero,
        "fpem/router_regret": zero,
        "fpem/hard_sinkhorn_balance_repaired": zero,
        "fpem/sinkhorn_soft_row_sum_mean": zero,
        "fpem/hyper_alpha_mean": zero,
        "fpem/hyper_delta_norm": zero,
        "fpem/hyper_route_proto_mode_uniform_warmup": zero,
        "fpem/hyper_route_proto_mode_uniform_fixed": zero,
        "fpem/hyper_route_proto_mode_sinkhorn": zero,
        "fpem/hyper_route_proto_mode_softmax": zero,
        "fpem/fallback_q_mean": zero,
        "fpem/fallback_q_max": zero,
        "fpem/env_q_sum_mean": zero,
        "fpem/oracle_fallback_rate": zero,
        "fpem/route_count_fallback": zero,
        "fpem/env_route_head_mode": zero,
        "fpem/route_entropy_mean": zero,
        "fpem/route_mean_distribution_entropy": zero,
        "fpem/effective_expert_number": zero,
        "fpem/max_expert_usage_ratio": zero,
        "fpem/min_expert_usage_ratio": zero,
        "fpem/prototype_pairwise_cosine": zero,
        "fpem/expert_prediction_pairwise_cosine": zero,
        "fpem/expert_collapse_warning": zero,
        "fpem/env_route_target_mode_env_prototype": zero,
        "fpem/env_route_target_mode_hybrid": zero,
        "fpem/env_route_hybrid_alpha": zero,
    }
    for idx in range(k_env):
        logs[f"fpem/env_route_count_head_{idx}"] = zero
        logs[f"fpem/env_route_oracle_count_head_{idx}"] = zero
        logs[f"fpem/route_count_env_head_{idx}"] = zero
        logs[f"fpem/route_soft_mean_expert_{idx}"] = zero
        logs[f"fpem/route_hard_count_expert_{idx}"] = zero
        logs[f"fpem/hard_count_expert_{idx}"] = zero
        logs[f"fpem/sinkhorn_soft_col_mass_expert_{idx}"] = zero
        logs[f"fpem/hyper_alpha_head_{idx}"] = zero
        logs[f"fpem/hyper_gamma_norm_head_{idx}"] = zero
        logs[f"fpem/hyper_beta_norm_head_{idx}"] = zero
        for expert_idx in range(k_env):
            logs[f"fpem/expert_cross_mae_group_{idx}_expert_{expert_idx}"] = zero

    train_mode = str(getattr(args, "fpem_env_route_train_mode", "soft_oracle")).lower()
    grad_compat = train_mode == "gradient_compat_route"
    hard_pred_sinkhorn = train_mode == "hard_prediction_sinkhorn"
    force_uniform = getattr(args, "fpem_force_uniform_route", False)
    if isinstance(force_uniform, str):
        force_uniform = force_uniform.lower() in {"1", "true", "yes", "y", "on"}
    else:
        force_uniform = bool(force_uniform)
    q_prob = q.float().clamp(1e-8, 1.0)
    if is_hyper:
        y_candidates = route_out["y_candidates"]
        loss_candidates = head_prediction_losses(y_candidates, target, scaler, getattr(args, "yita", 0.5))
        loss_env = loss_candidates[:, 1:] if q.shape[1] == k_env + 1 else loss_candidates
        q_env = q[:, 1:] if q.shape[1] == k_env + 1 else q
        q_env_sum = q_env.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        q_env_norm = q_env / q_env_sum
        q_env_prob = q_env_norm.float().clamp(1e-8, 1.0)
        entropy = -(q_env_prob * q_env_prob.log()).sum(dim=-1).mean().to(dtype=target.dtype)
        loss_head = loss_env
    else:
        loss_head = head_prediction_losses(y_heads, target, scaler, getattr(args, "yita", 0.5))
        q_env = q
        q_env_norm = q
        entropy = -(q_prob * q_prob.log()).sum(dim=-1).mean().to(dtype=target.dtype)

    if hard_pred_sinkhorn:
        training = bool(route_out.get("training", False))
        router_logits = route_out.get("env_route_logits_prediction", logits)
        if not torch.is_tensor(router_logits):
            router_logits = logits
        router_q = torch.softmax(router_logits.float(), dim=-1).to(dtype=y_heads.dtype)
        router_idx = router_q.detach().argmax(dim=-1)
        router_hard_q = F.one_hot(router_idx, num_classes=k_env).to(device=y_heads.device, dtype=y_heads.dtype)
        router_prediction = (
            router_hard_q.view(router_hard_q.shape[0], router_hard_q.shape[1], 1, 1, 1)
            * y_heads
        ).sum(dim=1)
        router_hard_mae = weighted_flow_mae(router_prediction, target, scaler, getattr(args, "yita", 0.5))

        if training:
            soft_q, hard_idx, hard_q, repaired = hard_prediction_sinkhorn_assignment(
                loss_head,
                temperature=float(getattr(args, "fpem_env_route_sinkhorn_tau", 1.0)),
                num_iters=int(getattr(args, "fpem_env_route_sinkhorn_iters", 5)),
            )
            q_oracle = hard_q
            selected_prediction = (
                hard_q.view(hard_q.shape[0], hard_q.shape[1], 1, 1, 1).detach()
                * y_heads
            ).sum(dim=1)
            expert_loss = (hard_q.detach() * loss_head).sum(dim=-1).mean()
            selected_prediction_loss = expert_loss
            router_oracle_loss = F.cross_entropy(router_logits.float(), hard_idx.detach())
            assignment_match = (router_idx == hard_idx).to(dtype=target.dtype).mean()
            sinkhorn_soft_entropy = (
                -(soft_q.float().clamp(1e-8, 1.0) * soft_q.float().clamp(1e-8, 1.0).log()).sum(dim=-1).mean()
            ).to(dtype=target.dtype)
            hard_counts = torch.bincount(hard_idx, minlength=k_env).to(dtype=target.dtype, device=target.device)[:k_env]
            hard_count_source = hard_counts
            oracle_hard_mae = weighted_flow_mae(selected_prediction, target, scaler, getattr(args, "yita", 0.5))
            lambda_router = float(getattr(args, "fpem_env_route_lambda_router_oracle", 0.5))
            total = selected_prediction_loss + lambda_router * router_oracle_loss
            soft_col_mass = soft_q.detach().sum(dim=0).to(dtype=target.dtype)
            soft_row_sum = soft_q.detach().sum(dim=1).mean().to(dtype=target.dtype)
        else:
            hard_idx = router_idx.detach()
            q_oracle = router_hard_q.detach()
            soft_q = router_q.detach()
            repaired = target.new_zeros(())
            expert_loss = router_hard_mae.detach()
            selected_prediction_loss = router_hard_mae.detach()
            router_oracle_loss = zero
            assignment_match = target.new_zeros(())
            sinkhorn_soft_entropy = target.new_zeros(())
            hard_count_source = torch.bincount(hard_idx, minlength=k_env).to(dtype=target.dtype, device=target.device)[:k_env]
            oracle_hard_mae = router_hard_mae.detach()
            total = zero
            soft_col_mass = router_q.detach().sum(dim=0).to(dtype=target.dtype)
            soft_row_sum = router_q.detach().sum(dim=1).mean().to(dtype=target.dtype)

        router_regret = (router_hard_mae.detach() - oracle_hard_mae.detach()).to(dtype=target.dtype)
        hard = hard_idx.detach()
        counts = torch.bincount(router_idx.detach(), minlength=k_env).to(dtype=target.dtype, device=target.device)[:k_env]
        q_mean_env = router_q.detach().mean(dim=0).to(dtype=target.dtype)[:k_env]
        q_mean_prob = q_mean_env.float().clamp_min(1e-8)
        mean_dist_entropy = -(q_mean_prob * q_mean_prob.log()).sum().to(dtype=target.dtype)
        effective_expert_number = torch.exp(mean_dist_entropy.float()).to(dtype=target.dtype)
        usage_ratio = hard_count_source / hard_count_source.sum().clamp_min(1.0)
        max_usage_ratio = usage_ratio.max() if usage_ratio.numel() else zero
        min_usage_ratio = usage_ratio.min() if usage_ratio.numel() else zero
        if k_env > 1:
            flat = scaler.inverse_transform(y_heads).permute(1, 0, 2, 3, 4).reshape(k_env, -1)
            flat = F.normalize(flat.float(), dim=-1, eps=1e-8)
            sim = flat.matmul(flat.t())
            expert_pairwise_cosine = sim[~torch.eye(k_env, dtype=torch.bool, device=sim.device)].mean().to(dtype=target.dtype)
        else:
            expert_pairwise_cosine = zero

        logs.update({
            "fpem/env_route_loss": total.detach(),
            "fpem/env_route_L_final": selected_prediction_loss.detach(),
            "fpem/env_route_L_route_soft": zero,
            "fpem/env_route_L_expert": expert_loss.detach(),
            "fpem/env_route_L_router_oracle": router_oracle_loss.detach(),
            "fpem/env_route_L_balance": zero,
            "fpem/env_route_L_diverse": zero,
            "fpem/env_route_L_proto_align": zero,
            "fpem/env_route_entropy": mean_dist_entropy.detach(),
            "fpem/route_entropy_mean": mean_dist_entropy.detach(),
            "fpem/route_mean_distribution_entropy": mean_dist_entropy.detach(),
            "fpem/effective_expert_number": effective_expert_number.detach(),
            "fpem/max_expert_usage_ratio": max_usage_ratio.detach(),
            "fpem/min_expert_usage_ratio": min_usage_ratio.detach(),
            "fpem/expert_prediction_pairwise_cosine": expert_pairwise_cosine.detach(),
            "fpem/env_route_q_max_mean": router_q.detach().max(dim=-1).values.mean(),
            "fpem/env_route_train_mode_hard_prediction_sinkhorn": target.new_tensor(1.0),
            "fpem/hard_sinkhorn_enabled": target.new_tensor(float(training)),
            "fpem/sinkhorn_soft_entropy": sinkhorn_soft_entropy.detach(),
            "fpem/sinkhorn_selected_loss": selected_prediction_loss.detach(),
            "fpem/router_assignment_accuracy": assignment_match.detach(),
            "fpem/router_assignment_agreement": assignment_match.detach(),
            "fpem/oracle_hard_mae": oracle_hard_mae.detach(),
            "fpem/router_hard_mae": router_hard_mae.detach(),
            "fpem/router_regret": router_regret.detach(),
            "fpem/hard_sinkhorn_balance_repaired": repaired.detach(),
            "fpem/sinkhorn_soft_row_sum_mean": soft_row_sum.detach(),
        })
        for idx in range(k_env):
            logs[f"fpem/env_route_count_head_{idx}"] = counts[idx].detach()
            logs[f"fpem/env_route_oracle_count_head_{idx}"] = hard_count_source[idx].detach()
            logs[f"fpem/route_count_env_head_{idx}"] = counts[idx].detach()
            logs[f"fpem/route_soft_mean_expert_{idx}"] = q_mean_env[idx].detach()
            logs[f"fpem/route_hard_count_expert_{idx}"] = hard_count_source[idx].detach()
            logs[f"fpem/hard_count_expert_{idx}"] = hard_count_source[idx].detach()
            logs[f"fpem/sinkhorn_soft_col_mass_expert_{idx}"] = soft_col_mass[idx].detach()
            hyper_alpha = route_out.get("hyper_alpha", None)
            if torch.is_tensor(hyper_alpha) and hyper_alpha.shape[-1] > idx:
                logs[f"fpem/hyper_alpha_head_{idx}"] = hyper_alpha[:, idx].detach().mean()
            gamma_norm = route_out.get("hyper_gamma_norm_per_head", None)
            if torch.is_tensor(gamma_norm) and gamma_norm.shape[0] > idx:
                logs[f"fpem/hyper_gamma_norm_head_{idx}"] = gamma_norm[idx].detach()
            beta_norm = route_out.get("hyper_beta_norm_per_head", None)
            if torch.is_tensor(beta_norm) and beta_norm.shape[0] > idx:
                logs[f"fpem/hyper_beta_norm_head_{idx}"] = beta_norm[idx].detach()
            group_mask = hard == idx
            for expert_idx in range(k_env):
                if bool(group_mask.any()):
                    value = loss_head[group_mask, expert_idx].mean()
                else:
                    value = zero
                logs[f"fpem/expert_cross_mae_group_{idx}_expert_{expert_idx}"] = value.detach()
        return total, logs, q_oracle.detach()

    if grad_compat and not force_uniform:
        pseudo_env_head, gc_cost, gc_score = gradient_compat_assignment(
            y_heads, target, scaler, str(getattr(args, "fpem_env_route_grad_norm_mode", "batch_head")).lower()
        )
        if is_hyper and q.shape[1] == k_env + 1:
            pseudo_head = pseudo_env_head + 1
        else:
            pseudo_head = pseudo_env_head
        q_oracle = F.one_hot(pseudo_head, num_classes=k).to(dtype=q.dtype, device=q.device)
        q_oracle_env = q_oracle[:, 1:] if is_hyper and q.shape[1] == k_env + 1 else q_oracle
        expert_loss = (q_oracle_env * loss_head).sum(dim=-1).mean()
        router_oracle_loss = F.cross_entropy(logits.float(), pseudo_head)
        grad_aux = loss_head.mean()
        if str(getattr(args, "fpem_use_gradcompat_aux", False)).lower() in {"1", "true", "yes", "y", "on"}:
            grad_aux_weighted = float(getattr(args, "fpem_lambda_gradcompat_aux", 0.0)) * grad_aux
        else:
            grad_aux_weighted = zero
    else:
        oracle_tau = max(float(getattr(args, "fpem_env_route_oracle_tau", 0.3)), 1e-6)
        if is_hyper:
            q_oracle = torch.softmax(-loss_candidates.detach() / oracle_tau, dim=1)
            q_oracle_env = q_oracle[:, 1:] if q.shape[1] == k_env + 1 else q_oracle
        else:
            q_oracle = torch.softmax(-loss_head.detach() / oracle_tau, dim=1)
            q_oracle_env = q_oracle
        expert_loss = (q_oracle_env * loss_env).sum(dim=-1).mean() if is_hyper else (q_oracle * loss_head).sum(dim=-1).mean()
        router_oracle_loss = (q_oracle * ((q_oracle.float().clamp(1e-8, 1.0)).log() - q_prob.log())).sum(dim=1).mean()
        grad_aux = zero
        grad_aux_weighted = zero

    if force_uniform:
        q_oracle = q.detach()
        q_oracle_env = q_oracle[:, 1:] if is_hyper and q.shape[1] == k_env + 1 else q_oracle
        expert_loss = (q_oracle_env * loss_head).sum(dim=-1).mean()
        router_oracle_loss = zero
        grad_aux = zero
        grad_aux_weighted = zero

    target_mode = str(
        route_out.get("env_route_target_mode", getattr(args, "fpem_env_route_target_mode", "prediction_oracle"))
    ).lower()
    proto_q = route_out.get("env_route_q_prototype", None)
    if torch.is_tensor(proto_q) and target_mode in {"env_prototype", "hybrid"} and not force_uniform:
        proto_q = proto_q.to(device=q.device, dtype=q.dtype)
        proto_q = proto_q / proto_q.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if is_hyper and q.shape[1] == k_env + 1 and proto_q.shape[1] == k_env:
            proto_full = torch.cat([q.new_zeros(q.shape[0], 1), proto_q], dim=1)
        else:
            proto_full = proto_q
        if target_mode == "hybrid":
            alpha = route_out.get("env_route_hybrid_alpha", None)
            alpha_value = float(alpha.detach().cpu().item()) if torch.is_tensor(alpha) else float(
                getattr(args, "fpem_env_route_hybrid_alpha", 1.0)
            )
            alpha_value = max(0.0, min(1.0, alpha_value))
            q_oracle = alpha_value * proto_full.detach() + (1.0 - alpha_value) * q_oracle.detach()
            logs["fpem/env_route_target_mode_hybrid"] = target.new_tensor(1.0)
            logs["fpem/env_route_hybrid_alpha"] = target.new_tensor(alpha_value)
        else:
            q_oracle = proto_full.detach()
            logs["fpem/env_route_target_mode_env_prototype"] = target.new_tensor(1.0)
            logs["fpem/env_route_hybrid_alpha"] = target.new_tensor(1.0)
        q_oracle = q_oracle / q_oracle.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        q_oracle_env = q_oracle[:, 1:] if is_hyper and q.shape[1] == k_env + 1 else q_oracle
        expert_loss = (q_oracle_env * loss_head).sum(dim=-1).mean()
        router_oracle_loss = (q_oracle * ((q_oracle.float().clamp(1e-8, 1.0)).log() - q_prob.log())).sum(dim=1).mean()

    final_loss = weighted_flow_mae(route_out["prediction"], target, scaler, getattr(args, "yita", 0.5))
    global_loss = weighted_flow_mae(route_out["y_global"], target, scaler, getattr(args, "yita", 0.5))
    route_soft_loss = weighted_flow_mae(route_out["y_route"], target, scaler, getattr(args, "yita", 0.5))
    q_mean = q_env_norm.mean(dim=0) if is_hyper else q.mean(dim=0)
    balance_loss = (q_mean - (1.0 / float(k_env))).pow(2).mean() if k_env > 1 else zero
    if k_env > 1:
        flat = scaler.inverse_transform(y_heads).permute(1, 0, 2, 3, 4).reshape(k_env, -1)
        flat = F.normalize(flat.float(), dim=-1, eps=1e-8)
        sim = flat.matmul(flat.t())
        diverse_loss = sim[~torch.eye(k_env, dtype=torch.bool, device=sim.device)].mean().to(dtype=target.dtype)
    else:
        diverse_loss = zero
    proto_align_loss = zero
    prototypes_for_align = route_out.get("env_prototypes", None)
    e_for_align = route_out.get("E_useful", None)
    proto_q_for_align = route_out.get("env_route_q_prototype", None)
    if (
        torch.is_tensor(prototypes_for_align)
        and torch.is_tensor(e_for_align)
        and torch.is_tensor(proto_q_for_align)
        and prototypes_for_align.shape[0] == k_env
        and proto_q_for_align.shape[-1] == k_env
    ):
        pooled = F.normalize(e_for_align.mean(dim=1).float(), dim=-1, eps=1e-8)
        proto = F.normalize(prototypes_for_align.float(), dim=-1, eps=1e-8)
        proto_sim = pooled.matmul(proto.t()).to(dtype=target.dtype)
        proto_q_align = proto_q_for_align.to(device=proto_sim.device, dtype=proto_sim.dtype)
        proto_q_align = proto_q_align / proto_q_align.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        proto_align_loss = -(proto_q_align * proto_sim).sum(dim=-1).mean()

    lambda_global_for_total = 0.0 if is_hyper else float(getattr(args, "fpem_env_route_lambda_global", 0.2))
    total = (
        float(getattr(args, "fpem_env_route_lambda_final", 1.0)) * final_loss
        + lambda_global_for_total * global_loss
        + float(getattr(args, "fpem_env_route_lambda_route_soft", 0.5)) * route_soft_loss
        + float(getattr(args, "fpem_env_route_lambda_expert", 0.2)) * expert_loss
        + float(getattr(args, "fpem_env_route_lambda_router_oracle", 0.5)) * router_oracle_loss
        + float(getattr(args, "fpem_env_route_lambda_balance", 0.01)) * balance_loss
        + float(getattr(args, "fpem_env_route_lambda_diverse", 0.001)) * diverse_loss
        + float(getattr(args, "fpem_env_route_lambda_proto_align", 0.0)) * proto_align_loss
        + float(getattr(args, "fpem_env_route_lambda_entropy", 0.0)) * entropy
        + grad_aux_weighted
    )

    hard = q.detach().argmax(dim=-1)
    oracle = q_oracle.detach().argmax(dim=-1)
    counts = torch.bincount(hard, minlength=k).to(dtype=target.dtype, device=target.device)
    oracle_counts = torch.bincount(oracle, minlength=k).to(dtype=target.dtype, device=target.device)
    fallback_offset = 1 if is_hyper and q.shape[1] == k_env + 1 else 0
    env_hard = hard - fallback_offset if fallback_offset == 1 else hard
    env_hard = env_hard.clamp_min(0)
    env_counts = torch.bincount(env_hard, minlength=k_env).to(dtype=target.dtype, device=target.device)[:k_env]
    env_counts = env_counts if fallback_offset == 0 else env_counts
    q_mean_env = q_env_norm.detach().mean(dim=0) if is_hyper else q.detach().mean(dim=0)
    q_mean_env = q_mean_env[:k_env]
    q_mean_prob = q_mean_env.float().clamp_min(1e-8)
    mean_dist_entropy = -(q_mean_prob * q_mean_prob.log()).sum().to(dtype=target.dtype)
    effective_expert_number = torch.exp(mean_dist_entropy.float()).to(dtype=target.dtype)
    usage_ratio = env_counts / env_counts.sum().clamp_min(1.0)
    max_usage_ratio = usage_ratio.max() if usage_ratio.numel() else zero
    min_usage_ratio = usage_ratio.min() if usage_ratio.numel() else zero
    expert_pairwise_cosine = diverse_loss.detach()
    prototypes = route_out.get("env_prototypes", None)
    if torch.is_tensor(prototypes) and prototypes.shape[0] > 1:
        proto = F.normalize(prototypes.detach().float(), dim=-1, eps=1e-8)
        proto_sim = proto.matmul(proto.t())
        prototype_pairwise_cosine = proto_sim[~torch.eye(proto.shape[0], dtype=torch.bool, device=proto.device)].mean().to(dtype=target.dtype)
    else:
        prototype_pairwise_cosine = zero
    logs.update({
        "fpem/env_route_loss": total.detach(),
        "fpem/env_route_L_final": final_loss.detach(),
        "fpem/env_route_L_global": global_loss.detach(),
        "fpem/env_route_L_route_soft": route_soft_loss.detach(),
        "fpem/env_route_L_expert": expert_loss.detach(),
        "fpem/env_route_L_router_oracle": router_oracle_loss.detach(),
        "fpem/env_route_L_balance": balance_loss.detach(),
        "fpem/env_route_L_diverse": diverse_loss.detach(),
        "fpem/env_route_L_proto_align": proto_align_loss.detach(),
        "fpem/env_route_entropy": entropy.detach(),
        "fpem/route_entropy_mean": entropy.detach(),
        "fpem/route_mean_distribution_entropy": mean_dist_entropy.detach(),
        "fpem/effective_expert_number": effective_expert_number.detach(),
        "fpem/max_expert_usage_ratio": max_usage_ratio.detach(),
        "fpem/min_expert_usage_ratio": min_usage_ratio.detach(),
        "fpem/prototype_pairwise_cosine": prototype_pairwise_cosine.detach(),
        "fpem/expert_prediction_pairwise_cosine": expert_pairwise_cosine.detach(),
        "fpem/env_route_q_max_mean": q.detach().max(dim=-1).values.mean(),
        "fpem/env_route_train_mode_gradient_compat": target.new_tensor(float(grad_compat and not force_uniform)),
        "fpem/env_route_gradient_compat_aux": grad_aux.detach(),
        "fpem/hyper_alpha_mean": route_out.get("hyper_alpha", zero).detach().mean() if torch.is_tensor(route_out.get("hyper_alpha", None)) else zero,
        "fpem/hyper_delta_norm": route_out.get("hyper_delta_norm", zero).detach() if torch.is_tensor(route_out.get("hyper_delta_norm", None)) else zero,
        "fpem/env_route_head_mode": target.new_tensor(1.0 if is_hyper else 0.0),
    })
    proto_mode = str(route_out.get("env_route_proto_mode", "")).lower()
    if proto_mode:
        logs["fpem/hyper_route_proto_mode_uniform_warmup"] = target.new_tensor(float(proto_mode == "uniform_warmup"))
        logs["fpem/hyper_route_proto_mode_uniform_fixed"] = target.new_tensor(float(proto_mode == "uniform_fixed"))
        logs["fpem/hyper_route_proto_mode_sinkhorn"] = target.new_tensor(float(proto_mode == "sinkhorn"))
        logs["fpem/hyper_route_proto_mode_softmax"] = target.new_tensor(float(proto_mode in {"softmax", "softmax_fallback"}))
    if is_hyper and fallback_offset == 1:
        fallback_q = q[:, 0].detach()
        logs.update({
            "fpem/fallback_q_mean": fallback_q.mean(),
            "fpem/fallback_q_max": fallback_q.max(),
            "fpem/env_q_sum_mean": q[:, 1:].detach().sum(dim=-1).mean(),
            "fpem/oracle_fallback_rate": (oracle == 0).to(dtype=target.dtype).mean(),
            "fpem/route_count_fallback": counts[0].detach(),
        })
    if grad_compat and not force_uniform:
        logs["fpem/env_route_gradient_compat_cost_mean"] = gc_cost.mean()
        logs["fpem/env_route_gradient_compat_score_mean"] = gc_score.mean()
    for idx in range(k_env):
        route_idx = idx + fallback_offset
        logs[f"fpem/env_route_count_head_{idx}"] = counts[route_idx].detach()
        logs[f"fpem/env_route_oracle_count_head_{idx}"] = oracle_counts[route_idx].detach()
        logs[f"fpem/route_count_env_head_{idx}"] = counts[route_idx].detach()
        logs[f"fpem/route_soft_mean_expert_{idx}"] = q_mean_env[idx].detach()
        logs[f"fpem/route_hard_count_expert_{idx}"] = env_counts[idx].detach()
        hyper_alpha = route_out.get("hyper_alpha", None)
        if torch.is_tensor(hyper_alpha) and hyper_alpha.shape[-1] > idx:
            logs[f"fpem/hyper_alpha_head_{idx}"] = hyper_alpha[:, idx].detach().mean()
        gamma_norm = route_out.get("hyper_gamma_norm_per_head", None)
        if torch.is_tensor(gamma_norm) and gamma_norm.shape[0] > idx:
            logs[f"fpem/hyper_gamma_norm_head_{idx}"] = gamma_norm[idx].detach()
        beta_norm = route_out.get("hyper_beta_norm_per_head", None)
        if torch.is_tensor(beta_norm) and beta_norm.shape[0] > idx:
            logs[f"fpem/hyper_beta_norm_head_{idx}"] = beta_norm[idx].detach()
    return total, logs, q_oracle.detach()


def future_mi_loss(e_useful, e_future, mu_head, logvar_head, args, training, epoch):
    zero = e_useful.new_zeros(())
    logs = {
        "fpem/future_mi_loss": zero,
        "fpem/future_mi_valid": zero,
        "fpem/future_mi_logvar_mean": zero,
        "fpem/future_mi_target_mode_env_encoder": zero,
    }
    if not training or e_future is None or mu_head is None or logvar_head is None:
        return zero, logs
    warmup = int(getattr(args, "fpem_future_mi_warmup_epochs", 0))
    if epoch is not None and int(epoch) < warmup:
        return zero, logs
    if str(getattr(args, "fpem_future_mi_detach_target", True)).lower() in {"1", "true", "yes", "y", "on"}:
        e_future = e_future.detach()
    with autocast_disabled():
        e_float = e_useful.float()
        target_float = e_future.float()
        pred_mu = mu_head(e_float)
        pred_logvar = logvar_head(e_float).clamp(-8.0, 8.0)
        var = pred_logvar.exp().clamp_min(1e-6)
        loss = (0.5 * ((target_float - pred_mu).pow(2) / var + pred_logvar)).mean()
    logs.update({
        "fpem/future_mi_loss": loss.detach(),
        "fpem/future_mi_valid": e_useful.new_tensor(1.0),
        "fpem/future_mi_logvar_mean": pred_logvar.detach().mean(),
        "fpem/future_mi_target_mode_env_encoder": e_useful.new_tensor(
            1.0 if str(getattr(args, "fpem_future_mi_target_mode", "env_encoder")).lower() == "env_encoder" else 0.0
        ),
    })
    return loss.to(dtype=e_useful.dtype), logs


def gain_weighted_swap_loss(pred_full, pred_inv, pred_swap, target, scaler, args, valid_sample=None):
    zero = pred_full.new_zeros(())
    logs = {
        "fpem/swap_loss": zero,
        "fpem/swap_diff_loss": zero,
        "fpem/swap_same_loss": zero,
        "fpem/swap_gain_mean": zero,
        "fpem/swap_s_gain_mean": zero,
    }
    full_err, mask = flow_error_view(pred_full.detach(), target, scaler, getattr(args, "yita", 0.5))
    inv_err, _ = flow_error_view(pred_inv.detach(), target, scaler, getattr(args, "yita", 0.5))
    swap_err, _ = flow_error_view(pred_swap, target, scaler, getattr(args, "yita", 0.5))
    swap_raw = scaler.inverse_transform(pred_swap)
    full_raw = scaler.inverse_transform(pred_full.detach())
    same_err = float(getattr(args, "yita", 0.5)) * (swap_raw[..., 0] - full_raw[..., 0]).abs()
    same_err = same_err + (1.0 - float(getattr(args, "yita", 0.5))) * (swap_raw[..., 1] - full_raw[..., 1]).abs()
    if valid_sample is not None:
        valid = valid_sample
        while valid.dim() < mask.dim():
            valid = valid.unsqueeze(-1)
        mask = mask & valid.expand_as(mask)
    gain = inv_err.detach() - full_err.detach()
    eta = float(getattr(args, "fpem_swap_gain_eta", 0.0))
    tau = max(float(getattr(args, "fpem_swap_gain_tau", 0.05)), 1e-6)
    s_gain = torch.sigmoid((gain - eta) / tau)
    if not bool(mask.any()):
        return zero, logs
    margin = float(getattr(args, "fpem_swap_margin", 0.01))
    swap_diff_loss = masked_mean(s_gain * F.relu(margin + full_err.detach() - swap_err), mask)
    swap_same_loss = masked_mean((1.0 - s_gain) * same_err, mask)
    swap_loss = float(getattr(args, "fpem_lambda_swap", 0.01)) * (
        float(getattr(args, "fpem_lambda_swap_diff", 1.0)) * swap_diff_loss
        + float(getattr(args, "fpem_lambda_swap_same", 0.05)) * swap_same_loss
    )
    logs.update({
        "fpem/swap_loss": swap_loss.detach(),
        "fpem/swap_diff_loss": swap_diff_loss.detach(),
        "fpem/swap_same_loss": swap_same_loss.detach(),
        "fpem/swap_gain_mean": masked_mean(gain.detach(), mask).detach(),
        "fpem/swap_s_gain_mean": masked_mean(s_gain.detach(), mask).detach(),
    })
    return swap_loss, logs
