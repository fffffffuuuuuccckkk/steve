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


def route_losses(route_out, target, scaler, args):
    q = route_out["env_route_q"]
    logits = route_out["env_route_logits"]
    mode = str(route_out.get("route_head_mode", getattr(args, "fpem_env_route_head_mode", "concat_input"))).lower()
    is_hyper = mode == "hyper_inv_film"
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
        "fpem/env_route_entropy": zero,
        "fpem/env_route_q_max_mean": zero,
        "fpem/env_route_train_mode_gradient_compat": zero,
        "fpem/env_route_gradient_compat_aux": zero,
        "fpem/hyper_alpha_mean": zero,
        "fpem/hyper_delta_norm": zero,
        "fpem/fallback_q_mean": zero,
        "fpem/fallback_q_max": zero,
        "fpem/env_q_sum_mean": zero,
        "fpem/oracle_fallback_rate": zero,
        "fpem/route_count_fallback": zero,
        "fpem/env_route_head_mode": zero,
    }
    for idx in range(k_env):
        logs[f"fpem/env_route_count_head_{idx}"] = zero
        logs[f"fpem/env_route_oracle_count_head_{idx}"] = zero
        logs[f"fpem/route_count_env_head_{idx}"] = zero

    train_mode = str(getattr(args, "fpem_env_route_train_mode", "soft_oracle")).lower()
    grad_compat = train_mode == "gradient_compat_route"
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

    if grad_compat:
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

    lambda_global_for_total = 0.0 if is_hyper else float(getattr(args, "fpem_env_route_lambda_global", 0.2))
    total = (
        float(getattr(args, "fpem_env_route_lambda_final", 1.0)) * final_loss
        + lambda_global_for_total * global_loss
        + float(getattr(args, "fpem_env_route_lambda_route_soft", 0.5)) * route_soft_loss
        + float(getattr(args, "fpem_env_route_lambda_expert", 0.2)) * expert_loss
        + float(getattr(args, "fpem_env_route_lambda_router_oracle", 0.5)) * router_oracle_loss
        + float(getattr(args, "fpem_env_route_lambda_balance", 0.01)) * balance_loss
        + float(getattr(args, "fpem_env_route_lambda_diverse", 0.001)) * diverse_loss
        + float(getattr(args, "fpem_env_route_lambda_entropy", 0.0)) * entropy
        + grad_aux_weighted
    )

    hard = q.detach().argmax(dim=-1)
    oracle = q_oracle.detach().argmax(dim=-1)
    counts = torch.bincount(hard, minlength=k).to(dtype=target.dtype, device=target.device)
    oracle_counts = torch.bincount(oracle, minlength=k).to(dtype=target.dtype, device=target.device)
    fallback_offset = 1 if is_hyper and q.shape[1] == k_env + 1 else 0
    logs.update({
        "fpem/env_route_loss": total.detach(),
        "fpem/env_route_L_final": final_loss.detach(),
        "fpem/env_route_L_global": global_loss.detach(),
        "fpem/env_route_L_route_soft": route_soft_loss.detach(),
        "fpem/env_route_L_expert": expert_loss.detach(),
        "fpem/env_route_L_router_oracle": router_oracle_loss.detach(),
        "fpem/env_route_L_balance": balance_loss.detach(),
        "fpem/env_route_L_diverse": diverse_loss.detach(),
        "fpem/env_route_entropy": entropy.detach(),
        "fpem/env_route_q_max_mean": q.detach().max(dim=-1).values.mean(),
        "fpem/env_route_train_mode_gradient_compat": target.new_tensor(float(grad_compat)),
        "fpem/env_route_gradient_compat_aux": grad_aux.detach(),
        "fpem/hyper_alpha_mean": route_out.get("hyper_alpha", zero).detach().mean() if torch.is_tensor(route_out.get("hyper_alpha", None)) else zero,
        "fpem/hyper_delta_norm": route_out.get("hyper_delta_norm", zero).detach() if torch.is_tensor(route_out.get("hyper_delta_norm", None)) else zero,
        "fpem/env_route_head_mode": target.new_tensor(1.0 if is_hyper else 0.0),
    })
    if is_hyper and fallback_offset == 1:
        fallback_q = q[:, 0].detach()
        logs.update({
            "fpem/fallback_q_mean": fallback_q.mean(),
            "fpem/fallback_q_max": fallback_q.max(),
            "fpem/env_q_sum_mean": q[:, 1:].detach().sum(dim=-1).mean(),
            "fpem/oracle_fallback_rate": (oracle == 0).to(dtype=target.dtype).mean(),
            "fpem/route_count_fallback": counts[0].detach(),
        })
    if grad_compat:
        logs["fpem/env_route_gradient_compat_cost_mean"] = gc_cost.mean()
        logs["fpem/env_route_gradient_compat_score_mean"] = gc_score.mean()
    for idx in range(k_env):
        route_idx = idx + fallback_offset
        logs[f"fpem/env_route_count_head_{idx}"] = counts[route_idx].detach()
        logs[f"fpem/env_route_oracle_count_head_{idx}"] = oracle_counts[route_idx].detach()
        logs[f"fpem/route_count_env_head_{idx}"] = counts[route_idx].detach()
    return total, logs, q_oracle.detach()


def future_mi_loss(e_useful, y_inv, target, encoder, mu_head, logvar_head, args, training, epoch):
    zero = e_useful.new_zeros(())
    logs = {
        "fpem/future_mi_loss": zero,
        "fpem/future_mi_valid": zero,
        "fpem/future_mi_logvar_mean": zero,
    }
    if not training or encoder is None:
        return zero, logs
    warmup = int(getattr(args, "fpem_future_mi_warmup_epochs", 0))
    if epoch is not None and int(epoch) < warmup:
        return zero, logs
    residual = (target[:, : y_inv.shape[1]] - y_inv.detach()).squeeze(1)
    target_env = encoder(residual)
    if str(getattr(args, "fpem_future_mi_detach_target", True)).lower() in {"1", "true", "yes", "y", "on"}:
        target_env = target_env.detach()
    with autocast_disabled():
        e_float = e_useful.float()
        target_float = target_env.float()
        pred_mu = mu_head(e_float)
        pred_logvar = logvar_head(e_float).clamp(-8.0, 8.0)
        var = pred_logvar.exp().clamp_min(1e-6)
        loss = (0.5 * ((target_float - pred_mu).pow(2) / var + pred_logvar)).mean()
    logs.update({
        "fpem/future_mi_loss": loss.detach(),
        "fpem/future_mi_valid": e_useful.new_tensor(1.0),
        "fpem/future_mi_logvar_mean": pred_logvar.detach().mean(),
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
