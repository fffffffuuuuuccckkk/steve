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
    if pred_raw.shape[-1] == 1 or target_raw.shape[-1] == 1:
        return masked_channel_mae(pred_raw[..., 0], target_raw[..., 0], mask_value)
    loss = float(yita) * masked_channel_mae(pred_raw[..., 0], target_raw[..., 0], mask_value)
    loss = loss + (1.0 - float(yita)) * masked_channel_mae(pred_raw[..., 1], target_raw[..., 1], mask_value)
    return loss


def target_mask(target_raw):
    return target_raw[..., 0] > 5.0


def flow_error_view(pred, target, scaler, yita=0.5):
    pred_raw = scaler.inverse_transform(pred)
    target_raw = scaler.inverse_transform(target)
    if pred_raw.shape[-1] == 1 or target_raw.shape[-1] == 1:
        err = (pred_raw[..., 0] - target_raw[..., 0]).abs()
        return err, target_mask(target_raw)
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
    if pred_raw.shape[-1] == 1 or target_raw.shape[-1] == 1:
        err = (pred_raw[..., 0] - target_raw[..., 0]).abs()
    else:
        err = float(yita) * (pred_raw[..., 0] - target_raw[..., 0]).abs()
        err = err + (1.0 - float(yita)) * (pred_raw[..., 1] - target_raw[..., 1]).abs()
    mask = target_mask(target_raw.squeeze(1)).unsqueeze(1).to(dtype=err.dtype)
    return (err * mask).sum(dim=(2, 3)) / mask.sum(dim=(2, 3)).clamp_min(1.0)


def load_level_expert_gate_losses(route_out, target, scaler, args, training, epoch):
    """Balanced load-expert loss plus detached soft supervision for the binary gate."""
    y_heads = route_out["y_env_heads"]
    y_inv = route_out["y_inv"]
    load_level = route_out["load_level"].detach().long()
    gate = route_out["env_use_gate"].reshape(-1)
    gate_logits = route_out["env_use_gate_logits"].reshape(-1)
    num_levels = int(y_heads.shape[1])
    zero = target.new_zeros(())

    loss_head = head_prediction_losses(
        y_heads, target, scaler, getattr(args, "yita", 0.5)
    )
    inv_sample_loss = head_prediction_losses(
        y_inv.unsqueeze(1), target, scaler, getattr(args, "yita", 0.5)
    )[:, 0]
    selected_sample_loss = loss_head.gather(1, load_level[:, None]).squeeze(1)

    level_means = []
    logs = {
        "fpem/load_expert_loss": zero,
        "fpem/environment_gate_loss": zero,
        "fpem/environment_gate_target_mean": zero,
        "fpem/environment_gate_value_mean": gate.detach().mean(),
        "fpem/environment_gate_value_min": gate.detach().min(),
        "fpem/environment_gate_value_max": gate.detach().max(),
        "fpem/environment_gate_positive_weight": zero,
        "fpem/environment_gate_negative_weight": zero,
        "fpem/environment_gate_supervision_enabled": zero,
        "fpem/environment_gate_warmup_active": zero,
        "fpem/load_level_assignment_detached": target.new_tensor(float(not load_level.requires_grad)),
    }
    for idx in range(num_levels):
        mask = load_level == idx
        count = mask.sum()
        logs[f"fpem/load_level_count_{idx}"] = count.detach().to(dtype=target.dtype)
        logs[f"fpem/load_level_ratio_{idx}"] = (
            count.detach().to(dtype=target.dtype) / max(float(load_level.numel()), 1.0)
        )
        logs[f"fpem/expert_{idx}_global_loss"] = loss_head[:, idx].detach().mean()
        if bool(mask.any()):
            selected_mean = loss_head[mask, idx].mean()
            level_means.append(selected_mean)
            logs[f"fpem/expert_{idx}_selected_loss"] = selected_mean.detach()
        else:
            logs[f"fpem/expert_{idx}_selected_loss"] = zero

    expert_loss = torch.stack(level_means).mean() if level_means else zero
    logs["fpem/load_expert_loss"] = expert_loss.detach()

    warmup_epochs = max(int(getattr(args, "fpem_environment_gate_warmup_epochs", 5)), 0)
    warmup_active = epoch is not None and int(epoch) <= warmup_epochs
    gate_enabled = bool(getattr(args, "fpem_use_environment_gate", True))
    gate_loss = zero
    gain = (inv_sample_loss.detach() - selected_sample_loss.detach())
    relative_margin = max(float(getattr(args, "fpem_environment_gate_margin_relative", 0.01)), 0.0)
    margin = relative_margin * inv_sample_loss.detach().clamp_min(0.0)
    temperature = max(float(getattr(args, "fpem_environment_gate_temperature", 0.05)), 1e-6)
    gate_target = torch.sigmoid((gain - margin) / temperature).detach()
    logs.update({
        "fpem/environment_gate_gain_mean": gain.mean(),
        "fpem/environment_gate_margin_mean": margin.mean(),
        "fpem/environment_gate_target_mean": gate_target.mean(),
        "fpem/environment_gate_warmup_active": target.new_tensor(float(warmup_active)),
    })

    if training and gate_enabled and not warmup_active:
        target_mean = gate_target.mean()
        strongly_imbalanced = bool((target_mean < 0.2) or (target_mean > 0.8))
        if strongly_imbalanced:
            positive_mass = gate_target.sum().clamp_min(1e-6)
            negative_mass = (1.0 - gate_target).sum().clamp_min(1e-6)
            total_mass = positive_mass + negative_mass
            positive_weight = (0.5 * total_mass / positive_mass).detach().clamp(max=20.0)
            negative_weight = (0.5 * total_mass / negative_mass).detach().clamp(max=20.0)
            positive_term = -gate_target * torch.nn.functional.logsigmoid(gate_logits)
            negative_term = -(1.0 - gate_target) * torch.nn.functional.logsigmoid(-gate_logits)
            gate_loss = (positive_weight * positive_term + negative_weight * negative_term).mean()
        else:
            positive_weight = gate.new_tensor(1.0)
            negative_weight = gate.new_tensor(1.0)
            gate_loss = F.binary_cross_entropy_with_logits(gate_logits.float(), gate_target.float())
        logs.update({
            "fpem/environment_gate_loss": gate_loss.detach(),
            "fpem/environment_gate_positive_weight": positive_weight,
            "fpem/environment_gate_negative_weight": negative_weight,
            "fpem/environment_gate_supervision_enabled": target.new_tensor(1.0),
        })

    return expert_loss, gate_loss, logs, {
        "loss_head": loss_head.detach(),
        "inv_sample_loss": inv_sample_loss.detach(),
        "selected_sample_loss": selected_sample_loss.detach(),
        "gain": gain.detach(),
        "gate_target": gate_target.detach(),
    }


def load_level_expert_hard_router_losses(
    route_out, target, scaler, args, training, epoch
):
    """Balanced fixed-expert loss and detached hard binary-router supervision."""
    y_heads = route_out["y_env_heads"]
    y_inv = route_out["y_inv"]
    load_level = route_out["load_level"].detach().long()
    router_logits = route_out["hard_router_logits"]
    num_levels = int(y_heads.shape[1])
    zero = target.new_zeros(())

    loss_head = head_prediction_losses(
        y_heads, target, scaler, getattr(args, "yita", 0.5)
    )
    inv_sample_loss = head_prediction_losses(
        y_inv.unsqueeze(1), target, scaler, getattr(args, "yita", 0.5)
    )[:, 0]
    selected_sample_loss = loss_head.gather(
        1, load_level[:, None]
    ).squeeze(1)

    level_means = []
    logs = {
        "fpem/load_expert_loss": zero,
        "fpem/hard_router_loss": zero,
        "fpem/hard_router_target_env_ratio": zero,
        "fpem/hard_router_predicted_env_ratio": (
            route_out["hard_router_predicted_route_id"].detach().float().mean()
        ),
        "fpem/hard_router_effective_env_ratio": (
            route_out["hard_route_id"].detach().float().mean()
        ),
        "fpem/hard_router_supervision_enabled": zero,
        "fpem/hard_router_warmup_active": route_out[
            "hard_router_warmup_active"
        ].detach(),
        "fpem/load_level_assignment_detached": target.new_tensor(
            float(not load_level.requires_grad)
        ),
    }
    for idx in range(num_levels):
        mask = load_level == idx
        count = mask.sum()
        logs[f"fpem/load_level_count_{idx}"] = count.detach().to(
            dtype=target.dtype
        )
        logs[f"fpem/load_level_ratio_{idx}"] = count.detach().to(
            dtype=target.dtype
        ) / max(float(load_level.numel()), 1.0)
        logs[f"fpem/expert_{idx}_global_loss"] = loss_head[
            :, idx
        ].detach().mean()
        if bool(mask.any()):
            selected_mean = loss_head[mask, idx].mean()
            level_means.append(selected_mean)
            logs[f"fpem/expert_{idx}_selected_loss"] = selected_mean.detach()
        else:
            logs[f"fpem/expert_{idx}_selected_loss"] = zero

    expert_loss = torch.stack(level_means).mean() if level_means else zero
    logs["fpem/load_expert_loss"] = expert_loss.detach()

    relative_margin = max(
        float(getattr(args, "fpem_hard_router_relative_margin", 0.01)), 0.0
    )
    margin = relative_margin * inv_sample_loss.detach().clamp_min(0.0)
    router_target = (
        selected_sample_loss.detach() + margin < inv_sample_loss.detach()
    ).long().detach()
    logs.update(
        {
            "fpem/hard_router_target_env_ratio": router_target.float().mean(),
            "fpem/hard_router_margin_mean": margin.mean(),
            "fpem/hard_router_gain_mean": (
                inv_sample_loss.detach() - selected_sample_loss.detach()
            ).mean(),
        }
    )

    warmup_epochs = max(
        int(getattr(args, "fpem_hard_router_warmup_epochs", 5)), 0
    )
    warmup_active = epoch is not None and int(epoch) <= warmup_epochs
    router_enabled = bool(
        getattr(args, "fpem_use_hard_environment_router", False)
    )
    router_loss = zero
    if training and router_enabled and not warmup_active:
        counts = torch.bincount(router_target, minlength=2).float()
        if bool((counts > 0).all()):
            weights = (
                router_target.new_tensor(float(router_target.numel()))
                .float()
                / (2.0 * counts)
            ).clamp(max=20.0)
        else:
            weights = router_logits.new_ones(2)
        router_loss = F.cross_entropy(
            router_logits.float(), router_target, weight=weights
        )
        logs.update(
            {
                "fpem/hard_router_loss": router_loss.detach(),
                "fpem/hard_router_class_weight_invariant": weights[0].detach(),
                "fpem/hard_router_class_weight_environment": weights[1].detach(),
                "fpem/hard_router_supervision_enabled": target.new_tensor(1.0),
            }
        )
    else:
        logs["fpem/hard_router_class_weight_invariant"] = zero
        logs["fpem/hard_router_class_weight_environment"] = zero

    return expert_loss, router_loss, logs, {
        "loss_head": loss_head.detach(),
        "inv_sample_loss": inv_sample_loss.detach(),
        "selected_sample_loss": selected_sample_loss.detach(),
        "gain": (
            inv_sample_loss.detach() - selected_sample_loss.detach()
        ),
        "router_target": router_target.detach(),
        "margin": margin.detach(),
    }


def load_level_expert_conservative_override_losses(
    route_out, target, scaler, args, training, epoch
):
    """Regret-aware conservative invariant override for observable experts.

    The environment expert selected by the leakage-safe observable load prior is
    the default prediction path.  The binary router only learns when switching
    to the invariant prediction would save enough masked MAE:

        delta = L_env - L_inv
        target_inv = 1[delta > margin]

    L_env/L_inv/delta are detached for the router objective, so this loss cannot
    update either counterfactual prediction branch through the supervision
    target.  The per-sample losses reuse head_prediction_losses(), which applies
    the same inverse-scaler and target mask as the formal forecasting metric.
    """
    y_heads = route_out["y_env_heads"]
    y_inv = route_out["y_inv"]
    load_level = route_out["load_level"].detach().long()
    router_logits = route_out["hard_router_logits"]
    num_levels = int(y_heads.shape[1])
    zero = target.new_zeros(())
    eps = 1e-6

    loss_head = head_prediction_losses(
        y_heads, target, scaler, getattr(args, "yita", 0.5)
    )
    inv_sample_loss = head_prediction_losses(
        y_inv.unsqueeze(1), target, scaler, getattr(args, "yita", 0.5)
    )[:, 0]
    selected_sample_loss = loss_head.gather(
        1, load_level[:, None]
    ).squeeze(1)

    level_means = []
    logs = {
        "fpem/load_expert_loss": zero,
        "fpem/conservative_override_router_loss": zero,
        "fpem/hard_router_loss": zero,
        "fpem/load_level_assignment_detached": target.new_tensor(
            float(not load_level.requires_grad)
        ),
    }
    for idx in range(num_levels):
        mask = load_level == idx
        count = mask.sum()
        logs[f"fpem/load_level_count_{idx}"] = count.detach().to(
            dtype=target.dtype
        )
        logs[f"fpem/load_level_ratio_{idx}"] = count.detach().to(
            dtype=target.dtype
        ) / max(float(load_level.numel()), 1.0)
        logs[f"fpem/expert_{idx}_global_loss"] = loss_head[
            :, idx
        ].detach().mean()
        if bool(mask.any()):
            selected_mean = loss_head[mask, idx].mean()
            level_means.append(selected_mean)
            logs[f"fpem/expert_{idx}_selected_loss"] = selected_mean.detach()
        else:
            logs[f"fpem/expert_{idx}_selected_loss"] = zero

    expert_loss = torch.stack(level_means).mean() if level_means else zero
    logs["fpem/load_expert_loss"] = expert_loss.detach()

    margin = max(float(getattr(args, "fpem_override_margin", 0.0)), 0.0)
    threshold = float(getattr(args, "fpem_override_threshold", 0.5))
    weight_min = max(float(getattr(args, "fpem_override_weight_min", 0.0)), 0.0)
    weight_max = max(float(getattr(args, "fpem_override_weight_max", 20.0)), weight_min + eps)
    harmful_switch_weight = max(
        float(getattr(args, "fpem_harmful_switch_weight", 1.0)), 0.0
    )

    # Positive delta means invariant is better.  These quantities must be
    # detached so the router target cannot train y_env/y_inv.
    env_loss_detached = selected_sample_loss.detach()
    inv_loss_detached = inv_sample_loss.detach()
    delta = (env_loss_detached - inv_loss_detached).detach()
    target_inv = (delta > margin).to(dtype=target.dtype).detach()

    if "conservative_inv_logit" in route_out:
        inv_logit = route_out["conservative_inv_logit"].reshape(-1)
    else:
        inv_logit = (router_logits[:, 0] - router_logits[:, 1]).reshape(-1)
    p_inv = torch.sigmoid(inv_logit.detach())
    use_inv = (p_inv > threshold)
    target_bool = target_inv > 0.5
    env_target_bool = ~target_bool

    selective_sample_loss = torch.where(
        use_inv, inv_loss_detached, env_loss_detached
    )
    oracle_sample_loss = torch.minimum(env_loss_detached, inv_loss_detached)
    all_environment_mae = env_loss_detached.mean()
    all_invariant_mae = inv_loss_detached.mean()
    selective_mae = selective_sample_loss.mean()
    oracle_mae = oracle_sample_loss.mean()
    router_accuracy = (use_inv == target_bool).to(dtype=target.dtype).mean()
    recalls = []
    if bool(target_bool.any()):
        recalls.append(use_inv[target_bool].to(dtype=target.dtype).mean())
    if bool(env_target_bool.any()):
        recalls.append((~use_inv[env_target_bool]).to(dtype=target.dtype).mean())
    balanced_accuracy = torch.stack(recalls).mean() if recalls else zero
    switch_count = use_inv.sum().to(dtype=target.dtype)
    correct_switch = use_inv & target_bool
    harmful_switch = use_inv & env_target_bool
    correct_switch_count = correct_switch.sum().to(dtype=target.dtype)
    harmful_switch_count = harmful_switch.sum().to(dtype=target.dtype)
    switch_precision = correct_switch_count / switch_count.clamp_min(1.0)
    saved_loss = (delta.clamp_min(0.0) * correct_switch.to(dtype=delta.dtype)).sum()
    saved_loss = saved_loss / max(float(delta.numel()), 1.0)
    added_loss = ((-delta).clamp_min(0.0) * harmful_switch.to(dtype=delta.dtype)).sum()
    added_loss = added_loss / max(float(delta.numel()), 1.0)
    net_gain = all_environment_mae - selective_mae
    oracle_gap = (all_environment_mae - oracle_mae).clamp_min(eps)
    oracle_gap_closed = net_gain / oracle_gap

    logs.update(
        {
            "fpem/conservative_all_environment_mae": all_environment_mae,
            "fpem/conservative_all_invariant_mae": all_invariant_mae,
            "fpem/conservative_selective_hard_routing_mae": selective_mae,
            "fpem/conservative_oracle_mae": oracle_mae,
            "fpem/conservative_router_accuracy": router_accuracy,
            "fpem/conservative_router_balanced_accuracy": balanced_accuracy,
            "fpem/conservative_invariant_switch_coverage": use_inv.to(dtype=target.dtype).mean(),
            "fpem/conservative_invariant_switch_precision": switch_precision,
            "fpem/conservative_correct_beneficial_switches": correct_switch_count,
            "fpem/conservative_harmful_invariant_switches": harmful_switch_count,
            "fpem/conservative_saved_loss_from_correct_switches": saved_loss,
            "fpem/conservative_added_loss_from_harmful_switches": added_loss,
            "fpem/conservative_net_routing_gain": net_gain,
            "fpem/conservative_oracle_gap_closed": oracle_gap_closed,
            "fpem/conservative_selected_threshold": target.new_tensor(threshold),
            "fpem/conservative_selected_margin": target.new_tensor(margin),
            "fpem/conservative_delta_mean": delta.mean(),
            "fpem/conservative_delta_positive_ratio": (delta > 0.0).to(dtype=target.dtype).mean(),
            "fpem/conservative_target_inv_ratio": target_inv.mean(),
            "fpem/conservative_p_inv_mean": p_inv.mean(),
            "fpem/conservative_p_inv_min": p_inv.min(),
            "fpem/conservative_p_inv_max": p_inv.max(),
            "fpem/conservative_prediction_is_hard_route": target.new_tensor(1.0),
            "fpem/conservative_prediction_is_soft_fusion": target.new_zeros(()),
            "fpem/conservative_router_target_detached": target.new_tensor(
                float(not target_inv.requires_grad)
            ),
            "fpem/conservative_delta_detached": target.new_tensor(
                float(not delta.requires_grad)
            ),
            "fpem/conservative_harmful_switch_weight": target.new_tensor(
                harmful_switch_weight
            ),
            "fpem/hard_router_predicted_env_ratio": (~use_inv).to(dtype=target.dtype).mean(),
            "fpem/hard_router_effective_env_ratio": route_out["hard_route_id"].detach().float().mean(),
            "fpem/hard_router_target_env_ratio": (1.0 - target_inv).mean(),
            "fpem/hard_router_gain_mean": delta.mean(),
            "fpem/hard_router_margin_mean": target.new_tensor(margin),
        }
    )

    warmup_flag = route_out.get("hard_router_warmup_active", zero)
    warmup_active = bool(
        torch.is_tensor(warmup_flag) and float(warmup_flag.detach().cpu().item()) > 0.5
    )
    router_enabled = bool(getattr(args, "fpem_conservative_inv_override", False))
    router_loss = zero
    if training and router_enabled and not warmup_active:
        sample_weight = delta.abs().clamp(weight_min, weight_max).detach()
        sample_weight = torch.where(
            target_bool,
            sample_weight,
            sample_weight * harmful_switch_weight,
        )
        sample_weight = sample_weight / sample_weight.mean().clamp_min(eps)
        router_loss = F.binary_cross_entropy_with_logits(
            inv_logit.float(),
            target_inv.float(),
            weight=sample_weight.float(),
        )
        logs.update(
            {
                "fpem/conservative_override_router_loss": router_loss.detach(),
                "fpem/hard_router_loss": router_loss.detach(),
                "fpem/conservative_router_supervision_enabled": target.new_tensor(1.0),
                "fpem/conservative_router_weight_mean": sample_weight.detach().mean(),
                "fpem/conservative_router_weight_min": sample_weight.detach().min(),
                "fpem/conservative_router_weight_max": sample_weight.detach().max(),
            }
        )
    else:
        logs.update(
            {
                "fpem/conservative_router_supervision_enabled": zero,
                "fpem/conservative_router_weight_mean": zero,
                "fpem/conservative_router_weight_min": zero,
                "fpem/conservative_router_weight_max": zero,
            }
        )

    return expert_loss, router_loss, logs, {
        "loss_head": loss_head.detach(),
        "inv_sample_loss": inv_loss_detached,
        "selected_sample_loss": env_loss_detached,
        "delta": delta,
        "target_inv": target_inv.detach(),
        "use_inv": use_inv.detach(),
        "router_target": target_inv.long().detach(),
        "margin": target.new_full((), margin),
        "threshold": target.new_full((), threshold),
    }


def _balanced_regret_weights(delta, weight_min, weight_max, eps=1e-6):
    """Normalize regret weights separately for env-better and inv-better groups."""
    raw = delta.abs().clamp(weight_min, weight_max).detach()
    inv_better = delta > 0.0
    env_better = ~inv_better
    weights = torch.zeros_like(raw)
    groups = [env_better, inv_better]
    present = [mask for mask in groups if bool(mask.any())]
    if not present:
        return torch.ones_like(raw)
    target_mass = float(raw.numel()) / float(len(present))
    for mask in present:
        group_raw = raw[mask]
        weights[mask] = group_raw * (target_mass / group_raw.sum().clamp_min(eps))
    return weights


def load_level_expert_counterfactual_risk_router_losses(
    route_out, target, scaler, args, training, epoch
):
    """Train a threshold-free risk router for env-vs-invariant hard selection.

    The router predicts two per-sample counterfactual risks in the order
    [L_env, L_inv].  The supervision targets are detached masked MAE values,
    computed with the same inverse-scaler/mask convention as the formal metric.
    Prediction branches are not updated by these router targets.
    """
    y_heads = route_out["y_env_heads"]
    y_inv = route_out["y_inv"]
    load_level = route_out["load_level"].detach().long()
    predicted_risks = route_out["counterfactual_predicted_risks"]
    num_levels = int(y_heads.shape[1])
    zero = target.new_zeros(())
    eps = 1e-6

    loss_head = head_prediction_losses(
        y_heads, target, scaler, getattr(args, "yita", 0.5)
    )
    inv_sample_loss = head_prediction_losses(
        y_inv.unsqueeze(1), target, scaler, getattr(args, "yita", 0.5)
    )[:, 0]
    env_sample_loss = loss_head.gather(1, load_level[:, None]).squeeze(1)

    level_means = []
    logs = {
        "fpem/load_expert_loss": zero,
        "fpem/counterfactual_risk_router_loss": zero,
        "fpem/counterfactual_risk_regression_loss": zero,
        "fpem/counterfactual_risk_ranking_loss": zero,
        "fpem/load_level_assignment_detached": target.new_tensor(
            float(not load_level.requires_grad)
        ),
    }
    for idx in range(num_levels):
        mask = load_level == idx
        count = mask.sum()
        logs[f"fpem/load_level_count_{idx}"] = count.detach().to(dtype=target.dtype)
        logs[f"fpem/load_level_ratio_{idx}"] = (
            count.detach().to(dtype=target.dtype) / max(float(load_level.numel()), 1.0)
        )
        logs[f"fpem/expert_{idx}_global_loss"] = loss_head[:, idx].detach().mean()
        if bool(mask.any()):
            selected_mean = loss_head[mask, idx].mean()
            level_means.append(selected_mean)
            logs[f"fpem/expert_{idx}_selected_loss"] = selected_mean.detach()
        else:
            logs[f"fpem/expert_{idx}_selected_loss"] = zero
    expert_loss = torch.stack(level_means).mean() if level_means else zero
    logs["fpem/load_expert_loss"] = expert_loss.detach()

    env_loss_detached = env_sample_loss.detach()
    inv_loss_detached = inv_sample_loss.detach()
    delta = (env_loss_detached - inv_loss_detached).detach()
    target_inv = (delta > 0.0).detach()
    risk_target = torch.stack([env_loss_detached, inv_loss_detached], dim=-1)
    risk_pred = predicted_risks.float()
    risk_route_idx = risk_pred.detach().argmin(dim=-1)  # 0=env, 1=inv
    use_inv = risk_route_idx == 1

    selective_sample_loss = torch.where(use_inv, inv_loss_detached, env_loss_detached)
    oracle_sample_loss = torch.minimum(env_loss_detached, inv_loss_detached)
    all_environment_mae = env_loss_detached.mean()
    all_invariant_mae = inv_loss_detached.mean()
    routed_mae = selective_sample_loss.mean()
    oracle_mae = oracle_sample_loss.mean()
    router_accuracy = (use_inv == target_inv).to(dtype=target.dtype).mean()
    env_target = ~target_inv
    recalls = []
    if bool(target_inv.any()):
        recalls.append(use_inv[target_inv].to(dtype=target.dtype).mean())
    if bool(env_target.any()):
        recalls.append((~use_inv[env_target]).to(dtype=target.dtype).mean())
    balanced_accuracy = torch.stack(recalls).mean() if recalls else zero
    switch_count = use_inv.sum().to(dtype=target.dtype)
    correct_switch = use_inv & target_inv
    harmful_switch = use_inv & env_target
    correct_switch_count = correct_switch.sum().to(dtype=target.dtype)
    harmful_switch_count = harmful_switch.sum().to(dtype=target.dtype)
    saved_loss = (
        delta.clamp_min(0.0) * correct_switch.to(dtype=delta.dtype)
    ).sum() / max(float(delta.numel()), 1.0)
    added_loss = (
        (-delta).clamp_min(0.0) * harmful_switch.to(dtype=delta.dtype)
    ).sum() / max(float(delta.numel()), 1.0)
    net_gain = all_environment_mae - routed_mae
    regret = routed_mae - oracle_mae
    oracle_gap = (all_environment_mae - oracle_mae).clamp_min(eps)
    oracle_gap_closed = net_gain / oracle_gap

    weight_min = max(float(getattr(args, "fpem_counterfactual_risk_weight_min", 0.0)), 0.0)
    weight_max = max(
        float(getattr(args, "fpem_counterfactual_risk_weight_max", 20.0)),
        weight_min + eps,
    )
    weights = _balanced_regret_weights(delta, weight_min, weight_max, eps=eps)
    regression_per_sample = F.smooth_l1_loss(
        risk_pred,
        risk_target.float(),
        reduction="none",
    ).mean(dim=-1)
    regression_loss = (
        regression_per_sample * weights
    ).sum() / weights.sum().clamp_min(eps)
    pred_delta = risk_pred[:, 0] - risk_pred[:, 1]
    sign = torch.where(delta >= 0.0, torch.ones_like(delta), -torch.ones_like(delta))
    ranking_temperature = max(
        float(getattr(args, "fpem_counterfactual_risk_ranking_temperature", 1.0)),
        eps,
    )
    ranking_per_sample = F.softplus(-(sign * pred_delta) / ranking_temperature)
    ranking_loss = (
        ranking_per_sample * weights
    ).sum() / weights.sum().clamp_min(eps)
    regression_weight = float(
        getattr(args, "fpem_counterfactual_risk_regression_weight", 1.0)
    )
    ranking_weight = float(
        getattr(args, "fpem_counterfactual_risk_ranking_weight", 0.5)
    )
    stage2_start = int(getattr(args, "fpem_counterfactual_risk_stage2_start_epoch", 20))
    stage2_active = bool(training and epoch is not None and int(epoch) >= stage2_start)
    router_loss = (
        regression_weight * regression_loss + ranking_weight * ranking_loss
        if stage2_active
        else zero
    )

    env_weight_sum = weights[env_target].sum() if bool(env_target.any()) else zero
    inv_weight_sum = weights[target_inv].sum() if bool(target_inv.any()) else zero
    logs.update(
        {
            "fpem/counterfactual_risk_router_loss": router_loss.detach(),
            "fpem/counterfactual_risk_regression_loss": regression_loss.detach(),
            "fpem/counterfactual_risk_ranking_loss": ranking_loss.detach(),
            "fpem/counterfactual_risk_stage2_active": target.new_tensor(float(stage2_active)),
            "fpem/counterfactual_risk_stage2_start_epoch": target.new_tensor(float(stage2_start)),
            "fpem/counterfactual_risk_all_environment_mae": all_environment_mae,
            "fpem/counterfactual_risk_all_invariant_mae": all_invariant_mae,
            "fpem/counterfactual_risk_routed_mae": routed_mae,
            "fpem/counterfactual_risk_oracle_mae": oracle_mae,
            "fpem/counterfactual_risk_router_accuracy": router_accuracy,
            "fpem/counterfactual_risk_router_balanced_accuracy": balanced_accuracy,
            "fpem/counterfactual_risk_inv_route_ratio": use_inv.to(dtype=target.dtype).mean(),
            "fpem/counterfactual_risk_env_route_ratio": (~use_inv).to(dtype=target.dtype).mean(),
            "fpem/counterfactual_risk_inv_switch_precision": correct_switch_count / switch_count.clamp_min(1.0),
            "fpem/counterfactual_risk_correct_inv_switches": correct_switch_count,
            "fpem/counterfactual_risk_harmful_inv_switches": harmful_switch_count,
            "fpem/counterfactual_risk_saved_loss": saved_loss,
            "fpem/counterfactual_risk_added_loss": added_loss,
            "fpem/counterfactual_risk_net_gain": net_gain,
            "fpem/counterfactual_risk_regret": regret,
            "fpem/counterfactual_risk_oracle_gap_closed": oracle_gap_closed,
            "fpem/counterfactual_risk_delta_mean": delta.mean(),
            "fpem/counterfactual_risk_target_inv_ratio": target_inv.to(dtype=target.dtype).mean(),
            "fpem/counterfactual_risk_weight_env_sum": env_weight_sum,
            "fpem/counterfactual_risk_weight_inv_sum": inv_weight_sum,
            "fpem/counterfactual_risk_prediction_is_hard_route": target.new_tensor(1.0),
            "fpem/counterfactual_risk_prediction_is_soft_fusion": target.new_zeros(()),
            "fpem/counterfactual_risk_target_detached": target.new_tensor(float(not risk_target.requires_grad)),
            "fpem/counterfactual_risk_uses_threshold": target.new_zeros(()),
            "fpem/counterfactual_risk_all_samples_used_for_router": target.new_tensor(1.0),
            "fpem/hard_router_predicted_env_ratio": (~use_inv).to(dtype=target.dtype).mean(),
            "fpem/hard_router_effective_env_ratio": route_out["hard_route_id"].detach().float().mean(),
            "fpem/hard_router_target_env_ratio": env_target.to(dtype=target.dtype).mean(),
            "fpem/hard_router_gain_mean": delta.mean(),
        }
    )

    return expert_loss, router_loss, logs, {
        "loss_head": loss_head.detach(),
        "inv_sample_loss": inv_loss_detached,
        "selected_sample_loss": env_loss_detached,
        "delta": delta,
        "target_inv": target_inv.detach().to(dtype=target.dtype),
        "use_inv": use_inv.detach(),
        "router_target": target_inv.long().detach(),
        "risk_target": risk_target.detach(),
        "predicted_risks": risk_pred.detach(),
    }


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
