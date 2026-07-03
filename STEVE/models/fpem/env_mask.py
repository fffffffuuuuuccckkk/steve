import torch
import torch.nn as nn


class EnvMask(nn.Module):
    def __init__(self, hidden_dim, mask_hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mask_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mask_hidden_dim, hidden_dim),
        )

    @staticmethod
    def zero_logs(ref):
        zero = ref.new_zeros(())
        one = ref.new_ones(())
        return {
            "fpem/mask_loss": zero,
            "fpem/mask_sparse_loss": zero,
            "fpem/mask_entropy_loss": zero,
            "fpem/mask_mean": one,
            "fpem/mask_entropy": zero,
            "fpem/mask_active_ratio": one,
        }

    def forward(
        self,
        env_repr,
        use_mask=True,
        lambda_sparse=0.0,
        lambda_entropy=0.0,
        temperature=1.0,
    ):
        logs = self.zero_logs(env_repr)
        if not use_mask:
            mask = torch.ones_like(env_repr)
            return env_repr, env_repr.new_zeros(env_repr.shape), mask, env_repr.new_zeros(()), logs

        temp = max(float(temperature), 1e-6)
        mask = torch.sigmoid(self.net(env_repr) / temp)
        e_useful = mask * env_repr
        e_discard = (1.0 - mask) * env_repr

        # Keep the entropy regularizer in FP32. In fp16 AMP, saturated masks can
        # otherwise create 0 * log(0) NaNs.
        mask_prob = mask.float().clamp(1e-6, 1.0 - 1e-6)
        entropy = -(mask_prob * mask_prob.log() + (1.0 - mask_prob) * (1.0 - mask_prob).log()).mean()
        sparse = mask.float().mean()
        mask_loss = float(lambda_sparse) * sparse + float(lambda_entropy) * entropy

        logs.update({
            "fpem/mask_loss": mask_loss.detach(),
            "fpem/mask_sparse_loss": sparse.detach(),
            "fpem/mask_entropy_loss": entropy.detach(),
            "fpem/mask_mean": mask.detach().mean(),
            "fpem/mask_entropy": entropy.detach(),
            "fpem/mask_active_ratio": (mask.detach() > 0.5).to(dtype=env_repr.dtype).mean(),
        })
        return e_useful, e_discard, mask, mask_loss.to(dtype=env_repr.dtype), logs
