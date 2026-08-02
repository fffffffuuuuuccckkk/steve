import inspect
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

from models.fpem.load_level_gate import EnvironmentUseGate, assign_load_levels
from models.fpem.losses import load_level_expert_gate_losses
from models.our_model import StableST
from run_tds_nyctaxi import fit_load_level_thresholds


class IdentityScaler:
    @staticmethod
    def inverse_transform(value):
        return value


def minimal_model_args():
    return SimpleNamespace(
        num_nodes=3,
        d_input=2,
        d_model=4,
        d_output=2,
        input_length=3,
        dropout=0.0,
        device="cpu",
        fpem_backbone="agcrn",
        fpem_use_pretrained_inv_agcrn=False,
        fpem_use_env_route=True,
        fpem_env_route_k=3,
        fpem_env_route_head_mode="hyper_inv_film_proto_input_add",
        fpem_use_load_level_experts=True,
        fpem_load_level_k=3,
        fpem_load_level_mode="train_quantile",
        fpem_load_level_thresholds=[10.0, 20.0],
        fpem_random_train_load_levels=[],
        fpem_randomize_train_load_level=False,
        fpem_use_environment_gate=True,
        fpem_environment_gate_hidden_dim=8,
        fpem_environment_gate_warmup_epochs=5,
        fpem_environment_gate_margin_relative=0.01,
        fpem_environment_gate_temperature=0.05,
        fpem_lambda_environment_gate=1.0,
        fpem_lambda_load_expert=0.2,
        fpem_env_use_exogenous=True,
        fpem_ignore_future_c=True,
        fpem_use_club_mi=False,
        fpem_use_confounder_extractor=False,
        fpem_use_env_mask=False,
        fpem_use_env_fusion=False,
        fpem_env_route_use_inv_fallback_expert=False,
        fpem_use_future_mi=False,
        fpem_use_swap=False,
        fpem_use_grad_consensus=False,
        fpem_use_env_supervision=False,
        fpem_use_inv_projector=False,
        fpem_use_inv_env_adversarial=False,
        fpem_use_cross_cov_sep=False,
        fpem_env_route_train_mode="soft_oracle",
        fpem_force_uniform_route=False,
        fpem_input_scaler_mean=0.0,
        fpem_input_scaler_std=1.0,
    )


class LoadLevelExpertGateTests(unittest.TestCase):
    def test_train_quantiles_do_not_use_validation_or_test(self):
        train = np.arange(9, dtype=np.float64)
        thresholds, levels, counts = fit_load_level_thresholds(train, 3)
        validation = np.asarray([-1.0e9, 1.0e9])
        test = np.asarray([-2.0e9, 2.0e9])
        thresholds_again, _, _ = fit_load_level_thresholds(train.copy(), 3)
        self.assertTrue(np.array_equal(thresholds, thresholds_again))
        self.assertEqual(counts.tolist(), [3, 3, 3])
        self.assertEqual(np.digitize(validation, thresholds, right=True).tolist(), [0, 2])
        self.assertEqual(np.digitize(test, thresholds, right=True).tolist(), [0, 2])
        self.assertEqual(levels.tolist(), [0, 0, 0, 1, 1, 1, 2, 2, 2])

    def test_assignment_is_detached_and_depends_only_on_observed_score(self):
        observed_history_score = torch.tensor([5.0, 15.0, 25.0], requires_grad=True)
        thresholds = torch.tensor([10.0, 20.0])
        levels = assign_load_levels(observed_history_score, thresholds)
        future_c_a = torch.zeros(3, 1, 3, 2)
        future_c_b = torch.full_like(future_c_a, 5.0)
        self.assertEqual(levels.tolist(), [0, 1, 2])
        self.assertFalse(levels.requires_grad)
        self.assertFalse(torch.equal(future_c_a, future_c_b))
        self.assertEqual(levels.tolist(), assign_load_levels(observed_history_score, thresholds).tolist())

    def test_each_expert_receives_only_its_level_gradient(self):
        torch.manual_seed(7)
        batch_size, num_experts = 6, 3
        y_heads = torch.randn(batch_size, num_experts, 1, 2, 2, requires_grad=True)
        y_inv = torch.zeros(batch_size, 1, 2, 2)
        target = torch.full((batch_size, 1, 2, 2), 10.0)
        load_level = torch.tensor([0, 1, 2, 0, 1, 2])
        gate_logits = torch.zeros(batch_size, requires_grad=True)
        args = SimpleNamespace(
            yita=0.5,
            fpem_environment_gate_warmup_epochs=5,
            fpem_use_environment_gate=True,
            fpem_environment_gate_margin_relative=0.01,
            fpem_environment_gate_temperature=0.05,
        )
        expert_loss, gate_loss, _logs, _diagnostics = load_level_expert_gate_losses(
            {
                "y_env_heads": y_heads,
                "y_inv": y_inv,
                "load_level": load_level,
                "env_use_gate": torch.sigmoid(gate_logits),
                "env_use_gate_logits": gate_logits,
            },
            target,
            IdentityScaler(),
            args,
            training=True,
            epoch=1,
        )
        self.assertEqual(float(gate_loss), 0.0)
        expert_loss.backward()
        for sample in range(batch_size):
            for expert in range(num_experts):
                grad_norm = float(y_heads.grad[sample, expert].abs().sum())
                if expert == int(load_level[sample]):
                    self.assertGreater(grad_norm, 0.0)
                else:
                    self.assertEqual(grad_norm, 0.0)

    def test_gate_forward_has_no_target_argument(self):
        gate = EnvironmentUseGate(4, 3, 8, dropout=0.0).eval()
        signature = inspect.signature(gate.forward)
        self.assertNotIn("target", signature.parameters)
        z_inv = torch.randn(5, 3, 4)
        e_useful = torch.randn(5, 3, 4)
        level = torch.tensor([0, 1, 2, 1, 0])
        first = gate(z_inv, e_useful, level)["env_use_gate"]
        second = gate(z_inv, e_useful, level)["env_use_gate"]
        self.assertTrue(torch.equal(first, second))

    def test_checkpoint_round_trip_preserves_thresholds(self):
        args = minimal_model_args()
        first = StableST(args, adj=torch.eye(args.num_nodes), embed_size=args.d_model, output_dim=2)
        payload = {
            "model": first.state_dict(),
            "fpem_load_level_state": first.get_load_level_state_for_checkpoint(),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "checkpoint.pth")
            torch.save(payload, path)
            loaded = torch.load(path, map_location="cpu")
        second = StableST(args, adj=torch.eye(args.num_nodes), embed_size=args.d_model, output_dim=2)
        second.load_state_dict(loaded["model"], strict=True)
        second.load_load_level_state_from_checkpoint(loaded["fpem_load_level_state"])
        self.assertTrue(torch.equal(first.load_level_thresholds, second.load_level_thresholds))


if __name__ == "__main__":
    unittest.main()
