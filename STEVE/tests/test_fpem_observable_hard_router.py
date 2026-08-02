import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from models.fpem.load_level_gate import (
    HardEnvironmentUseRouter,
    hard_select_invariant_or_environment,
    select_load_expert,
)
from models.fpem.losses import load_level_expert_hard_router_losses
from models.fpem.observable_load_prior import (
    ensure_observable_load_prior_cache,
)


class IdentityScaler:
    @staticmethod
    def inverse_transform(value):
        return value


def raw_fixture():
    train = np.asarray(
        [
            [
                [[0.0, 0.0], [1.0, 2.0]],
                [[0.0, 1.0], [2.0, 4.0]],
            ],
            [
                [[0.0, 2.0], [3.0, 6.0]],
                [[0.0, 3.0], [4.0, 8.0]],
            ],
            [
                [[0.0, 4.0], [2.0, 4.0]],
                [[0.0, 4.0], [1.0, 2.0]],
            ],
        ],
        dtype=np.float64,
    )
    return {
        "train": {"x": train},
        "val": {"x": train[:2].copy()},
        "test": {"x": train[1:].copy()},
    }


class ObservableLoadPriorTests(unittest.TestCase):
    def test_cache_formula_zero_capacity_and_reuse(self):
        raw = raw_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observable_load_prior_k3_v1.npz"
            first = ensure_observable_load_prior_cache(
                path, raw, random_seed=17
            )
            self.assertTrue(first["created"])
            np.testing.assert_array_equal(
                first["CP"],
                np.max(raw["train"]["x"], axis=(0, 1)),
            )
            self.assertTrue(
                np.all(first["train_c_obs"][:, 0, 0] == 0)
            )
            expected = np.clip(
                np.ceil(
                    5.0
                    * raw["train"]["x"][:, -1, 1, 1]
                    / first["CP"][1, 1]
                ),
                0,
                5,
            )
            np.testing.assert_array_equal(
                first["train_c_obs"][:, 1, 1], expected
            )

            changed_same_shape = raw_fixture()
            changed_same_shape["train"]["x"].fill(999.0)
            second = ensure_observable_load_prior_cache(
                path, changed_same_shape, random_seed=999
            )
            self.assertFalse(second["created"])
            np.testing.assert_array_equal(first["CP"], second["CP"])
            np.testing.assert_array_equal(
                first["train_c_obs"], second["train_c_obs"]
            )
            self.assertEqual(
                int(second["random_assignment_seed"]), 17
            )

    def test_random_assignment_preserves_each_split_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = ensure_observable_load_prior_cache(
                Path(directory) / "prior.npz",
                raw_fixture(),
                random_seed=23,
            )
            for split in ("train", "val", "test"):
                np.testing.assert_array_equal(
                    np.bincount(
                        payload[f"{split}_expert_id"], minlength=3
                    ),
                    np.bincount(
                        payload[f"{split}_random_expert_id"],
                        minlength=3,
                    ),
                )


class HardRouterTests(unittest.TestCase):
    def test_hard_prediction_has_no_weighted_sum(self):
        y_inv = torch.tensor([[[[1.0]]], [[[2.0]]]])
        y_env = torch.tensor([[[[10.0]]], [[[20.0]]]])
        route = torch.tensor([0, 1])
        result = hard_select_invariant_or_environment(
            y_inv, y_env, route
        )
        self.assertTrue(
            torch.equal(
                result, torch.tensor([[[[1.0]]], [[[20.0]]]])
            )
        )

    def test_router_forward_has_no_target_argument(self):
        signature = inspect.signature(HardEnvironmentUseRouter.forward)
        self.assertEqual(
            list(signature.parameters),
            ["self", "z_inv", "e_useful", "load_level"],
        )

    def test_fixed_expert_selection_is_detached(self):
        heads = torch.arange(3 * 3, dtype=torch.float32).reshape(
            3, 3, 1, 1, 1
        )
        level = torch.tensor([0, 2, 1], requires_grad=False)
        selected = select_load_expert(heads, level)
        self.assertEqual(selected[:, 0, 0, 0].tolist(), [0.0, 5.0, 7.0])

    def test_balanced_expert_loss_and_detached_router_target(self):
        batch, experts = 6, 3
        heads = torch.randn(
            batch, experts, 1, 2, 2, requires_grad=True
        )
        y_inv = torch.randn(batch, 1, 2, 2, requires_grad=True)
        target = torch.full((batch, 1, 2, 2), 10.0)
        level = torch.tensor([0, 0, 1, 1, 2, 2])
        router_logits = torch.randn(batch, 2, requires_grad=True)
        route_out = {
            "y_env_heads": heads,
            "y_inv": y_inv,
            "load_level": level,
            "hard_router_logits": router_logits,
            "hard_router_predicted_route_id": router_logits.argmax(
                dim=-1
            ),
            "hard_route_id": router_logits.argmax(dim=-1),
            "hard_router_warmup_active": torch.tensor(0.0),
        }
        args = SimpleNamespace(
            yita=0.5,
            fpem_hard_router_relative_margin=0.01,
            fpem_hard_router_warmup_epochs=0,
            fpem_use_hard_environment_router=True,
        )
        expert_loss, router_loss, _logs, diagnostics = (
            load_level_expert_hard_router_losses(
                route_out,
                target,
                IdentityScaler(),
                args,
                training=True,
                epoch=1,
            )
        )
        self.assertFalse(diagnostics["router_target"].requires_grad)
        (expert_loss + router_loss).backward()
        for sample in range(batch):
            for expert in range(experts):
                nonzero = bool(
                    (heads.grad[sample, expert].abs() > 0).any()
                )
                self.assertEqual(nonzero, expert == int(level[sample]))
        self.assertIsNotNone(router_logits.grad)


if __name__ == "__main__":
    unittest.main()
