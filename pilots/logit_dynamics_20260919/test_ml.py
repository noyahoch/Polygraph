"""Small allocated-GPU gradient, checkpoint and feature integration checks."""
import os
import unittest

if not os.environ.get("SLURM_JOB_ID"):
    raise RuntimeError("Tests, including numerical CPU work, must run inside Slurm")

import io
import numpy as np
import torch
from .features import build_features, fit_normalizer, normalize
from .train import LayerHeads


class IntegrationTests(unittest.TestCase):
    def test_gradients_features_checkpoint(self):
        self.assertTrue(torch.cuda.is_available(), "GPU preflight is required")
        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)
        heads = LayerHeads().cuda()
        self.assertEqual(sum(p.numel() for p in heads.parameters()), 922800)
        cls = torch.randn(8, 12, 768, device="cuda")
        target = torch.arange(8, device="cuda")
        before = [head.weight.detach().clone() for head in heads.heads]
        optimizer = torch.optim.AdamW(heads.parameters(), lr=0.001, weight_decay=0.0)
        output = heads(cls)
        loss = sum(torch.nn.functional.cross_entropy(output[:, layer], target) for layer in range(12))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in heads.parameters()))
        optimizer.step()
        self.assertTrue(all(not torch.equal(previous, head.weight) for previous, head in zip(before, heads.heads)))
        final = torch.randn(8, 100).numpy()
        with torch.no_grad():
            reference = heads(cls).cpu().numpy()
        buffer = io.BytesIO()
        torch.save(heads.state_dict(), buffer)
        buffer.seek(0)
        restored = LayerHeads().cuda()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        with torch.no_grad():
            np.testing.assert_allclose(restored(cls).cpu().numpy(), reference, atol=1e-6, rtol=1e-6)
        features = build_features(reference, final)
        self.assertEqual(features.shape, (8, 85))
        norm = fit_normalizer(features[:6])
        normalized = torch.from_numpy(normalize(features, norm)).cuda()
        probe = torch.nn.Linear(85, 1).cuda()
        self.assertEqual(sum(p.numel() for p in probe.parameters()), 86)
        targets = torch.tensor([0, 1] * 4, device="cuda", dtype=torch.float32)
        before_probe = probe.weight.detach().clone()
        optimizer = torch.optim.AdamW(probe.parameters(), lr=0.001, weight_decay=0.01)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(probe(normalized).flatten(), targets)
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before_probe, probe.weight))
        self.assertTrue(torch.isfinite(probe(normalized)).all())


if __name__ == "__main__":
    unittest.main()
