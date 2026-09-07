"""Meaningful numerical checks of the serialized visualization evidence."""

import json
import gzip
import unittest
from pathlib import Path

import numpy as np

from train_trace import forward


class TraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = Path(__file__).parent
        cls.package = json.loads(gzip.decompress((cls.directory.parents[1] / "public/traces/recall.json.gz").read_bytes()))
        cls.weights = {k: np.asarray(v) for k, v in cls.package["weights"].items()}

    def test_weights_and_recordings_reproduce(self):
        with np.load(self.directory / "recall-weights.npz") as checkpoint:
            for name, weight in self.weights.items():
                np.testing.assert_array_equal(checkpoint[name], weight)
        for example in self.package["examples"]:
            ids = np.asarray([example["inputIds"]])
            for name, run in example["runs"].items():
                ablate = None if name == "baseline" else (int(name[1]), int(name[3]))
                probabilities, _, _ = forward(self.weights, ids, ablate)
                np.testing.assert_allclose(probabilities[0], run["probabilities"], rtol=0, atol=1e-12)

    def test_source_contributions_and_residuals(self):
        for example in self.package["examples"]:
            for run_name, run in example["runs"].items():
                for layer, block in enumerate(run["blocks"]):
                    a = np.asarray(block["attention"])
                    v = np.asarray(block["v"])
                    wo = self.weights[f"blocks.{layer}.wo"]
                    contributions = np.asarray(block["sourceContributions"])
                    # Explicit scalar sums, independent from export's einsum.
                    for h in range(2):
                        disabled = run_name == f"L{layer}H{h}"
                        for source in range(8):
                            for coordinate in range(16):
                                write = sum(float(v[h, source, k]) * float(wo[h * 8 + k, coordinate])
                                            for k in range(8))
                                for destination in range(8):
                                    expected = 0 if disabled else a[h, destination, source] * write
                                    self.assertAlmostEqual(contributions[h, destination, source, coordinate],
                                                           expected, places=11)
                    np.testing.assert_allclose(contributions.sum(axis=2), block["projectedHeads"], atol=1e-12)
                    np.testing.assert_allclose(contributions.sum(axis=(0, 2)), block["attentionOutput"], atol=1e-12)
                    np.testing.assert_allclose(np.asarray(block["input"]) + block["attentionOutput"], block["residual1"], atol=1e-12)
                    np.testing.assert_allclose(np.asarray(block["residual1"]) + block["mlpDown"], block["output"], atol=1e-12)

    def test_causal_mask_and_normalization(self):
        for example in self.package["examples"]:
            for run in example["runs"].values():
                for block in run["blocks"]:
                    for h, head in enumerate(block["attention"]):
                        for dest, row in enumerate(head):
                            self.assertAlmostEqual(sum(row), 1.0, places=13)
                            for source in range(8):
                                if source > dest:
                                    self.assertIsNone(block["scores"][h][dest][source])
                                    self.assertEqual(row[source], 0.0)
                                else:
                                    self.assertTrue(np.isfinite(block["scores"][h][dest][source]))

    def test_interventions_rerun_later_computation(self):
        for example in self.package["examples"]:
            baseline = example["runs"]["baseline"]
            self.assertEqual(int(np.argmax(baseline["probabilities"][-1])), example["targetId"])
            for h in range(2):
                ablated = example["runs"][f"L0H{h}"]
                self.assertTrue(np.all(np.asarray(ablated["blocks"][0]["headOutputs"])[h] == 0))
                self.assertFalse(np.allclose(ablated["blocks"][1]["attention"], baseline["blocks"][1]["attention"]))
            strongest = example["runs"][example["strongestAblation"]]
            self.assertLess(strongest["probabilities"][-1][example["targetId"]], 0.01)


if __name__ == "__main__":
    unittest.main()
