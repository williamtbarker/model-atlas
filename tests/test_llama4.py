"""Regression gates for the independent Llama 4 architecture inventory."""

import gzip
import json
import math
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from exporters.llama4 import llama4_maverick
from exporters.model_ir import parameter_count, write_package


class Llama4InventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ir = llama4_maverick()
        cls.entities = {e["id"]: e for e in cls.ir["entities"]}
        cls.tensors = {t["id"]: t for t in cls.ir["tensors"]}

    def test_full_inventory_reconciles_publisher_aggregate(self):
        # Count independently by architectural family, not by exported tensor list.
        d, f, dense_f, vocab = 5120, 8192, 16384, 202048
        attention_and_norm = 2 * d * 40 * 128 + 2 * d * 8 * 128 + 2 * d
        decoder = (
            2 * vocab * d
            + d
            + 48 * attention_and_norm
            + 24 * 3 * d * dense_f
            + 24 * (129 * 3 * d * f + 128 * d)
        )
        vd, vf = 1408, 5632
        vision_block = 4 * vd * vd + 4 * vd + 2 * vd * vf + vf + vd + 4 * vd
        vision = (
            vd * 3 * 14 * 14
            + vd
            + 577 * vd
            + 4 * vd
            + 34 * vision_block
            + vf * 4096
            + 4096**2
            + 4096 * d
        )
        self.assertEqual(decoder, 400_711_848_960)
        self.assertEqual(vision, 871_932_416)
        self.assertEqual(decoder + vision, 401_583_781_376)
        self.assertEqual(parameter_count(self.ir), decoder + vision)
        self.assertTrue(
            all(
                t["evidence"] == "derived" and "values" not in t
                for t in self.ir["tensors"]
            )
        )

    def test_dense_moe_and_gqa_patterns_are_explicit(self):
        blocks = [e for e in self.ir["entities"] if e["kind"] == "TransformerBlock"]
        vision = [e for e in self.ir["entities"] if e["kind"] == "VisionBlock"]
        self.assertEqual(len(blocks), 48)
        self.assertEqual(len(vision), 34)
        for i in range(48):
            p = f"language_model.model.layers.{i}"
            ff = self.entities[p + ".feed_forward"]
            self.assertEqual(
                ff["kind"], "ExpertCluster" if i % 2 else "ActivationLayer"
            )
            a = self.entities[p + ".self_attn"]["attrs"]
            self.assertEqual(
                (a["queryHeads"], a["kvHeads"], a["headDim"]), (40, 8, 128)
            )
            self.assertEqual(
                a["attentionPattern"],
                "global causal" if i % 4 == 3 else "chunked causal",
            )
            self.assertEqual(
                self.entities[p + ".self_attn.heads.39"]["attrs"]["kvGroupIndex"], 7
            )
            self.assertEqual(a["queryKeyNormalization"], "none")
            if i % 4 == 3:
                self.assertEqual(a["temperatureSettingEvidence"], "unavailable")
                self.assertIn("not established", a["queryPreprocessing"])
            if i % 2:
                router = self.entities[p + ".feed_forward.router"]
                self.assertEqual(router["attrs"]["topK"], 1)
                self.assertEqual(
                    ff["attrs"]["routingWeightApplication"], "expert input"
                )

    def test_packed_expert_partitions_cover_every_scalar_exactly_once(self):
        partitions = defaultdict(list)
        for t in self.ir["tensors"]:
            storage = t.get("storage")
            if storage:
                partitions[storage["sourceTensor"]].append(t)
        self.assertEqual(
            len(partitions), 48
        )  # Two packed arrays in each of 24 MoE blocks.
        for source, tensors in partitions.items():
            shape = tensors[0]["storage"]["sourceShape"]
            self.assertEqual(
                sum(math.prod(t["shape"]) for t in tensors), math.prod(shape)
            )
            intervals = defaultdict(list)
            for t in tensors:
                s = t["storage"]
                expert = s["expertIndex"]
                self.assertIn(expert, range(128))
                self.assertEqual(s["sourceShape"], shape)
                if source.endswith(".gate_up_proj"):
                    self.assertEqual(t["shape"], [8192, 5120])
                    intervals[expert].append((s["offset"], s["offset"] + t["shape"][0]))
                else:
                    self.assertEqual(t["shape"], [5120, 8192])
                    intervals[expert].append((0, shape[2]))
            self.assertEqual(len(intervals), 128)
            for slices in intervals.values():
                ordered = sorted(slices)
                self.assertEqual(ordered[0][0], 0)
                self.assertEqual(ordered[-1][1], shape[2])
                for a, b in zip(ordered, ordered[1:]):
                    self.assertEqual(a[1], b[0])

    def test_image_branch_fuses_upstream_and_is_not_a_decoder_extension(self):
        self.assertEqual(self.entities["vision_model"]["kind"], "VisionEncoder")
        self.assertEqual(
            self.entities["multi_modal_projector.linear_1"]["parentId"], "vision_model"
        )
        self.assertEqual(
            self.ir["metadata"]["initialScope"], "language_model.model.layers"
        )
        edges = {(e["from"], e["to"]) for e in self.ir["edges"]}
        self.assertIn(("multi_modal_projector.linear_1", "fusion"), edges)
        self.assertIn(("fusion", "language_model.model.layers"), edges)
        self.assertFalse(
            any(
                src.startswith("language_model.lm_head")
                and dst.startswith("vision_model")
                for src, dst in edges
            )
        )
        for e in self.ir["entities"]:
            if e["kind"] == "VisionBlock":
                self.assertFalse(e["attrs"]["causal"])

    def test_export_is_reproducible_and_contains_no_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.gz", Path(tmp) / "b.gz"
            write_package(self.ir, a)
            write_package(llama4_maverick(), b)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertLess(a.stat().st_size, 600_000)
            decoded = json.loads(gzip.decompress(a.read_bytes()))
            self.assertTrue(all("values" not in t for t in decoded["tensors"]))


if __name__ == "__main__":
    unittest.main()
