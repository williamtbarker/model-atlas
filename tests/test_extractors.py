import gzip
import json
import math
import tempfile
import unittest
from pathlib import Path
from exporters.bundled import ROOT, gpt2, deepseek4
from exporters.deepseek4_shapes import derive_catalog
from exporters.extract import from_torch
from exporters.model_ir import parameter_count, write_package


class ExtractorTests(unittest.TestCase):
    def test_gpt2_independent_formula_and_tied_embeddings(self):
        ir = gpt2()
        d = 768
        n = 12
        expected = 50257 * d + 1024 * d + n * (12 * d * d + 13 * d) + 2 * d
        self.assertEqual(parameter_count(ir), expected)
        self.assertEqual(expected, 124439808)

    def test_full_hero_name_and_storage_reconciliation(self):
        spec = json.loads((ROOT / "references/deepseek4-shapes.json").read_text())
        catalog = derive_catalog(spec["config"])
        index = json.loads(
            gzip.decompress((ROOT / "references/deepseek4-index.json.gz").read_bytes())
        )
        self.assertEqual(set(catalog), set(index["weight_map"]))
        widths = {"BF16": 2, "F32": 4, "I64": 8, "I8": 1, "F8_E4M3": 1, "F8_E8M0": 1}
        actual = sum(
            t["stored_elements"] * widths[t["stored_dtype"]] for t in catalog.values()
        )
        self.assertEqual(actual, index["metadata"]["total_size"])
        seen = 0
        for path in (ROOT / "references/headers").glob("*.gz"):
            header = json.loads(gzip.decompress(path.read_bytes()))
            for name, tensor in header.items():
                if name == "__metadata__":
                    continue
                self.assertEqual(catalog[name]["stored_shape"], tensor["shape"], name)
                self.assertEqual(catalog[name]["stored_dtype"], tensor["dtype"], name)
                seen += 1
        self.assertEqual(seen, 7030)

    def test_torch_traversal_never_executes_and_deduplicates_parameter_identity(self):
        class Param:
            shape = (3, 4)
            dtype = "torch.float32"

        p = Param()

        class Leaf:
            def named_parameters(self, recurse=False):
                return [("weight", p)]

            def named_buffers(self, recurse=False):
                return []

        class Model:
            def named_modules(self):
                return [("", self), ("first", Leaf()), ("second", Leaf())]

            def named_parameters(self, recurse=False):
                return []

            def named_buffers(self, recurse=False):
                return []

            def forward(self, *args):
                raise AssertionError("Inference must not run")

        ir = from_torch(Model())
        self.assertEqual(parameter_count(ir), 12)
        second = next(e for e in ir["entities"] if e["id"] == "module:second")
        self.assertEqual(second["tensorRefs"], ["first.weight"])

    def test_package_serialization_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.gz"
            b = Path(tmp) / "b.gz"
            ir = gpt2()
            write_package(ir, a)
            write_package(ir, b)
            self.assertEqual(a.read_bytes(), b.read_bytes())


if __name__ == "__main__":
    unittest.main()
