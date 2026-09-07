"""Independent structural checks for the metadata-only Qwen package."""
import gzip
import hashlib
import json
import math
import unittest
from collections import defaultdict
from pathlib import Path

from exporters.qwen35 import derive_catalog, load_references, qwen35
from exporters.model_ir import parameter_count, validate

ROOT = Path(__file__).resolve().parent.parent


class QwenStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.provenance, cls.checkpoint = load_references(ROOT)
        cls.ir = qwen35(ROOT)
        cls.entities = {e['id']: e for e in cls.ir['entities']}
        cls.tensors = {t['id']: t for t in cls.ir['tensors']}

    def test_exact_scoped_inventory_and_browser_budgets(self):
        validate(self.ir)
        catalog = derive_catalog(self.config)
        scoped = {name for name in self.checkpoint['weight_map']
                  if name.startswith('model.language_model.') or name == 'lm_head.weight'}
        self.assertEqual(set(catalog), scoped)
        self.assertEqual(len(catalog), 1038)
        self.assertEqual(parameter_count(self.ir), 396346350336)
        self.assertEqual(sum(math.prod(s) for s in catalog.values()), 396346350336)
        self.assertEqual(self.ir['metadata']['scope'], 'Text decoder')
        self.assertEqual(len(self.ir['metadata']['excludedComponents']), 2)
        self.assertFalse(self.ir['metadata']['reconciliation']['headersVerified'])
        self.assertFalse(any(t['id'].startswith(('mtp.', 'model.visual.')) for t in self.ir['tensors']))
        self.assertLess(len(self.ir['entities']), 200000)
        self.assertLess(len(self.ir['tensors']), 100000)
        raw = gzip.decompress((ROOT / 'public/models/qwen35-397b.atlas.json.gz').read_bytes())
        self.assertLess(len(raw), 64000000)
        self.assertEqual(json.loads(raw), self.ir)

    def test_fused_expert_views_partition_storage_without_duplicates(self):
        catalog = derive_catalog(self.config)
        partitions = defaultdict(list)
        for t in self.ir['tensors']:
            storage = t.get('storage')
            if storage:
                name = storage['checkpointTensor']
                self.assertEqual(storage['checkpointShape'], catalog[name])
                start = int(storage['offsetElements'])
                partitions[name].append((start, start + math.prod(t['shape'])))
                self.assertNotIn(name, self.tensors)
            else:
                self.assertEqual(t['shape'], catalog[t['id']])
        self.assertEqual(len(partitions), 120)
        for name, spans in partitions.items():
            cursor = 0
            for start, end in sorted(spans):
                self.assertEqual(start, cursor, f'Gap or duplicate in {name}')
                cursor = end
            self.assertEqual(cursor, math.prod(catalog[name]))
        for i in [0, 31, 59]:
            expert = f'layers.{i}.ffn.experts.511'
            selected = [t for t in self.ir['tensors'] if t['id'].startswith(expert + '.')]
            self.assertEqual(len(selected), 3)
            self.assertEqual(sum(math.prod(t['shape']) for t in selected), 12582912)

    def test_hybrid_schedule_states_and_gates_are_distinct(self):
        schedule = self.config['text_config']['layer_types']
        kinds = [self.entities[f'layers.{i}.attn']['kind'] for i in range(60)]
        self.assertEqual(kinds.count('LinearAttention'), 45)
        self.assertEqual(kinds.count('AttentionHead'), 15)
        for i, token_mixer in enumerate(schedule):
            attention = self.entities[f'layers.{i}.attn']
            if token_mixer == 'linear_attention':
                self.assertEqual(attention['kind'], 'LinearAttention')
                self.assertEqual(attention['attrs']['stateShape'], [64, 128, 128])
                self.assertEqual(attention['attrs']['keyHeads'], 16)
                self.assertNotIn('tensorId', self.entities[f'layers.{i}.attn.state'])
            else:
                self.assertEqual(attention['attrs']['queryHeads'], 32)
                self.assertEqual(attention['attrs']['kvHeads'], 2)
                self.assertTrue(attention['attrs']['outputGate'])
                self.assertEqual(self.tensors[f'model.language_model.layers.{i}.self_attn.q_proj.weight']['shape'], [16384, 4096])
        self.assertEqual(self.entities['layers.0.ffn.gate']['attrs']['topK'], 10)
        self.assertEqual(self.entities['layers.0.ffn.experts']['attrs']['count'], 512)
        self.assertEqual(self.entities['layers.0.ffn.shared_gate']['kind'], 'Gate')

    def test_pinned_config_and_checkpoint_index_hashes(self):
        refs = ROOT / 'references'
        self.assertEqual(hashlib.sha256((refs / 'qwen35-config.json').read_bytes()).hexdigest(), self.provenance['sha256']['config.json'])
        self.assertEqual(hashlib.sha256(gzip.decompress((refs / 'qwen35-index.json.gz').read_bytes())).hexdigest(), self.provenance['sha256']['model.safetensors.index.json'])


if __name__ == '__main__':
    unittest.main()
