import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from finreason.adapter_assets import install_adapter, verify_adapter
from finreason.generation_guard import install_guard
from finreason.infer import item_seed, sha
from scripts.score_predictions import verify_run_spec


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.zip = self.root / 'adapter.zip'
        config = json.dumps({'peft_type': 'LORA', 'task_type': 'CAUSAL_LM'}).encode()
        self.payload = {'adapter_model.safetensors': b'fixture-not-model-weights', 'adapter_config.json': config}
        self.record = self.archive(self.payload)

    def archive(self, payload):
        with zipfile.ZipFile(self.zip, 'w') as z:
            for k, v in payload.items():
                z.writestr(k, v)
        return {'bytes': self.zip.stat().st_size,
                'sha256': hashlib.sha256(self.zip.read_bytes()).hexdigest(),
                'files': {k: {'bytes': len(v), 'sha256': hashlib.sha256(v).hexdigest()} for k, v in payload.items()}}

    def test_verified_install_and_tampered_config_refusal(self):
        out = self.root / 'installed'
        install_adapter(self.record, out, self.zip)
        manifest = {'adapters': {'sft': self.record}}
        self.assertTrue(verify_adapter(out, manifest, 'sft')['released_asset_verified'])
        (out / 'adapter_config.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'changed'):
            verify_adapter(out, manifest, 'sft')

    def test_corrupted_archive_leaves_no_install(self):
        self.zip.write_bytes(self.zip.read_bytes() + b'corruption')
        out = self.root / 'installed'
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            install_adapter(self.record, out, self.zip)
        self.assertFalse(out.exists())

    def test_correct_hash_cannot_authorize_path_traversal(self):
        record = self.archive({'../escaped': b'x'})
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            install_adapter(record, self.root / 'installed', self.zip)
        self.assertFalse((self.root / 'escaped').exists())

    def test_existing_destination_not_overwritten(self):
        out = self.root / 'installed'
        out.mkdir()
        (out / 'keep').write_text('preserve')
        with self.assertRaises(FileExistsError):
            install_adapter(self.record, out, self.zip)
        self.assertEqual((out / 'keep').read_text(), 'preserve')

    def test_weight_tag_mismatch_rejected(self):
        out = self.root / 'installed'
        install_adapter(self.record, out, self.zip)
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            verify_adapter(out, {'adapters': {'sft': self.record}}, 'other')


class GenerationGuardTests(unittest.TestCase):
    def test_fallback_and_resolved_drift_are_rejected(self):
        class Effective:
            temperature = 0.7
        class Base:
            def _prepare_generation_config(self, config, **kwargs):
                return Effective(), kwargs
        base = Base()
        calls = install_guard(base, {'temperature': 0.7, 'use_model_defaults': False})
        with self.assertRaisesRegex(ValueError, 'fallback'):
            base._prepare_generation_config(None)
        base._prepare_generation_config(None, use_model_defaults=False)
        self.assertEqual(len(calls), 1)
        Effective.temperature = 0.6
        with self.assertRaisesRegex(ValueError, 'changed'):
            base._prepare_generation_config(None, use_model_defaults=False)
        self.assertEqual(len(calls), 1)


class ScoreIdentityTests(unittest.TestCase):
    def test_changed_seed_prompt_or_missing_actual_config_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            prompts = Path(folder) / 'prompts.jsonl'
            prompts.write_text('prompt source')
            spec = dict(prompts_sha256=sha(prompts), config={'seed': 42}, generation_policy={'temperature': .7})
            pred = dict(id='x', seed=item_seed(42, 'x'), generation_resolver_calls=1, effective_generation={'temperature': .7})
            verify_run_spec(spec, prompts, [pred])
            for field, bad in [('seed', 1), ('generation_resolver_calls', 0), ('effective_generation', {'temperature': .6})]:
                changed = dict(pred, **{field: bad})
                with self.assertRaises(ValueError):
                    verify_run_spec(spec, prompts, [changed])
            prompts.write_text('different prompt source')
            with self.assertRaisesRegex(ValueError, 'different prompts'):
                verify_run_spec(spec, prompts, [pred])


if __name__ == '__main__':
    unittest.main()
