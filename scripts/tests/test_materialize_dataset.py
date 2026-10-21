import gzip
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('materialize_dataset', Path(__file__).resolve().parents[1] / 'materialize_dataset.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MaterializeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / 'restored'
        chunks = [b'{"id":"first"}\n', b'{"id":"second"}\n']
        self.expected = b''.join(chunks)
        # Manifest order is intentionally different from filename sorting.
        names = ['z.jsonl.gz', 'a.jsonl.gz']
        for name, data in zip(names, chunks):
            (self.root / name).write_bytes(gzip.compress(data))
        self.entry = {'configuration': 'tier-a', 'split': 'test', 'records': 2,
                      'source_sha256': hashlib.sha256(self.expected).hexdigest(),
                      'shards': [{'file': name} for name in names]}
        self.write_manifest()

    def write_manifest(self):
        (self.root / 'SPLITS.json').write_text(json.dumps([self.entry]))

    def test_order_and_verified_repeat(self):
        path = module.materialize(self.root, 'tier-a', 'test', self.output)
        self.assertEqual(path.read_bytes(), self.expected)
        self.assertEqual(module.materialize(self.root, 'tier-a', 'test', self.output), path)

    def test_bad_checksum_does_not_publish_partial_output(self):
        self.entry['source_sha256'] = '0' * 64
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'Checksum'):
            module.materialize(self.root, 'tier-a', 'test', self.output)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_existing_different_file_is_preserved(self):
        self.output.mkdir()
        path = self.output / 'test.jsonl'
        path.write_bytes(b'local data\n')
        with self.assertRaisesRegex(ValueError, 'not overwritten'):
            module.materialize(self.root, 'tier-a', 'test', self.output)
        self.assertEqual(path.read_bytes(), b'local data\n')


if __name__ == '__main__':
    unittest.main()
