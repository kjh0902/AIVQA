from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_rag_pipeline


class RagPipelineCacheTest(unittest.TestCase):
    def _args(self, root: Path):
        with patch("sys.argv", ["run_rag_pipeline.py"]):
            args = run_rag_pipeline.parse_args()
        for split in ("train", "validation", "test"):
            path = root / f"{split}.json"
            path.write_text(json.dumps([]), encoding="utf-8")
            setattr(args, f"{split}_json", path)
        args.dataset_root = root
        return args

    def test_training_cli_contains_no_retrieval_options(self) -> None:
        with patch("sys.argv", ["run_rag_pipeline.py"]):
            args = run_rag_pipeline.parse_args()
        for removed_option in (
            "qdrant_path",
            "qdrant_url",
            "collection",
            "rag_device",
            "search_max_new_tokens",
        ):
            self.assertFalse(hasattr(args, removed_option))

    def test_missing_cache_fails_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = self._args(root)
            with patch.object(run_rag_pipeline, "RAG_CACHE_DIR", root / "rag_cache"):
                with self.assertRaisesRegex(FileNotFoundError, "build_rag_cache.py"):
                    run_rag_pipeline.validate_args(args)

    def test_all_three_caches_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = self._args(root)
            cache_dir = root / "rag_cache"
            cache_dir.mkdir()
            for split in ("train", "validation", "test"):
                (cache_dir / f"{split}.json").write_text("[]", encoding="utf-8")
            with patch.object(run_rag_pipeline, "RAG_CACHE_DIR", cache_dir):
                run_rag_pipeline.validate_args(args)


if __name__ == "__main__":
    unittest.main()
