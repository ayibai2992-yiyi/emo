# -*- coding: utf-8 -*-
"""
P0 冒烟：Dataset 按用户划分 + 配置读写闭环。

运行（在 oiu 目录）:
  python tests/test_p0_dataset_and_config.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class P0DatasetConfigTest(unittest.TestCase):
    def test_user_split_no_leakage(self):
        from src.data.dataset import load_and_split_csv

        rows = []
        for u in range(10):
            for i in range(5):
                rows.append(
                    {
                        "user_id": f"u{u}",
                        "text": f"句子{u}-{i}",
                        "label_id": i % 8,
                        "is_crisis": 1 if i % 8 == 7 else 0,
                        "conversation_history": f"历史{u}",
                    }
                )
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "mini.csv")
            pd.DataFrame(rows).to_csv(path, index=False)
            bundles = load_and_split_csv(path, seed=42)
            train_u = set(bundles.train_df["user_id"])
            val_u = set(bundles.val_df["user_id"])
            test_u = set(bundles.test_df["user_id"])
            self.assertEqual(len(train_u & val_u), 0)
            self.assertEqual(len(train_u & test_u), 0)
            self.assertEqual(len(val_u & test_u), 0)
            self.assertTrue(bundles.meta["n_train"] > 0)
            self.assertTrue(min(bundles.user_to_idx.values()) >= 1)
            self.assertNotIn(0, bundles.user_to_idx.values())  # 0 预留给未知用户

    def test_config_yaml_roundtrip(self):
        from src.utils.config import get_default_config, Config

        cfg = get_default_config()
        cfg.training.batch_size = 4
        cfg.model.meta_learning_algorithm = "hybrid"
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "cfg.yaml")
            cfg.to_yaml(path)
            loaded = Config.from_yaml(path)
            self.assertEqual(loaded.training.batch_size, 4)
            self.assertEqual(loaded.model.meta_learning_algorithm, "hybrid")
            self.assertEqual(loaded.training.freeze_bert_layers, cfg.training.freeze_bert_layers)

    def test_risk_score_not_saturated(self):
        from src.monitoring.deep_analyzer import PersonalizedDeepAnalyzer
        import numpy as np

        a = PersonalizedDeepAnalyzer(allow_heuristic=True)
        vec = np.array([0.4, 0.1, 0.05, 0.2, 0.1, 0.05, 0.05, 0.05])
        score = a.calculate_risk_score(vec)
        self.assertLess(score, 9.0)  # 不应轻易顶满
        score2 = a.calculate_risk_score(vec, crisis_prob=0.3)
        self.assertAlmostEqual(score2, 3.0, places=5)


if __name__ == "__main__":
    unittest.main()
