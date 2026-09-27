# -*- coding: utf-8 -*-
"""
最小单测：UnifiedEmotionModel.predict() 文本 + 历史，不得出现 shape 错误。
运行（在 oiu 目录下）:
  python -m pytest tests/test_unified_model_predict.py -q
或:
  python tests/test_unified_model_predict.py
"""

from __future__ import annotations

import os
import sys
import unittest

# 保证以仓库 oiu 为根时可导入 src
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TestUnifiedPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

    def test_predict_with_history_no_shape_error(self):
        import torch
        from src.models.unified_model import UnifiedEmotionModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = UnifiedEmotionModel(
            bert_model_name="bert-base-chinese",
            num_emotions=8,
            hidden_size=768,
            graph_hidden_size=256,
            num_graph_layers=3,
            num_attention_heads=8,
            meta_adapter_type="maml",
            device=device,
        )
        model.to(device)
        model.eval()

        text = "最近压力很大，感觉有点撑不住了。"
        history = ["昨晚失眠", "今天任务很多", "心情一直不太好"]
        out = model.predict(text=text, user_id=1, conversation_history=history, support_examples=None)

        self.assertIn("emotion_distribution", out)
        self.assertIn("intermediate_results", out)
        self.assertEqual(len(out["emotion_distribution"]), 8)
        probs = list(out["emotion_distribution"].values())
        s = sum(float(p) for p in probs)
        self.assertAlmostEqual(s, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
