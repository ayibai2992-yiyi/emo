# -*- coding: utf-8 -*-
"""
算法创新点集成测试：
- RACF / graph fallback（沿用）
- CPEB 反事实 + 辅助损失
- Hybrid Prototype+MAML
- 危机辅任务头 + 多任务损失
- 阈值校准

运行（在 oiu 目录下）:
  python -m pytest tests/test_algorithm_innovations.py -q
  python tests/test_algorithm_innovations.py
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.cpeb import CPEBModel
from src.models.meta_learner import HybridPrototypeMAMLAdapter, MetaEmotionAdapter
from src.models.thegn import create_dialogue_graph
from src.models.unified_model import EvidenceAwareFusion, UnifiedEmotionModel
from src.training.losses import UnifiedLoss
from src.evaluation.threshold_calibration import ThresholdCalibrator


class FakeBert(nn.Module):
    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(128, hidden_size)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.embedding(input_ids)
        if attention_mask is not None:
            hidden = hidden * attention_mask.unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=hidden)


class AlgorithmInnovationTest(unittest.TestCase):
    def _build_model(self, adapter_type: str = "hybrid"):
        with patch("src.models.cpeb.AutoModel.from_pretrained", return_value=FakeBert()), \
             patch("src.models.cpeb.AutoTokenizer.from_pretrained", return_value=object()):
            return UnifiedEmotionModel(
                bert_model_name="fake-bert",
                num_emotions=8,
                hidden_size=16,
                graph_hidden_size=8,
                num_graph_layers=2,
                num_attention_heads=1,
                meta_adapter_type=adapter_type,
                num_inner_steps=1,
                dropout=0.0,
                device="cpu",
                proto_max_shots=5,
                maml_full_shots=20,
            )

    def test_racf_fusion_weights_sum_to_one(self):
        fusion = EvidenceAwareFusion(8, 16, 8, dropout=0.0)
        out = fusion(
            F.softmax(torch.randn(4, 8), dim=1),
            F.softmax(torch.randn(4, 8), dim=1),
            F.softmax(torch.randn(4, 8), dim=1),
            torch.randn(4, 16),
            torch.tensor([[1.0], [0.0], [1.0], [0.0]]),
        )
        self.assertEqual(out["fusion_weights"].shape, (4, 3))
        self.assertTrue(torch.allclose(out["fusion_weights"].sum(dim=1), torch.ones(4)))

    def test_cpeb_counterfactual_and_aux_losses(self):
        with patch("src.models.cpeb.AutoModel.from_pretrained", return_value=FakeBert()), \
             patch("src.models.cpeb.AutoTokenizer.from_pretrained", return_value=object()):
            model = CPEBModel(
                bert_model_name="fake-bert",
                num_emotions=8,
                hidden_size=16,
                dropout=0.0,
                device="cpu",
            )
        input_ids = torch.randint(0, 64, (4, 5))
        attention_mask = torch.ones_like(input_ids)
        user_ids = torch.tensor([1, 2, 1, 3])
        outputs = model(
            input_ids, attention_mask, user_ids,
            return_baseline=True, return_counterfactual=True
        )
        self.assertIn("emotion_counterfactual", outputs)
        self.assertIn("causal_aux_losses", outputs)
        aux = outputs["causal_aux_losses"]
        for key in ("baseline_kl", "despair_consistency", "counterfactual_consistency"):
            self.assertIn(key, aux)
            self.assertTrue(torch.is_tensor(aux[key]))

        # 不同用户基线应能改变去偏结果（至少形状一致）
        self.assertEqual(outputs["emotion_debiased"].shape, outputs["emotion_counterfactual"].shape)

    def test_hybrid_meta_adapter_shot_curve(self):
        adapter = HybridPrototypeMAMLAdapter(
            input_size=16, hidden_size=8, num_emotions=8,
            num_inner_steps=1, proto_max_shots=5, maml_full_shots=20,
        )
        query = torch.randn(2, 16)
        for k in [1, 5, 10, 20]:
            support = torch.randn(k, 16)
            labels = torch.randint(0, 8, (k,))
            probs = adapter(query, support, labels)
            self.assertEqual(probs.shape, (2, 8))
            self.assertTrue(torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5))
            self.assertAlmostEqual(adapter._mix_weight(k), 0.0 if k < 5 else (1.0 if k >= 20 else (k - 5) / 15.0))

    def test_crisis_head_and_multitask_loss(self):
        model = self._build_model("hybrid")
        input_ids = torch.randint(0, 64, (3, 5))
        attention_mask = torch.ones_like(input_ids)
        user_ids = torch.tensor([1, 2, 3])
        labels = torch.tensor([1, 7, 0])  # 含绝望

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            user_ids=user_ids,
            conversation_graphs=None,
            return_intermediate=True,
        )
        self.assertEqual(outputs["crisis_prob"].shape, (3,))
        self.assertTrue(torch.all((outputs["crisis_prob"] >= 0) & (outputs["crisis_prob"] <= 1)))
        self.assertTrue(torch.all(outputs["graph_availability"] == 0))

        criterion = UnifiedLoss(num_emotions=8)
        losses = criterion(outputs, labels, compute_causal=True, compute_crisis=True)
        self.assertIn("crisis", losses)
        self.assertIn("causal", losses)
        self.assertIn("total", losses)
        losses["total"].backward()
        self.assertIsNotNone(model.crisis_head[0].weight.grad)
        self.assertGreater(model.crisis_head[0].weight.grad.abs().sum().item(), 0.0)

    def test_cold_start_vs_history_aware_graph(self):
        model = self._build_model("hybrid")
        input_ids = torch.randint(0, 64, (2, 5))
        attention_mask = torch.ones_like(input_ids)
        user_ids = torch.tensor([1, 2])
        graph = create_dialogue_graph(["a", "b", "c"], torch.randn(3, 16))
        outputs = model(
            input_ids, attention_mask, user_ids,
            conversation_graphs=[graph, None],
            return_intermediate=True,
        )
        self.assertEqual(outputs["graph_availability"].squeeze(1).tolist(), [1.0, 0.0])

    def test_threshold_calibrator_best_f1_and_fixed_recall(self):
        rng = np.random.default_rng(0)
        y_true = np.array([0] * 80 + [1] * 20)
        y_score = np.concatenate([
            rng.uniform(0.0, 0.4, size=80),
            rng.uniform(0.5, 1.0, size=20),
        ])

        cal_f1 = ThresholdCalibrator(strategy="best_f1")
        res_f1 = cal_f1.fit(y_true, y_score)
        self.assertTrue(cal_f1.is_locked())
        self.assertGreaterEqual(res_f1.f1, 0.0)
        preds = cal_f1.predict(y_score)
        self.assertEqual(preds.shape, y_true.shape)

        cal_rec = ThresholdCalibrator(strategy="fixed_recall", target_recall=0.85)
        res_rec = cal_rec.fit(y_true, y_score)
        self.assertGreaterEqual(res_rec.recall, 0.85 - 1e-6)
        # 测试集只应用锁定阈值
        locked = res_rec.threshold
        self.assertEqual(cal_rec.threshold, locked)


if __name__ == "__main__":
    unittest.main()
