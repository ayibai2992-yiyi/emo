import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.thegn import create_dialogue_graph
from src.models.unified_model import EvidenceAwareFusion, UnifiedEmotionModel


class FakeBert(nn.Module):
    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(128, hidden_size)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.embedding(input_ids)
        if attention_mask is not None:
            hidden = hidden * attention_mask.unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=hidden)


class UnifiedEmotionModelInnovationTest(unittest.TestCase):
    def _build_model(self):
        with patch("src.models.cpeb.AutoModel.from_pretrained", return_value=FakeBert()), \
             patch("src.models.cpeb.AutoTokenizer.from_pretrained", return_value=object()):
            return UnifiedEmotionModel(
                bert_model_name="fake-bert",
                num_emotions=8,
                hidden_size=16,
                graph_hidden_size=8,
                num_graph_layers=2,
                num_attention_heads=2,
                num_inner_steps=1,
                dropout=0.0,
                device="cpu"
            )

    def test_evidence_aware_fusion_outputs_valid_weights(self):
        fusion = EvidenceAwareFusion(
            num_emotions=8,
            context_size=16,
            hidden_size=8,
            dropout=0.0
        )
        batch_size = 3
        outputs = fusion(
            emotion_debiased=F.softmax(torch.randn(batch_size, 8), dim=1),
            emotion_adapted=F.softmax(torch.randn(batch_size, 8), dim=1),
            emotion_graph=F.softmax(torch.randn(batch_size, 8), dim=1),
            text_features=torch.randn(batch_size, 16),
            graph_availability=torch.tensor([[1.0], [0.0], [1.0]])
        )

        self.assertEqual(outputs["emotion_final"].shape, (batch_size, 8))
        self.assertEqual(outputs["fusion_weights"].shape, (batch_size, 3))
        self.assertTrue(torch.allclose(outputs["fusion_weights"].sum(dim=1), torch.ones(batch_size)))
        self.assertTrue(torch.allclose(outputs["emotion_final"].sum(dim=1), torch.ones(batch_size)))

    def test_no_graph_uses_learnable_fallback_instead_of_zero_vector(self):
        model = self._build_model()
        input_ids = torch.randint(0, 64, (2, 5))
        attention_mask = torch.ones_like(input_ids)
        user_ids = torch.tensor([1, 2])

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            user_ids=user_ids,
            conversation_graphs=None,
            return_intermediate=True
        )

        self.assertTrue(torch.all(outputs["graph_availability"] == 0))
        self.assertFalse(torch.allclose(outputs["emotion_graph"], torch.zeros_like(outputs["emotion_graph"])))
        self.assertTrue(torch.allclose(outputs["emotion_graph"].sum(dim=1), torch.ones(2), atol=1e-5))

    def test_partial_graph_batch_keeps_availability_and_gradients(self):
        model = self._build_model()
        input_ids = torch.randint(0, 64, (2, 5))
        attention_mask = torch.ones_like(input_ids)
        user_ids = torch.tensor([1, 2])

        graph_features = torch.randn(3, 16)
        graph = create_dialogue_graph(["第一轮", "第二轮", "当前轮"], graph_features)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            user_ids=user_ids,
            conversation_graphs=[graph],
            return_intermediate=True
        )

        self.assertEqual(outputs["graph_availability"].squeeze(1).tolist(), [1.0, 0.0])
        loss = F.cross_entropy(outputs["fusion_logits"], torch.tensor([1, 7]))
        loss.backward()

        fallback_grad = model.graph_fallback[0].weight.grad
        gate_grad = model.fusion_layer.gate_network[0].weight.grad
        self.assertIsNotNone(fallback_grad)
        self.assertIsNotNone(gate_grad)
        self.assertGreater(fallback_grad.abs().sum().item(), 0.0)
        self.assertGreater(gate_grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
