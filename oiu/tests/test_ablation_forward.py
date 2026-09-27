# -*- coding: utf-8 -*-
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TestAblationForward(unittest.TestCase):
    def test_forward_variants_no_shape_error(self):
        import torch
        from src.models.ablation import ModuleAblation
        from src.models.unified_model import UnifiedEmotionModel

        device = "cpu"
        m = UnifiedEmotionModel(device=device)
        m.eval()
        tok = m.cpeb_model.tokenizer
        if tok is None:
            self.skipTest("无 tokenizer，跳过需 BERT 的消融前向测试")

        text = "测试消融"
        uid = 0
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=64, padding=True)
        ids = inputs["input_ids"].to(device)
        mask = inputs["attention_mask"].to(device)
        u = torch.tensor([uid], device=device)

        hist = ["上一句", "再上一句"]
        hi = tok(hist + [text], return_tensors="pt", truncation=True, max_length=128, padding=True)
        with torch.no_grad():
            hf = m.cpeb_model.encode_text(
                hi["input_ids"].to(device), hi["attention_mask"].to(device)
            )
        from src.models.thegn import create_dialogue_graph

        g = create_dialogue_graph(hist + [text], hf)

        for ab in [
            ModuleAblation.full(),
            ModuleAblation.no_cpeb(),
            ModuleAblation.no_mea(),
            ModuleAblation.no_thegn(),
            ModuleAblation(False, False, False),
        ]:
            with torch.no_grad():
                out = m.forward(
                    ids,
                    mask,
                    u,
                    [g],
                    None,
                    None,
                    return_intermediate=True,
                    ablation=ab,
                )
            self.assertEqual(out["emotion_final"].shape, (1, 8))


if __name__ == "__main__":
    unittest.main()
