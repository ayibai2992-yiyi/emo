# -*- coding: utf-8 -*-
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.experiment.protocol import ExperimentProtocol
from src.experiment.splits import user_stratified_masks, summarize_split
from src.experiment.metrics_protocol import (
    compute_emotion_metrics,
    compute_crisis_detection_metrics,
    labels_to_crisis_binary,
    crisis_score_from_probs,
)


class TestProtocol(unittest.TestCase):
    def test_user_split_disjoint_users(self):
        users = np.array(["a"] * 5 + ["b"] * 5 + ["c"] * 5 + ["d"] * 5)
        tr, va, te = user_stratified_masks(users, 0.5, 0.25, 0.25, seed=0)
        u_tr = set(users[tr])
        u_va = set(users[va])
        u_te = set(users[te])
        self.assertTrue(u_tr.isdisjoint(u_va))
        self.assertTrue(u_tr.isdisjoint(u_te))
        self.assertTrue(u_va.isdisjoint(u_te))
        s = summarize_split(users, tr, va, te)
        self.assertIn("train", s)

    def test_metrics_shapes(self):
        y_true = np.array([0, 1, 7, 3, 7])
        y_pred = np.array([0, 2, 7, 3, 0])
        m = compute_emotion_metrics(y_true, y_pred)
        self.assertIn("macro_f1", m)
        cb = labels_to_crisis_binary(y_true, (7,))
        self.assertListEqual(cb.tolist(), [0, 0, 1, 0, 1])
        proba = np.eye(8)[y_pred] * 0.9 + 0.1 / 8
        proba /= proba.sum(axis=1, keepdims=True)
        scores = crisis_score_from_probs(proba, (7,))
        cm = compute_crisis_detection_metrics(cb, scores, threshold=0.3)
        self.assertIn("pr_auc", cm)
        self.assertIn("crisis_recall", cm)
        self.assertIn("false_positive_rate", cm)


if __name__ == "__main__":
    unittest.main()
