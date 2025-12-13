from typing import Dict, List, Tuple
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
	precision_recall_fscore_support,
	accuracy_score,
	hamming_loss,
)


def _binarize(true_labels: List[List[str]], pred_labels: List[List[str]], all_labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
	"""Convert label lists to binary indicator matrices.

	Args:
		true_labels: List of lists of true JEL codes
		pred_labels: List of lists of predicted JEL codes
		all_labels: Ordered list of all possible labels

	Returns:
		(Y_true, Y_pred) as arrays with shape (n_samples, n_labels)
	"""
	label_index = {l: i for i, l in enumerate(all_labels)}
	n = len(true_labels)
	k = len(all_labels)
	y_true = np.zeros((n, k), dtype=int)
	y_pred = np.zeros((n, k), dtype=int)

	for r, labs in enumerate(true_labels):
		for l in labs:
			if l in label_index:
				y_true[r, label_index[l]] = 1

	for r, labs in enumerate(pred_labels):
		for l in labs:
			if l in label_index:
				y_pred[r, label_index[l]] = 1

	return y_true, y_pred


def compute_multilabel_metrics(true_labels: List[List[str]], pred_labels: List[List[str]], all_labels: List[str]) -> Dict:
	"""Compute standard multi-label metrics.

	Returns a dictionary with micro/macro precision/recall/F1, subset accuracy,
	hamming loss, and per-label metrics.
	"""
	y_true, y_pred = _binarize(true_labels, pred_labels, all_labels)

	# Subset accuracy: exact match of all labels per instance
	subset_acc = accuracy_score(y_true, y_pred)

	# Hamming loss: fraction of wrong labels
	h_loss = hamming_loss(y_true, y_pred)

	# Micro & Macro metrics
	prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
		y_true, y_pred, average="micro", zero_division=0
	)
	prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
		y_true, y_pred, average="macro", zero_division=0
	)

	# Per-label metrics
	per_label = {}
	prec_l, rec_l, f1_l, support_l = precision_recall_fscore_support(
		y_true, y_pred, average=None, zero_division=0
	)
	for i, lab in enumerate(all_labels):
		per_label[lab] = {
			"support": int(support_l[i]),
			"precision": float(prec_l[i]),
			"recall": float(rec_l[i]),
			"f1": float(f1_l[i]),
		}

	return {
		"subset_accuracy": float(subset_acc),
		"hamming_loss": float(h_loss),
		"precision_micro": float(prec_micro),
		"recall_micro": float(rec_micro),
		"f1_micro": float(f1_micro),
		"precision_macro": float(prec_macro),
		"recall_macro": float(rec_macro),
		"f1_macro": float(f1_macro),
		"per_label": per_label,
	}


def save_metrics(metrics: Dict, output_path: str) -> None:
	"""Save metrics to a JSON file."""
	out = Path(output_path)
	out.parent.mkdir(parents=True, exist_ok=True)
	with out.open("w", encoding="utf-8") as f:
		json.dump(metrics, f, ensure_ascii=False, indent=2)

