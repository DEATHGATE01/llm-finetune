from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple


def normalize_text(text: str) -> str:
	lowered = text.lower().strip()
	lowered = re.sub(r"\s+", " ", lowered)
	return lowered


def tokenize(text: str) -> List[str]:
	return normalize_text(text).split()


def ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
	if len(tokens) < n:
		return []
	return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def modified_precision(reference: Sequence[str], candidate: Sequence[str], n: int) -> float:
	cand_ngrams = ngrams(candidate, n)
	if not cand_ngrams:
		return 0.0
	ref_counts = Counter(ngrams(reference, n))
	cand_counts = Counter(cand_ngrams)
	clipped = 0
	for ng, count in cand_counts.items():
		clipped += min(count, ref_counts.get(ng, 0))
	return clipped / max(1, sum(cand_counts.values()))


def brevity_penalty(reference_len: int, candidate_len: int) -> float:
	if candidate_len == 0:
		return 0.0
	if candidate_len > reference_len:
		return 1.0
	return math.exp(1 - (reference_len / candidate_len))


def bleu_score(reference_text: str, candidate_text: str, max_n: int = 2) -> float:
	reference = tokenize(reference_text)
	candidate = tokenize(candidate_text)
	if not candidate:
		return 0.0

	precisions = []
	for n in range(1, max_n + 1):
		p = modified_precision(reference, candidate, n)
		if p == 0:
			return 0.0
		precisions.append(p)

	geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
	bp = brevity_penalty(len(reference), len(candidate))
	return bp * geo_mean


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
	rows = len(a) + 1
	cols = len(b) + 1
	dp = [[0] * cols for _ in range(rows)]
	for i in range(1, rows):
		for j in range(1, cols):
			if a[i - 1] == b[j - 1]:
				dp[i][j] = dp[i - 1][j - 1] + 1
			else:
				dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
	return dp[-1][-1]


def rouge_l_f1(reference_text: str, candidate_text: str) -> float:
	reference = tokenize(reference_text)
	candidate = tokenize(candidate_text)
	if not reference or not candidate:
		return 0.0
	lcs = lcs_length(reference, candidate)
	precision = lcs / len(candidate)
	recall = lcs / len(reference)
	if precision + recall == 0:
		return 0.0
	return (2 * precision * recall) / (precision + recall)


def exact_match(reference_text: str, candidate_text: str) -> float:
	return 1.0 if normalize_text(reference_text) == normalize_text(candidate_text) else 0.0


def evaluate_pair(reference_text: str, candidate_text: str) -> Dict[str, float]:
	return {
		"bleu2": round(bleu_score(reference_text, candidate_text, max_n=2), 6),
		"rouge_l_f1": round(rouge_l_f1(reference_text, candidate_text), 6),
		"exact_match": round(exact_match(reference_text, candidate_text), 6),
	}


def aggregate_scores(rows: List[Dict[str, float]]) -> Dict[str, float]:
	if not rows:
		return {"bleu2": 0.0, "rouge_l_f1": 0.0, "exact_match": 0.0}

	keys = ["bleu2", "rouge_l_f1", "exact_match"]
	return {
		key: round(sum(r[key] for r in rows) / len(rows), 6)
		for key in keys
	}

