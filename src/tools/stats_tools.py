# src/tools/stats_tools.py

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List


def compute_sentiment_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calcula estadísticas de distribución de sentimientos.

    results: lista de dicts que deben tener al menos:
        - "sentiment": "positive" | "neutral" | "negative"

    Devuelve:
        {
          "total": int,
          "counts": {"positive": n_pos, ...},
          "distribution": {"positive": 0.x, ...}
        }
    """
    sentiments = [r.get("sentiment", "").lower().strip() for r in results]
    sentiments = [s for s in sentiments if s]  # quitar vacíos

    total = len(sentiments)
    if total == 0:
        return {"total": 0, "counts": {}, "distribution": {}}

    counts = Counter(sentiments)
    distribution = {label: counts[label] / total for label in counts}

    return {
        "total": total,
        "counts": dict(counts),
        "distribution": distribution,
    }


def compute_accuracy_with_labels(
    results: List[Dict[str, Any]],
    true_label_key: str = "true_label",
) -> Dict[str, Any]:
    """
    Calcula accuracy simple comparando predicciones con etiquetas verdaderas.

    results: lista de dicts con claves:
        - "sentiment": predicción
        - true_label_key: etiqueta real (por defecto "true_label")

    Devuelve:
        {
          "total": n,
          "matched": k,
          "accuracy": k / n
        }
    """
    total = 0
    matched = 0

    for r in results:
        pred = str(r.get("sentiment", "")).lower().strip()
        true = str(r.get(true_label_key, "")).lower().strip()
        if not true:
            continue
        total += 1
        if pred == true:
            matched += 1

    if total == 0:
        acc = 0.0
    else:
        acc = matched / total

    return {
        "total": total,
        "matched": matched,
        "accuracy": acc,
    }
