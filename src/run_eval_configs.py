# src/run_eval_configs.py

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from chains.sentiment_chain import build_sentiment_agent_chain
from tools.stats_tools import compute_sentiment_stats, compute_accuracy_with_labels


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "examples_raw.json"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)


def run_eval_for_config(config_name: str) -> Dict[str, Any]:
    """Ejecuta el agente sobre el dataset para una config dada (A o B)."""

    print("\n" + "=" * 80)
    print(f"EVALUATING CONFIG {config_name}")
    print("=" * 80)

    chain = build_sentiment_agent_chain(config=config_name)
    examples = json.loads(DATA_PATH.read_text(encoding="utf-8"))

    results: List[Dict[str, Any]] = []

    for ex in examples:
        user_text = ex["text"]
        true_label = ex["label"]

        out = chain.invoke({"user_text": user_text})

        result = {
            "id": ex["id"],
            "user_text": user_text,
            "true_label": true_label,
            "sentiment": out["sentiment"],
            "score": out["score"],
            "short_reason": out["short_reason"],
            "explanation": out["explanation"],
            "suggested_reply": out["suggested_reply"],
            "config": config_name,
        }
        results.append(result)

    # Estadísticas y accuracy
    stats = compute_sentiment_stats(results)
    acc = compute_accuracy_with_labels(results, true_label_key="true_label")

    summary = {
        "config": config_name,
        "stats": stats,
        "accuracy": acc,
        "n_examples": len(results),
    }

    # Guardar log a disco
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"eval_{config_name}_{timestamp}.json"

    payload = {
        "config": config_name,
        "created_at": timestamp,
        "summary": summary,
        "results": results,
    }
    log_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved log for config {config_name} in: {log_path}")
    print("\nSummary:")
    print(f"- Total examples: {summary['n_examples']}")
    print(f"- Accuracy: {summary['accuracy']['accuracy']:.2f}")
    print("- Distribution:", summary["stats"]["distribution"])

    return summary


def main():
    summaries = {}
    for cfg in ["A", "B"]:
        summaries[cfg] = run_eval_for_config(cfg)

    print("\n" + "#" * 80)
    print("COMPARISON A vs B")
    print("#" * 80)

    for cfg, s in summaries.items():
        acc = s["accuracy"]["accuracy"]
        dist = s["stats"]["distribution"]
        print(f"\nConfig {cfg}:")
        print(f"  - Accuracy: {acc:.2f}")
        print(f"  - Distribution: {dist}")


if __name__ == "__main__":
    main()
