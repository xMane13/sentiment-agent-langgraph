# src/run_batch_demo.py

import json
from pathlib import Path

from chains.sentiment_chain import build_sentiment_agent_chain
from tools.stats_tools import compute_sentiment_stats, compute_accuracy_with_labels


def main():
    # Usamos la config A (más determinista) para este demo
    chain = build_sentiment_agent_chain(config="A")

    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "examples_raw.json"

    examples = json.loads(data_path.read_text(encoding="utf-8"))

    results = []

    for ex in examples:
        print("=" * 80)
        print(f"ID: {ex['id']} | true label: {ex['label']}")
        print(f"TEXT: {ex['text']}\n")

        out = chain.invoke({"user_text": ex["text"]})

        # copiar la salida y añadir la etiqueta real
        result = {
            "id": ex["id"],
            "text": ex["text"],
            "true_label": ex["label"],
            "sentiment": out["sentiment"],
            "score": out["score"],
            "short_reason": out["short_reason"],
            "explanation": out["explanation"],
            "suggested_reply": out["suggested_reply"],
        }
        results.append(result)

        print("Predicted sentiment:", result["sentiment"])
        print("Score:", result["score"])
        print("Short reason:", result["short_reason"])
        print()

    # ---- Estadísticas agregadas ----
    print("\n" + "#" * 80)
    print("AGGREGATED STATISTICS\n")

    stats = compute_sentiment_stats(results)
    acc = compute_accuracy_with_labels(results)

    print(f"Total texts analyzed: {stats['total']}")
    print("Counts:", stats["counts"])
    print("Distribution:")
    for label, frac in stats["distribution"].items():
        print(f"  {label}: {frac:.2f}")

    print("\nAccuracy vs true labels:")
    print(f"  total with label: {acc['total']}")
    print(f"  matched:         {acc['matched']}")
    print(f"  accuracy:        {acc['accuracy']:.2f}")


if __name__ == "__main__":
    main()
