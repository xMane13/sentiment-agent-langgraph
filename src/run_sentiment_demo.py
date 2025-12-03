# src/run_sentiment_demo.py

import json
from pathlib import Path

from chains.sentiment_chain import build_sentiment_agent_chain


def main():
    # Construimos la cadena con la config A (más determinista)
    chain = build_sentiment_agent_chain(config="A")

    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "examples_raw.json"

    examples = json.loads(data_path.read_text(encoding="utf-8"))

    # Probamos con los primeros 5
    for ex in examples[:5]:
        print("=" * 80)
        print(f"ID: {ex['id']} | true label: {ex['label']}")
        print(f"TEXT: {ex['text']}\n")

        result = chain.invoke({"user_text": ex["text"]})

        print("Predicted sentiment:", result["sentiment"])
        print("Score:", result["score"])
        print("Short reason:", result["short_reason"])
        print("\nExplanation:\n", result["explanation"])
        print("\nSuggested reply:\n", result["suggested_reply"])
        print()


if __name__ == "__main__":
    main()
