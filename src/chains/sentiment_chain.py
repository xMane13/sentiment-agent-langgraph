from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from models.llm_config import get_llm


# ---------- Carga de templates desde /prompts ----------

BASE_DIR = Path(__file__).resolve().parents[2]  # carpeta raíz del proyecto
PROMPTS_DIR = BASE_DIR / "prompts"


def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    return path.read_text(encoding="utf-8")


sentiment_prompt_tmpl = ChatPromptTemplate.from_template(
    load_prompt("sentiment_prompt.txt")
)
explanation_prompt_tmpl = ChatPromptTemplate.from_template(
    load_prompt("explanation_prompt.txt")
)
reply_prompt_tmpl = ChatPromptTemplate.from_template(
    load_prompt("reply_prompt.txt")
)


# ---------- Parser del JSON de sentimiento ----------

def _parse_sentiment_str(text: str) -> Dict[str, Any]:
    """
    Recibe el string bruto del modelo y extrae:
    - sentiment
    - score
    - short_reason
    a partir del JSON que incluimos en el prompt.
    """

    raw_str = str(text)

    # Intentar extraer el primer bloque {...}
    start = raw_str.find("{")
    end = raw_str.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON found in model output: {raw_str}")

    json_str = raw_str[start : end + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        json_str_clean = json_str.replace("\n", " ")
        data = json.loads(json_str_clean)

    sentiment = str(data.get("sentiment", "")).strip().lower()
    score = data.get("score", 0.0)
    try:
        score = float(score)
    except Exception:
        score = 0.0

    short_reason = data.get("short_reason", "")

    return {
        "sentiment": sentiment,
        "score": score,
        "short_reason": short_reason,
    }


# ---------- Builder de la "cadena" de análisis ----------

def build_sentiment_agent_chain(config: str = "A") -> RunnableLambda:
    """
    Devuelve un Runnable que:
      1) Usa el prompt de sentimiento (JSON) y lo parsea
      2) Genera una explicación
      3) Genera una respuesta sugerida

    Se llama igual que antes: chain({"user_text": "..."})
    """

    llm = get_llm(config)
    str_parser = StrOutputParser()

    # 1) Runnable para clasificación de sentimiento -> dict con sentiment, score, short_reason
    def _run_sentiment(inputs: Dict[str, Any]) -> Dict[str, Any]:
        user_text = inputs["user_text"]
        # prompt -> llm -> string
        raw_output = (
            sentiment_prompt_tmpl
            | llm
            | str_parser
        ).invoke({"user_text": user_text})

        sentiment_info = _parse_sentiment_str(raw_output)
        return {
            **inputs,
            **sentiment_info,
        }

    sentiment_runnable = RunnableLambda(_run_sentiment)

    # 2) Runnable para explicación
    def _run_explanation(inputs: Dict[str, Any]) -> Dict[str, Any]:
        user_text = inputs["user_text"]
        sentiment = inputs["sentiment"]
        short_reason = inputs["short_reason"]

        explanation = (
            explanation_prompt_tmpl
            | llm
            | str_parser
        ).invoke(
            {
                "user_text": user_text,
                "sentiment": sentiment,
                "short_reason": short_reason,
            }
        )

        return {
            **inputs,
            "explanation": explanation,
        }

    explanation_runnable = RunnableLambda(_run_explanation)

    # 3) Runnable para respuesta sugerida
    def _run_reply(inputs: Dict[str, Any]) -> Dict[str, Any]:
        user_text = inputs["user_text"]
        sentiment = inputs["sentiment"]

        reply = (
            reply_prompt_tmpl
            | llm
            | str_parser
        ).invoke(
            {
                "user_text": user_text,
                "sentiment": sentiment,
            }
        )

        return {
            **inputs,
            "suggested_reply": reply,
        }

    reply_runnable = RunnableLambda(_run_reply)

    # 4) Pipeline completo: aplica los tres pasos en orden
    def _full_pipeline(inputs: Dict[str, Any]) -> Dict[str, Any]:
        out1 = sentiment_runnable.invoke(inputs)
        out2 = explanation_runnable.invoke(out1)
        out3 = reply_runnable.invoke(out2)
        # devolvemos sólo las claves importantes
        return {
            "sentiment": out3["sentiment"],
            "score": out3["score"],
            "short_reason": out3["short_reason"],
            "explanation": out3["explanation"],
            "suggested_reply": out3["suggested_reply"],
        }

    return RunnableLambda(_full_pipeline)
