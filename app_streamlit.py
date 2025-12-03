import sys
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st

# -------------------------------------------------------------------
# Ajustar path para poder importar desde src/
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from chains.sentiment_chain import build_sentiment_agent_chain
from tools.stats_tools import compute_sentiment_stats


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def get_chain(config: str = "A"):
    # Devuelve (y cachea en session_state) la cadena de análisis para
    # la configuración indicada ("A" o "B").
    if "chains" not in st.session_state:
        st.session_state["chains"] = {}

    if config not in st.session_state["chains"]:
        st.session_state["chains"][config] = build_sentiment_agent_chain(config=config)

    return st.session_state["chains"][config]


def run_single_analysis(text: str, config: str = "A") -> Dict[str, Any]:
    chain = get_chain(config)
    out = chain.invoke({"user_text": text})
    return out


def run_batch_analysis(
    texts: List[str],
    config: str = "A",
) -> List[Dict[str, Any]]:
    chain = get_chain(config)
    results: List[Dict[str, Any]] = []
    for t in texts:
        out = chain.invoke({"user_text": t})
        results.append(
            {
                "text": t,
                "sentiment": out["sentiment"],
                "score": out["score"],
                "short_reason": out["short_reason"],
                "explanation": out["explanation"],
                "suggested_reply": out["suggested_reply"],
            }
        )
    return results


def parse_batch_input(raw_text: str) -> List[str]:
    # Convierte el texto pegado por el usuario en una lista de comentarios.
    # Reglas:
    # - Si contiene '||', se usa ese separador.
    # - Si no, se usa una línea por comentario.
    raw_text = raw_text.strip()
    if not raw_text:
        return []

    if "||" in raw_text:
        parts = [p.strip() for p in raw_text.split("||") if p.strip()]
    else:
        parts = [line.strip() for line in raw_text.split("\n") if line.strip()]

    return parts


# -------------------------------------------------------------------
# UI principal
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Sentiment & Feedback Agent",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Sentiment & Feedback Agent")
st.markdown(
    """
Este agente analiza comentarios de usuarios (en español o inglés), 
clasifica el sentimiento y sugiere una respuesta de atención al cliente.  

Puedes usarlo en modo **Single** (un comentario) o **Batch** (varios a la vez).
"""
)

# Sidebar: configuración
st.sidebar.header("⚙️ Configuración")

mode = st.sidebar.radio(
    "Modo de uso",
    options=["Single comment", "Batch analysis"],
    index=0,
)

config_choice = st.sidebar.radio(
    "Modelo / parámetros",
    options=[
        "A - Deterministic (temp=0.1, top_p=0.8, top_k=30)",
        "B - Creative (temp=0.7, top_p=0.95, top_k=50)",
    ],
    index=0,
)

config = "A" if config_choice.startswith("A") else "B"

st.sidebar.info(
    "Config A es más estable/determinista.\n\n"
    "Config B es más creativa y puede variar más las respuestas."
)

# Inicializar almacenamiento de resultados en sesión
if "session_results" not in st.session_state:
    st.session_state["session_results"] = []  # lista de dicts


# -------------------------------------------------------------------
# Modo SINGLE
# -------------------------------------------------------------------

if mode == "Single comment":
    st.subheader("🔍 Single comment analysis")

    text = st.text_area(
        "Escribe o pega un comentario:",
        height=150,
        placeholder="Ejemplo: El producto llegó rápido y en perfectas condiciones. Muy satisfecho.",
    )

    if st.button("Analizar comentario", type="primary"):
        if not text.strip():
            st.warning("Por favor, ingresa un comentario primero.")
        else:
            with st.spinner("Analizando sentimiento..."):
                out = run_single_analysis(text.strip(), config=config)

            # Guardar en resultados de sesión
            st.session_state["session_results"].append(
                {
                    "text": text.strip(),
                    "sentiment": out["sentiment"],
                    "score": out["score"],
                    "short_reason": out["short_reason"],
                    "explanation": out["explanation"],
                    "suggested_reply": out["suggested_reply"],
                    "config": config,
                }
            )

            # Mostrar resultados
            st.markdown("### Resultado")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"**Sentiment:** `{out['sentiment']}`  \n"
                    f"**Score:** `{out['score']:.2f}`"
                )
                st.markdown("**Short reason:**")
                st.write(out["short_reason"])
            with col2:
                st.markdown("**Explanation:**")
                st.write(out["explanation"])

            st.markdown("**Suggested reply:**")
            st.success(out["suggested_reply"])


# -------------------------------------------------------------------
# Modo BATCH
# -------------------------------------------------------------------

if mode == "Batch analysis":
    st.subheader("📊 Batch sentiment analysis")

    st.markdown(
        """
Puedes pegar varios comentarios:

- O bien separados por líneas (un comentario por línea),
- O bien en una sola línea usando `||` como separador.

Ejemplo:

```text
El envío llegó con retraso y nadie ayudó. || Amazing quality! I will definitely buy again. || El producto está bien, nada especial.
```
"""
    )

    raw_batch = st.text_area(
        "Comentarios (batch):",
        height=200,
    )

    if st.button("Analizar batch", type="primary"):
        texts = parse_batch_input(raw_batch)
        if not texts:
            st.warning("No se encontraron comentarios válidos. Revisa el formato.")
        else:
            with st.spinner(f"Analizando {len(texts)} comentarios..."):
                results = run_batch_analysis(texts, config=config)

            # Acumular resultados en la sesión
            for r in results:
                r["config"] = config
                st.session_state["session_results"].append(r)

            # Stats solo del batch actual
            stats = compute_sentiment_stats(results)

            st.markdown("### Resumen del batch actual")

            st.write(f"**Total comentarios:** {stats['total']}")
            st.write("**Counts:**")
            st.write(stats["counts"])

            st.write("**Distribution:**")
            dist_rows = [
                {"sentiment": label, "fraction": frac}
                for label, frac in stats["distribution"].items()
            ]
            if dist_rows:
                st.dataframe(dist_rows, use_container_width=True)

            st.markdown("### Resultados detallados (batch actual)")
            st.dataframe(
                [
                    {
                        "text": r["text"],
                        "sentiment": r["sentiment"],
                        "score": r["score"],
                        "short_reason": r["short_reason"],
                    }
                    for r in results
                ],
                use_container_width=True,
            )


# -------------------------------------------------------------------
# Resumen de la sesión completa
# -------------------------------------------------------------------

st.markdown("---")
st.markdown("### 🧾 Session summary (all analyses during this run)")

session_results: List[Dict[str, Any]] = st.session_state["session_results"]

if not session_results:
    st.info("Aún no se ha ejecutado ningún análisis en esta sesión.")
else:
    stats_session = compute_sentiment_stats(session_results)
    st.write(f"**Total comentarios en sesión:** {stats_session['total']}")
    st.write("**Counts:**")
    st.write(stats_session["counts"])
    st.write("**Distribution:**")
    dist_rows_session = [
        {"sentiment": label, "fraction": frac}
        for label, frac in stats_session["distribution"].items()
    ]
    st.dataframe(dist_rows_session, use_container_width=True)

    # Tabla breve de últimos resultados
    st.markdown("**Últimos resultados (máx 10):**")
    last_rows = session_results[-10:]
    st.dataframe(
        [
            {
                "text": r["text"],
                "sentiment": r["sentiment"],
                "score": r["score"],
                "config": r.get("config", ""),
            }
            for r in last_rows
        ],
        use_container_width=True,
    )

    if st.button("Limpiar resultados de sesión"):
        st.session_state["session_results"] = []
        st.rerun()
