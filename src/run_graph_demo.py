# src/run_graph_demo.py

from __future__ import annotations

from graph.graph_builder import build_agent_graph


def demo_single(app):
    print("\n" + "=" * 80)
    print("DEMO: SINGLE MODE\n")

    state = app.invoke(
        {"user_input": "El producto llegó rápido y en perfectas condiciones. Muy satisfecho."},
        config={"configurable": {"thread_id": "session-1"}},
    )

    print(state["final_output"])
    print("\nEstado interno (keys):", list(state.keys()))


def demo_batch(app):
    print("\n" + "=" * 80)
    print("DEMO: BATCH MODE\n")

    # Formato: batch: texto1 || texto2 || texto3
    user_input = (
        "batch: "
        "El envío llegó con una semana de retraso y nadie respondió mis correos. || "
        "Amazing quality! I will definitely buy again. || "
        "El producto está bien, pero nada especial."
    )

    state = app.invoke(
        {"user_input": user_input},
        config={"configurable": {"thread_id": "session-2"}},
    )

    print(state["final_output"])
    print("\nEstado interno (keys):", list(state.keys()))


def demo_memory(app):
    print("\n" + "=" * 80)
    print("DEMO: MEMORY / MULTIPLE CALLS IN SAME THREAD\n")

    # Primera llamada en session-3
    s1 = app.invoke(
        {"user_input": "El servicio fue pésimo, nunca más compro aquí."},
        config={"configurable": {"thread_id": "session-3"}},
    )
    print("Primera llamada (session-3):")
    print(s1["final_output"])
    print(f"\nResultados acumulados: {len(s1.get('results', []))}")

    # Segunda llamada en la MISMA session-3
    s2 = app.invoke(
        {"user_input": "El producto en sí estaba bien, pero el envío fue lento."},
        config={"configurable": {"thread_id": "session-3"}},
    )
    print("\nSegunda llamada (session-3):")
    print(s2["final_output"])
    print(f"\nResultados acumulados: {len(s2.get('results', []))}")

    # Observa que 'results' sigue creciendo gracias al checkpointer.


def main():
    app = build_agent_graph()

    demo_single(app)
    demo_batch(app)
    demo_memory(app)


if __name__ == "__main__":
    main()
