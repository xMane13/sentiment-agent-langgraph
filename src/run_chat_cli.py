# src/run_chat_cli.py

from __future__ import annotations

from graph.graph_builder import build_agent_graph


def main():
    app = build_agent_graph()

    print("=" * 80)
    print(" Sentiment & Feedback Agent (CLI)")
    print(" Usa LangGraph + gemma3:1b")
    print(" Comandos:")
    print("   - Texto normal  => análisis de 1 comentario")
    print("   - 'batch: ...'  => análisis de varios usando '||' como separador")
    print("   - 'exit' o 'quit' => salir")
    print("=" * 80)

    # Un thread_id fijo para que la memoria (results) se acumule en esta sesión
    thread_id = "cli-session-1"

    while True:
        try:
            user_input = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nSaliendo...")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "salir"}:
            print("Bye!")
            break

        # Invocamos el grafo
        state = app.invoke(
            {"user_input": user_input},
            config={"configurable": {"thread_id": thread_id}},
        )

        final_output = state.get("final_output", "")
        print("\nAgent>\n" + final_output)


if __name__ == "__main__":
    main()
