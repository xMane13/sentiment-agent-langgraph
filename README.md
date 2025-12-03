# Sentiment & Feedback Agent with LangChain and LangGraph

This project implements an LLM-powered intelligent agent for **sentiment analysis and customer feedback support**, built with **LangChain**, **LangGraph**, and an open-source LLM served locally via **Ollama**.

The agent:

- Accepts comments in **Spanish or English**.
- Classifies sentiment as **positive**, **neutral**, or **negative**.
- Produces a **confidence score** in `[0, 1]`.
- Generates a short **reason** and a longer **explanation**.
- Suggests a **polite reply** as if it were a customer support assistant.
- Supports both **single-comment** and **batch** analysis, with basic statistics.

A **Streamlit UI** is included so that non-technical users can interact with the agent.

---

## 1. Project Overview

The goal of this project is to build a complete intelligent agent pipeline, not just a single model call. The system combines:

- **Prompt engineering** (few-shot, structured JSON output).
- **LangChain runnables** for modular sentiment → explanation → reply.
- **LangGraph** for routing, graph-based workflows and memory.
- **Open LLMs via Ollama** for fully local inference.
- **Evaluation scripts** to compare parameter configurations.
- A **Streamlit app** for interactive exploration.

---

## 2. Architecture

### 2.1 High-level flow

1. User provides a comment (or multiple comments).
2. A LangChain pipeline:
   - Classifies sentiment + score + short reason.
   - Generates a natural language explanation.
   - Generates a suggested reply.
3. A LangGraph workflow:
   - Routes between **single** and **batch** analysis.
   - Accumulates results in a shared state (`AgentState`).
   - Computes simple statistics across comments.
4. A Streamlit app:
   - Exposes single and batch modes.
   - Displays per-comment analysis and session summary.

### 2.2 Main components

- **LLM configuration**:  
  `src/models/llm_config.py`  
  Defines the open-source LLM served via Ollama and two parameter configurations:

  - **Config A (deterministic)**: `temperature=0.1, top_p=0.8, top_k=30`  
  - **Config B (creative)**: `temperature=0.7, top_p=0.95, top_k=50`

- **Sentiment chain**:  
  `src/chains/sentiment_chain.py`  
  A LangChain runnable that:
  - Reads a `user_text`.
  - Produces: `sentiment`, `score`, `short_reason`, `explanation`, `suggested_reply`.

- **LangGraph workflow**:  
  `src/graph/state.py`, `src/graph/nodes.py`, `src/graph/graph_builder.py`  
  Defines:
  - `router_node` → chooses single vs batch.
  - `single_analysis_node` and `batch_analysis_node`.
  - `stats_node` → computes aggregate statistics.
  - `final_output_node` → builds human-readable summaries.
  - `MemorySaver` → session-level memory via `thread_id`.

- **Tools**:  
  `src/tools/stats_tools.py`  
  Simple Python functions for:
  - `compute_sentiment_stats`
  - `compute_accuracy_with_labels`

- **Prompts**:  
  `prompts/`
  - `sentiment_prompt.txt`
  - `explanation_prompt.txt`
  - `reply_prompt.txt`

---

## 3. Repository Structure

```text
sentiment-agent-langgraph/
├── app_streamlit.py          # Streamlit UI
├── README.md
├── requirements.txt
├── data/
│   └── examples_raw.json     # Small labelled dataset (10 examples)
├── logs/                     # Evaluation logs (JSON) – usually gitignored
├── prompts/
│   ├── sentiment_prompt.txt
│   ├── explanation_prompt.txt
│   └── reply_prompt.txt
├── notebooks/                # Optional notebooks for experiments
└── src/
    ├── models/
    │   └── llm_config.py
    ├── chains/
    │   └── sentiment_chain.py
    ├── graph/
    │   ├── state.py
    │   ├── nodes.py
    │   └── graph_builder.py
    ├── tools/
    │   └── stats_tools.py
    ├── run_sentiment_demo.py
    ├── run_batch_demo.py
    ├── run_graph_demo.py
    └── run_eval_configs.py
```

---

## 4. Setup

### 4.1 Requirements

- Python **3.10+** (tested with 3.12)
- [Ollama](https://ollama.com/) installed and running
- A small open-source model pulled in Ollama (e.g. `tinyllama`)
- pip / virtualenv

### 4.2 Create and activate virtual environment

```bash
cd sentiment-agent-langgraph

python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows (PowerShell/CMD)
```

### 4.3 Install dependencies

```bash
pip install -r requirements.txt
```

### 4.4 Pull the LLM model in Ollama

By default, `llm_config.py` assumes a lightweight model such as `tinyllama`.  
Pull it with:

```bash
ollama pull tinyllama
```

If you want to use a different model, change `MODEL_NAME` in:

```python
# src/models/llm_config.py
MODEL_NAME = "tinyllama"
```

and pull the corresponding model in Ollama.

---

## 5. How to Run

### 5.1 CLI demo (single examples)

Run a small demo over the labelled dataset:

```bash
source .venv/bin/activate
python src/run_sentiment_demo.py
```

This script:
- Loads a few examples from `data/examples_raw.json`.
- Runs the sentiment agent (Config A by default).
- Prints the model’s outputs to the console.

### 5.2 CLI demo (batch mode)

```bash
python src/run_batch_demo.py
```

This script:
- Takes the same dataset.
- Processes all examples as a batch.
- Prints a table-like summary.

### 5.3 LangGraph demo

```bash
python src/run_graph_demo.py
```

This script:
- Builds the LangGraph workflow.
- Runs the graph multiple times with a fixed `thread_id`.
- Shows how `results` and `stats` accumulate across calls.

### 5.4 Streamlit UI

Launch the web interface:

```bash
streamlit run app_streamlit.py
```

Features:
- **Single comment** mode:
  - Paste a comment and get sentiment, score, explanation and suggested reply.
- **Batch analysis** mode:
  - Paste multiple comments (one per line or separated by `||`).
  - See class distribution and a table with per-comment results.
- **Session summary**:
  - Shows statistics across all analyses performed during the current run.

---

## 6. Evaluation

The script `src/run_eval_configs.py` compares **Config A** and **Config B** on the small labelled dataset:

```bash
python src/run_eval_configs.py
```

It:
- Loads `data/examples_raw.json` (10 labelled comments).
- Runs the agent with `config="A"` and `config="B"`.
- Saves logs under `logs/` as JSON files.
- Prints a summary for each configuration.

### 6.1 Current results

On the 10-example dataset:

- **Config A**
  - Accuracy: `0.90`
  - Distribution: `{ positive: 0.4, neutral: 0.3, negative: 0.3 }`
- **Config B**
  - Accuracy: `0.90`
  - Distribution: `{ positive: 0.4, neutral: 0.3, negative: 0.3 }`

This indicates that, on this small set, both configurations make the same classification decisions. The main differences appear in the **style** of explanations and replies: Config A is more stable and concise, while Config B is slightly more varied and verbose.

---

## 7. Design Highlights

- **Prompt engineering**:
  - Few-shot examples in the sentiment prompt.
  - Structured JSON output (`sentiment`, `score`, `short_reason`).
  - Separate prompts for explanation and reply.

- **LangChain**:
  - Uses the runnables API (`ChatPromptTemplate` + LLM + parsers).
  - The sentiment pipeline is a single reusable runnable used from:
    - CLI scripts.
    - LangGraph nodes.
    - Streamlit UI.

- **LangGraph**:
  - `router_node` selects single vs batch.
  - Analysis nodes call the shared sentiment pipeline.
  - `stats_node` and `final_output_node` build summaries.
  - `MemorySaver` provides session-level memory via `thread_id`.

- **Tools**:
  - Simple Python functions (no LLM) for statistics and accuracy.

---

## 8. Possible Extensions

Some ideas for future work:

- Use a larger labelled dataset for more robust evaluation.
- Add more fine-grained sentiment labels (e.g., *very positive*, *slightly negative*).
- Store results in a database and build dashboards for longitudinal analysis.
- Add more tools and router nodes (e.g., intent detection or topic clustering).
- Experiment with different open-source models in Ollama.

