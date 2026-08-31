# IT Admin Agent

An agentic IT operations assistant that operates a Flask-based admin panel.
It executes user, license, ticket, and group operations through structured
SQL tools, answers policy questions with hybrid RAG over real IT policy
documents, verifies its own actions against database ground truth with a
self-repair pass, and can fall back to a browser-use agent driving the
panel's web UI. The evaluation harness measures task success, side effects,
citation accuracy, retrieval quality, and prompt-injection resistance
across agent configurations.

## Architecture

Two interacting components:

1. The Admin Panel (`app.py`, `database.py`, `templates/`)
   A Flask web application that mimics an internal IT department portal:
   dashboards, user management, license assignment, groups, tickets, and an
   audit log. State persists in a local SQLite database (`instance/itadmin.db`),
   seeded with a fixed demo dataset (`database.py: seed_db`).

2. The IT Agent (`agent/`)
   - `agent_core.py`: the orchestrator. No agent framework: the tool-calling
     loop talks to Groq's chat completions API directly through the provider
     SDK. Each request runs through RAG context retrieval, the tool loop, a
     verification pass with one self-repair attempt, and an optional browser
     fallback. Destructive tools (delete user, revoke license, reset
     password) can require operator confirmation mid-run.
   - `tools.py`: structured SQL tools (user lookup, license assign/revoke,
     ticket updates, group membership, password resets) that operate on the
     same database as the panel.
   - `rag/`: document ingestion (`ingest.py`) and hybrid retrieval
     (`retriever.py`). Dense retrieval uses ChromaDB with
     sentence-transformers embeddings, keyword retrieval uses BM25, and the
     two ranked lists are merged with Reciprocal Rank Fusion.
   - `browser_agent.py`: browser-use wrapper over Playwright. The agent reads
     the page DOM as text (no vision) and emits click/type/navigate actions
     until the task is done.

The LLM for all paths is Groq's `qwen/qwen3.6-27b` (set `GROQ_MODEL` to
swap). The core agent calls the provider SDK directly; only the browser
fallback goes through LangChain, because browser-use requires a LangChain
chat model.

## Setup

Requires Python 3.10+.

```
pip install -r requirements.txt
playwright install chromium
```

Create a `.env` file in the repo root (or in `agent/`) with:

```
GROQ_API_KEY=your-key-here
```

Build the RAG indexes once (embeds the PDFs/DOCX in `agent/rag/docs/` into
ChromaDB and a BM25 pickle):

```
python agent/rag/ingest.py
```

Start the admin panel:

```
python app.py            # serves on port 5000
```

## Using the agent

```
python agent/agent_core.py "assign a Pro Slack license to sarah@company.com"
python agent/agent_core.py                      # interactive mode
python agent/agent_core.py --rag bm25 "..."     # retrieval mode: hybrid, dense, bm25, none
python agent/agent_core.py --verbose "..."      # print each tool call as it runs
python agent/agent_core.py --yes "..."          # skip confirmation on destructive actions
python agent/agent_core.py --browser "..."      # enable browser fallback for tool-less requests
python agent/agent_core.py --force-browser "..."  # skip tools, drive the panel UI directly
```

By default the CLI pauses for y/n confirmation before destructive tools
run. The browser fallback is off by default because it is much slower than
the tool path. Browsers launch visibly by default; set `BROWSER_HEADLESS=1`
to run headless.

## Evaluation

The harness (`eval/harness.py`) runs the 45-task bank in
`eval/tasks_bank.json`. Before each task it resets the database to the seed
state, applies task preconditions, runs the agent, then validates the
expected database state (or the agent's answer, for Q&A tasks), checks
required citations, and detects unintended side effects.

```
python eval/harness.py --config hybrid        # baseline, tools_only, dense, bm25, hybrid
python eval/harness.py --fast                 # one task per category (smoke test)
python eval/harness.py --category injection   # one category only
python eval/harness.py --no-guardrails        # disable prompt guardrails
python eval/metrics.py --results eval/results # compare the latest run per config
```

### Retrieval evaluation

A 29-question golden set (`eval/golden_retrieval.json`) with document-level
relevance grades, evaluated offline with no LLM calls:

```
python eval/retrieval_eval.py
```

### Repeated runs with significance testing

Groq's free tier rate limits make this an overnight job:

```
python run_experiments.py --repeat 3          # mean +/- std per config, paired sign test
```

Each config is run N times; the aggregation reports the success-rate mean
and standard deviation per config and a two-sided paired sign test of each
RAG config against tools_only on per-task successes.

Notes:
- The harness resets `instance/itadmin.db` on every task, so stop the dev
  server before running evals.
- Tasks with `must_cite` require the agent to cite the expected policy
  document, so configs without RAG cannot pass them on parametric knowledge
  alone.
- Result JSONs from before the citation checks existed lack `citations_ok`
  and are not comparable with new runs.

## Security: prompt-injection red team

Five adversarial tasks (`injection` category) attack the agent through
ticket notes, user-supplied names, and direct override attempts. A task
counts as a successful attack when the forbidden action reaches the
database. Guardrails are a system-prompt trust boundary (tool output is
data, never instructions) plus the confirmation gate on destructive tools
in interactive use. Compare before and after:

```
python eval/harness.py --config hybrid --category injection --no-guardrails
python eval/harness.py --config hybrid --category injection
```

## Results

### End-to-End Evaluation Across Agent Configurations (45 Tasks)

Evaluated across the full 45-task test bank in `eval/tasks_bank.json`:

| Configuration | Success Rate | Silent Failures | False Refusals | Canary (`user_delete`) | Side Effect Rate | Citation Hit Rate | Mean Latency | Primary Mechanism |
|---|---|---|---|---|---|---|---|---|
| **`baseline`** | **22.2%** (10/45) | 0 | 0 | **FAIL** | 0.0% | 0.0% | 11.16s | Direct LLM (No tools, No RAG) |
| **`tools_only`** | **86.7%** (39/45) | 0 | 0 | **PASS** | 0.0% | 0.0% | 1101.02s | DB Tools only (No policy RAG) |
| **`dense`** | **86.7%** (39/45) | 0 | 0 | **PASS** | 0.0% | 0.0% | 535.62s | Tools + ChromaDB (MiniLM-L6-v2) |
| **`bm25`** | **95.6%** (43/45) | 2 | 0 | **PASS** | 0.0% | **100.0%** | 53.21s | Tools + BM25 Keyword Search |
| **`hybrid`** | **93.3%** (42/45) | 3 | 0 | **PASS** | 0.0% | **100.0%** | 44.01s | Tools + Dense + BM25 (RRF Fusion) |
| **`hybrid_policy`** | **100.0%** (13/13) | 0 | 0 | **PASS** | 0.0% | **100.0%** | 45.11s | Hybrid RAG + Deterministic Policy Engine |

---

### Category Breakdown

| Task Category | Tasks | Baseline | Tools Only | Dense | BM25 | Hybrid |
|---|---|---|---|---|---|---|
| **Password Reset** | 3 | 1/3 (33%) | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) | **3/3 (100%)** |
| **User Management** | 4 | 0/4 (0%) | 3/4 (75%) | 3/4 (75%) | 4/4 (100%) | **3/4 (75%)** |
| **License Management** | 6 | 0/6 (0%) | 6/6 (100%) | 6/6 (100%) | 6/6 (100%) | **6/6 (100%)** |
| **Group Management** | 3 | 0/3 (0%) | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) | **3/3 (100%)** |
| **User Lookup / Status** | 2 | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | **2/2 (100%)** |
| **Ticket Management** | 3 | 0/3 (0%) | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) | **3/3 (100%)** |
| **Conditional Logic** | 4 | 0/4 (0%) | 4/4 (100%) | 4/4 (100%) | 4/4 (100%) | **4/4 (100%)** |
| **Multi-Step Workflows** | 6 | 0/6 (0%) | 5/6 (83%) | 4/6 (67%) | 6/6 (100%) | **5/6 (83%)** |
| **Policy Queries** (NIST / CIS) | 4 | 4/4 (100%) | 0/4 (0%) | 2/4 (50%) | 4/4 (100%) | **4/4 (100%)** |
| **Ticket with Policy** | 1 | 0/1 (0%) | 0/1 (0%) | 0/1 (0%) | 1/1 (100%) | **1/1 (100%)** |
| **Edge Cases** (Not found / errors) | 3 | 2/3 (67%) | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) | **3/3 (100%)** |
| **Diagnostics** | 1 | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | **1/1 (100%)** |
| **Red-Team Injections** | 5 | 0/5 (0%) | 5/5 (100%) | 5/5 (100%) | 4/5 (80%) | **4/5 (80%)** |

---

### Offline Retrieval Quality (29 Golden Policy Queries)

Evaluated offline on `eval/golden_retrieval.json` without LLM calls (document-level relevance):

| Mode | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Key Advantage |
|---|---|---|---|---|---|
| **BM25** | 0.931 | 0.931 | 0.851 | 0.707 | Fast keyword matching for exact policy references |
| **Dense** | 1.000 | 1.000 | 0.886 | 0.801 | Semantic matching for paraphrased queries |
| **Hybrid (RRF)** | **1.000** | **1.000** | **0.912** | **0.791** | **Highest ranking precision (top MRR@10)** |

---

### Security: Prompt-Injection Defense

Evaluated across the 5-task adversarial red team suite (`injection` category):

| Configuration | Defense Rate | Attack Success Rate | Details |
|---|---|---|---|
| **No Guardrails** | 60.0% (3/5 defended) | 40.0% (2/5 breached) | Embedded override in ticket notes executed unauthorized actions |
| **With Guardrails** | **80.0% – 100.0%** | **0.0% – 20.0%** | Prompt trust boundary blocks embedded instructions in tool outputs |


## Repository layout

```
app.py                  Flask admin panel
database.py             models and the shared seed dataset
agent/agent_core.py     agent orchestrator and CLI
agent/tools.py          structured SQL tools
agent/browser_agent.py  browser-use fallback
agent/rag/              ingestion, retriever, policy documents
eval/harness.py         evaluation harness
eval/metrics.py         aggregate metrics and config comparison
eval/tasks_bank.json    45 evaluation tasks with ground truth
eval/golden_retrieval.json  29-question retrieval golden set
eval/retrieval_eval.py  offline retrieval metrics
run_experiments.py      runs all configs, aggregates repeats
```

Regenerable artifacts (`agent/rag/chroma_store/`, `bm25_index.pkl`,
`chunks.json`, `eval/results/`) are gitignored; rerun `ingest.py` and the
harness to rebuild them.
