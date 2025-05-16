# ScamShield‑AI

> **An AI‑powered anti‑scam assistant that harnesses Retrieval‑Augmented Generation (RAG) with state‑of‑the‑art large language models to detect, analyze, and respond to scam‑related content in both Chinese and English.**
>
> *LoRA or other parameter‑efficient fine‑tuning is **not** enabled yet, but the roadmap includes optional support once the local‑model pipeline is stable.*

## Authors

* [Jerry Hung](https://github.com/JerryHung1030)
* [Ken Su](https://github.com/ken-su)
* [SJ](https://github.com/sj)

ScamShield‑AI ingests **real‑world fraud cases** (currently focused on Taiwan) and a growing set of regulatory references, turns them into vector embeddings with Qdrant, and leverages GPT‑4o —or a local Llama model— to provide judgment calls, evidence highlighting, and confidence scores.

⚠️ **Project status: *alpha preview* — APIs may break during the upcoming refactors.** See the new [Development Status & Roadmap](#🚧-development-status--roadmap) section for details.

---

## Table of Contents

1. [✨ Features](#features)
2. [🗺️ Architecture Overview](#architecture-overview)
3. [📂 Project Structure](#project-structure)
4. [🚀 Quick Start](#quick-start)
5. [⚙️ Configuration](#configuration)
6. [🛠️ CLI & API Usage](#cli--api-usage)
7. [🧪 Testing](#testing)
8. [🤝 Contributing](#contributing)
9. [📄 License](#license)

---

## ✨ Features

* **End‑to‑end RAG pipeline** powered by **OpenAI GPT‑4o** (default) or **local Llama** adapter.
* **Hierarchical JSON ingestion** (`level1 → level5`) with automatic flattening, optional chunk‑splitting, and schema validation.
* **Pluggable embeddings** via `langchain-openai` (defaults to `text‑embedding‑ada‑002`).
* **Vector store abstraction** built on **Qdrant**, supporting upsert, search w/ metadata filters, and async operations.
* **PromptBuilder** that fits the entire query + candidates within a configurable token budget and gracefully backs off.
* **ResultFormatter** that parses the LLM JSON output, merges similarity scores, filters by confidence, and normalizes direction (forward / reverse / both).
* **Job orchestration** with **Redis‑RQ** (batch ingest + RAG jobs) and a minimal **FastAPI** facade.
* **Rich logging** via **loguru** with daily rotation.
* **Extensive unit tests** (PyTest) with mocks for fast, cost‑free CI.

---

## 🗺️ Architecture Overview

```text
┌──────────┐     ┌───────────────┐     ┌──────────┐
│  Input   │     │   Embedding   │     │  Qdrant  │
│ JSON(s)  │──►──│  Manager      │──►──│ Vector DB│
└──────────┘     └───────────────┘     └──────────┘
      │                                   ▲
      ▼                                   │
┌────────────┐  search(top‑k)  ┌──────────┴──────┐
│  RAGEngine │───────────────►│ PromptBuilder   │
└─────┬──────┘                 └────────┬───────┘
      │   LLM request / stream          │prompt
      ▼                                 ▼
┌────────────┐                   ┌──────────────┐
│  LLMProxy  │◄── GPT‑4o / Llama │ ResultFormat │
└────────────┘                   └──────┬───────┘
      │                                   │
      ▼                                   ▼
┌──────────────┐                   ┌──────────────┐
│  RAG Result  │◄──────────────────│    Client    │
└──────────────┘                   └──────────────┘
```

---

## 📂 Project Structure

```text
├── src/
│   ├── rag_core/            # Core RAG logic (domain, application, infrastructure)
│   │   ├── domain/          # Pydantic schemas, validation, exceptions
│   │   ├── application/     # PromptBuilder, RAGEngine, ResultFormatter
│   │   └── infrastructure/  # Embeddings, VectorStore, LLM adapters
│   ├── interfaces/
│   │   ├── cli_main.py      # One‑shot CLI pipeline
│   │   ├── jobs/            # RQ Job runner
│   │   └── api/             # FastAPI facade
│   ├── data/                # Example hierarchical JSON datasets
│   └── config/              # YAML + pydantic‑settings
├── tests/                   # PyTest suites with extensive mocks
├── requirements.txt         # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Prerequisites

* **Python ≥ 3.12**
* **Docker** (for Qdrant & Redis) *or* native installations
* An **OpenAI API key** (required for GPT-4o)

### 2. Environment Setup

```bash
# set OpenAI API Key
$ export OPENAI_API_KEY="your-api-key-here"

# Or, you can create a .env file
$ echo "OPENAI_API_KEY=your-api-key-here" > .env
```

> ⚠️ **Important Note**: Make sure to set up your API Key before running any commands, otherwise the program will not function properly.

### 3. Clone & Install

```bash
# 1⃣  Clone
$ git clone https://github.com/your-org/scamshield-ai.git
$ cd scamshield-ai

# 2⃣  Create venv
$ python -m venv .venv && source .venv/bin/activate

# 3⃣  Install deps
$ pip install -r requirements.txt
```

### 4. Spin up services

```bash
# Qdrant
$ docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# Redis (for RQ)
$ docker run -d --name redis -p 6379:6379 redis:7
```

### 5. Run the **demo CLI**

```bash
$ python -m src.interfaces.cli_main
```

This will ingest the sample `scam_input.json` & `scam_references.json`, execute a reverse‑direction RAG, and write results to `src/interfaces/output/rag_result.json`.

### 6. Launch the **FastAPI** endpoint (optional)

```bash
$ uvicorn src.interfaces.api.fastapi_app:app --reload --port 8000
```

---

## ⚙️ Configuration

ScamShield-AI uses a layered configuration system that combines environment variables, YAML files, and Pydantic settings for maximum flexibility.

### Environment Variables

The following environment variables can be used to override default settings:

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | - | Yes |
| `LLM_MODEL` | LLM model to use | `gpt-4o` | No |
| `QDRANT_URL` | Qdrant server URL | `http://localhost:6333` | No |
| `QDRANT_COLLECTION` | Qdrant collection name | `dev_rag_collection` | No |
| `IS_DEBUG` | Enable debug mode | `false` | No |

### Configuration Files

The system uses two types of configuration files:

1. **Base Configuration** (`src/config/settings.base.yml`):
   - Contains default settings for all environments
   - Version controlled
   - Should not contain sensitive information

2. **Local Configuration** (`settings.local.yml`):
   - Overrides base settings for local development
   - Not version controlled
   - Can contain environment-specific settings

Example base configuration:

```yaml
system:
  is_debug: true
  log_dir: logs
  log_file_path: app.log

llm:
  model: gpt-4o
  temperature: 0.7
  max_tokens: 4096
  max_prompt_tokens: 8000

vector_db:
  url: http://localhost:6333
  collection: dev_rag_collection
  vector_size: 1536

scenario:
  direction: reverse        # forward / reverse / both
  rag_k_forward: 5
  rag_k_reverse: 20
  cof_threshold: 0.6
  reference_json: "src/data/scam_references.json"
  input_json: "src/data/scam_input.json"
```

### Configuration Priority

Settings are applied in the following order (highest to lowest priority):

1. Environment variables
2. Local configuration file (`settings.local.yml`)
3. Base configuration file (`settings.base.yml`)
4. Default values in code

> **Security Note**: Always store sensitive information (API keys, credentials) in environment variables or `.env` files, never in configuration files.

---

## 🛠️ CLI & API Usage

### CLI

```bash
# Forward direction (input → reference)
$ python -m src.interfaces.cli_main --run-id run_1 \
      --direction forward
```

### Programmatic (FastAPI + RQ)

```python
from src.interfaces.api.fastapi_app import start_rag_job, RAGJobRunner
from src.rag_core.infrastructure import setup_core

embed_mgr, vec_index, llm_mgr = setup_core()
runner = RAGJobRunner(vec_index, RAGEngine(embed_mgr, vec_index, llm_mgr))

job_id = start_rag_job(
    job_runner=runner,
    project_id="demo_proj",
    scenario={"direction": "reverse", "rag_k_reverse": 20},
    input_json_path="/path/input.json",
    reference_json_path="/path/ref.json",
    callback_url="https://your.backend/api/rag_callback"
)
print("Enqueued RAG job:", job_id)
```

---

## 🧪 Testing

```bash
$ pytest -q
```

The test suite mocks **OpenAI**, **Qdrant**, and **Embeddings** to provide fast, deterministic results (< 5 s on a laptop).

---

## 🚧 Development Status & Roadmap

> **Status — Alpha preview.** The fundamental RAG loop is stable, but the orchestration layer is being redesigned.

### ✅ Completed so far

* Core `rag_core` engine (embedding → search → prompt → LLM → result) proven via CLI demo.
* Hierarchical JSON ingestion, schema validation, and optional chunk‑splitting.
* Qdrant vector store adapter with async helpers.
* OpenAI GPT‑4o & local Llama adapters.
* Initial Redis‑RQ + FastAPI proof‑of‑concept.
* CI‑ready PyTest suite with heavy mocking.

### 🛠️ In progress

* **Orchestration decoupling** – Redis/RQ dependencies will be extracted from `rag_core`; the job queue becomes an *optional* outer service.
* **Packaging** – publish `rag_core` as a standalone `pip install scamshield-rag` so any backend can import and wire up its own queue.
* **Scenario templates** – ship more scoring rules, prompt recipes, and evaluation scripts.

### 🗓️ Planned / help wanted

* Reference implementation of a lightweight `FastAPI` + `StarterQueue` repo demonstrating how to plug in your own Redis, Sidekiq, or Celery workers.
* Async streaming helpers (WebSocket / SSE) for real‑time UIs.
* Multilingual (JP/KR/EN) scam datasets & regulatory corpora.
* Memory‑aware chunking + performance benchmarks.
* **Local fine‑tuning workflow** — LoRA / QLoRA recipes for Llama‑3 or Phi‑3 to run completely offline.

---

## 🤝 Contributing

1. Fork → feature branch → Pull Request.
2. Make sure `pytest` passes and `pre‑commit run --all-files` shows no lint errors.
3. Add / update docs where applicable.

We welcome new **datasets**, **scenario templates**, and **LLM adapters**!

---

## 📄 License

Proprietary Software License Agreement

Copyright (c) 2024 Institute for Information Industry (III), Cyber Security Technology Institute (CSTI)

All rights reserved. This software is proprietary and confidential. Unauthorized copying, modification, distribution, or use is strictly prohibited.

> © 2024 Institute for Information Industry (III), Cyber Security Technology Institute (CSTI). Built with ❤️ in Taiwan.
