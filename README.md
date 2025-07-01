# rag-core

> **A production-ready Retrieval-Augmented Generation service** that **schedules**, **stores**, and **serves** document-aware intelligence through a single set of high-level Web APIs—built to power multiple downstream products such as **Fraudlens** and **Relulens-AI**.

---

## Authors

* [Jerry Hung](https://github.com/JerryHung1030)
* [Ken Su](https://github.com/ken22i)
* [SJ](https://github.com/shih1999)

---

## Table of Contents

1. [✨ Features](#-features)
2. [🗺️ Architecture Overview](#-architecture-overview)
3. [📂 Project Structure](#-project-structure)
4. [🚀 Quick Start](#-quick-start)
5. [⚙️ Configuration](#-configuration)
6. [🛠️ API Usage](#-api-usage)
7. [🚧 Development Status & Roadmap](#-development-status--roadmap)
8. [🤝 Contributing](#-contributing)
9. [📄 License](#-license)

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
rag-core/
├── src/                              # Python source root
│   ├── rag_core/                     # Core: Domain / Application / Infrastructure
│   │   ├── domain/                   #   • Business objects and Pydantic models
│   │   ├── application/              #   • Use-cases: PromptBuilder, RAGEngine, etc.
│   │   ├── infrastructure/           #   • Adapters: Embeddings, VectorStore, LLM
│   │   └── utils/                    #   • Shared utilities (token counter, blacklist, etc.)
│   ├── interfaces/                   # Entry layer: CLI, API, Job Runner
│   │   ├── api/                      #   • FastAPI web layer
│   │   ├── jobs/                     #   • Redis-RQ job executor
│   │   └── cli_main.py               #   • One-off CLI pipeline
│   ├── config/                       # Configuration: YAML + Pydantic Settings
│   └── data/                         # Example datasets (demo JSON)
│
├── tests/                            # PyTest suites (heavy use of mocks, fast CI)
│   └── …                             #   • Grouped by layer: domain / application / infra
│
├── docker-compose.yml                # One-click launch for Qdrant, Redis, API
├── Dockerfile                        # Build runtime image
├── requirements.txt                  # Python dependencies
├── README.md                         # This document
└── LICENSE                           # License
```

---

## 🚀 Quick Start

### 0. Run with **Docker Compose** (Recommended)

**The fastest way to get started.** This will install Docker, verify the installation, and launch all services (Qdrant, Redis, API) with a single command.

1. **Install Docker**

Download the official installation script:
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
```
Run the installation script:
```bash
sudo sh get-docker.sh
```
Verify Docker installation:
```bash
docker --version
sudo docker run hello-world
```

2. **Set your OpenAI API Key**

Make sure to set your `.env` file or the `OPENAI_API_KEY` environment variable before starting the services, otherwise the application will not work.

3. **Launch all services**

```bash
docker compose up --build
```

The API will be available at [http://localhost:8000](http://localhost:8000) by default.

---

## ⚙️ Configuration

fraudlens-rag-core uses a layered configuration system that combines environment variables, YAML files, and Pydantic settings for maximum flexibility.

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

## 🛠️ API Usage

#### Main Endpoints

| Name | Method | Route | Function |
|:-:|:-:|:-:|:-:|
| submit_rag_job | POST | `/api/v1/rag` | Submit a new RAG job |
| get_rag_job_status | GET | `/api/v1/rag/{job_id}/status` | Query the status of a specified job |
| get_rag_job_result | GET | `/api/v1/rag/{job_id}/result` | Retrieve the result of a specified job |
| list_rag_jobs | GET | `/api/v1/rag` | List all jobs (optionally filtered by project) |
| delete_rag_job | DELETE | `/api/v1/rag/{job_id}` | Delete a specified job |

#### Notes
For detailed API specifications, please refer to [RAGCore-X_api.xlsx](docs/RAGCore-X_api.xlsx).

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
* **Scenario templates** – ship more scoring rules, prompt recipes, and evaluation scripts.

### 🗓️ Planned / help wanted

* Reference implementation of a lightweight `FastAPI` + `StarterQueue` repo demonstrating how to plug in your own Redis, Sidekiq, or Celery workers.
* Async streaming helpers (WebSocket / SSE) for real‑time UIs.
* Multilingual (JP/KR/EN) scam datasets & regulatory corpora.
* Memory‑aware chunking + performance benchmarks.

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

