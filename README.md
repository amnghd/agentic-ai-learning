# Agentic AI Learning

A hands-on curriculum for building production-ready agentic AI systems — covering SLM fine-tuning, RAG, multi-agent frameworks, observability, evaluation, and production deployment.

**Duration:** 6–8 weeks | **Level:** Advanced | **Prerequisites:** Python, ML fundamentals, NLP basics

---

## Curriculum at a Glance

| Week | Topic | Key Technologies |
|------|-------|-----------------|
| 1 | Foundation & Environment Setup | Docker, GitHub Actions, pre-commit, Poetry |
| 2 | SLM Fine-Tuning | Hugging Face, LoRA/QLoRA, PEFT, W&B |
| 3 | Retrieval Augmented Generation (RAG) | Pinecone/Qdrant, LangChain, RAGAS |
| 4 | Multi-Agent Systems — AutoGen | AutoGen, OpenAI/Anthropic APIs |
| 5 | Workflow Orchestration — LangGraph & LangSmith | LangGraph, LangSmith, OpenTelemetry |
| 6 | Evaluation & Experimentation — Braintrust | Braintrust, A/B testing |
| 7 | Integration & Production Deployment | FastAPI, Kubernetes, Prometheus/Grafana |
| 8 | Advanced Patterns & Capstone | Memory systems, planning agents |

---

## Repository Structure

```
agentic-ai-learning/
├── Projects/
│   ├── README.md                  # Full curriculum overview
│   ├── CURRICULUM_OVERVIEW.md     # Learning objectives & assessment
│   ├── PROJECT_STRUCTURE.md       # Project layout guide
│   ├── QUICK_START.md             # Getting started fast
│   ├── STARTER_TEMPLATES.md       # Code templates
│   ├── week1/                     # Foundation & Environment Setup
│   ├── week2/                     # SLM Fine-Tuning
│   ├── week3/                     # RAG Systems
│   ├── week4/                     # AutoGen
│   ├── week5/                     # LangGraph & LangSmith
│   ├── week6/                     # Braintrust
│   ├── week7/                     # Integration & Deployment
│   └── week8/                     # Advanced Topics & Capstone
└── document/
    └── Designing Agentic Customer Support Systems.pdf
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Git

### Week 1 Setup

```bash
git clone https://github.com/amnghd/agentic-ai-learning.git
cd agentic-ai-learning/Projects/week1
bash scripts/setup.sh
```

This creates a virtual environment, installs all dependencies, and installs pre-commit hooks.

### Running with Docker

```bash
cd Projects/week1
docker compose -f docker/docker-compose.yml up
# Jupyter available at http://localhost:8888
```

---

## Weekly Projects

Each `weekN/` directory contains:
- `README.md` — objectives, tasks, deliverables, and evaluation criteria
- `src/` — source code
- `tests/` — test suite
- `scripts/` — setup and utility scripts
- `docker/` — containerization config (where applicable)
- `notebooks/` — Jupyter notebooks

---

## Assessment

| Component | Weight |
|-----------|--------|
| Weekly project submissions | 30% |
| Capstone project | 60% |
| Participation & collaboration | 10% |

---

## Core Frameworks

- **AutoGen** — multi-agent conversations
- **LangGraph** — state machine workflows
- **LangSmith** — observability and debugging
- **Braintrust** — evaluation and A/B testing
- **Hugging Face** — models and datasets
- **Weights & Biases / MLflow** — experiment tracking
