# ML Agentic Platform

Production-ready modular scaffold for biometric fraud detection with multi-agent orchestration.

## Core Modules
- Data Engineering
- Biometric Agent
- Graph Fraud Agent
- Federated Learning Agent
- RAG Agent
- Multi-Agent Orchestration using LangGraph
- API Layer (FastAPI)
- MLOps (tracking, registry, monitoring)

## Project Layout
```text
ml-agentic-platform/
  src/app/
    api/
    core/
    data_engineering/
    agents/
    orchestration/
    mlops/
    schemas/
    services/
  configs/
  infra/
  tests/
  scripts/
```

## Quick Start
1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:
   ```bash
   pip install -e .[dev]
   ```
3. Copy env:
   ```bash
   copy .env.example .env
   ```
4. Run API:
   ```bash
   uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Development Commands
- Lint: `ruff check src tests`
- Type check: `mypy src`
- Tests: `pytest`
