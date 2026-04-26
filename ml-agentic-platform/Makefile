.PHONY: install lint type-check test run train-federated train-graph-fraud

install:
	pip install -e .[dev]

lint:
	ruff check src tests

type-check:
	mypy src

test:
	pytest

run:
	uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

## Federated learning — runs Flower tabular simulation on IEEE-CIS data.
## Override any option via make variable, e.g.:  make train-federated ROUNDS=3 MODEL=nn
FEATURES_DIR ?= data/features/ieee_cis_fraud
PROCESSED_DIR ?= data/processed/ieee_cis_fraud
FL_OUTPUT_DIR ?= artifacts/federated_learning
NUM_CLIENTS ?= 3
ROUNDS ?= 5
MODEL ?= logistic
train-federated:
	fl-tabular-sim \
		--features-dir $(FEATURES_DIR) \
		--processed-dir $(PROCESSED_DIR) \
		--output-dir $(FL_OUTPUT_DIR) \
		--num-clients $(NUM_CLIENTS) \
		--rounds $(ROUNDS) \
		--model-type $(MODEL)

## Graph fraud — trains Temporal GNN on IEEE-CIS data.
## Override any option via make variable, e.g.:  make train-graph-fraud EPOCHS=5 BACKBONE=gcn
IEEE_CSV ?= data/processed/ieee_cis_fraud/train_processed.csv
GF_OUTPUT_DIR ?= artifacts/temporal_gnn
BACKBONE ?= gat
EPOCHS ?= 80
train-graph-fraud:
	train-graph-fraud \
		--dataset ieee_cis \
		--ieee-csv-path $(IEEE_CSV) \
		--output-dir $(GF_OUTPUT_DIR) \
		--backbone $(BACKBONE) \
		--epochs $(EPOCHS)
