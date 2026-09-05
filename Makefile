.PHONY: test pilot pilot-background analyze replication replication-background replication-analyze compare-experiments

test:
	.venv/bin/python -m pytest -q

pilot:
	.venv/bin/python -m dci_pilot.sweep --seeds 0:19
	.venv/bin/python -m dci_pilot.analyze

pilot-background:
	mkdir -p results/logs
	nohup sh -c '.venv/bin/python -u -m dci_pilot.sweep --seeds 0:19 && .venv/bin/python -u -m dci_pilot.analyze' > results/logs/sweep.log 2>&1 & echo $$!

analyze:
	.venv/bin/python -m dci_pilot.analyze

replication:
	.venv/bin/python -m pytest -q
	.venv/bin/python -m dci_pilot.sweep --seeds 100:119 --config replication_config.yaml --output results/replication
	.venv/bin/python -m dci_pilot.checkpoint_ablation --checkpoints results/replication/raw --output results/replication/checkpoint_analysis --config replication_config.yaml --context replication
	.venv/bin/python -m dci_pilot.replication_analysis

replication-background:
	mkdir -p results/replication/logs
	nohup sh -c '.venv/bin/python -u -m dci_pilot.sweep --seeds 100:119 --config replication_config.yaml --output results/replication && .venv/bin/python -u -m dci_pilot.checkpoint_ablation --checkpoints results/replication/raw --output results/replication/checkpoint_analysis --config replication_config.yaml --context replication && .venv/bin/python -u -m dci_pilot.replication_analysis' > results/replication/logs/replication.log 2>&1 & echo $$!

replication-analyze:
	.venv/bin/python -m dci_pilot.replication_analysis

compare-experiments:
	.venv/bin/python -m pytest -q
	.venv/bin/python -m dci_pilot.compare_experiments

.PHONY: ci ci-build dl-moe-test dl-moe-validate

# Separate pytest processes prevent equal test module names in the two packages colliding.
ci: test dl-moe-test dl-moe-validate ci-build
	.venv/bin/python -m compileall -q src dl_moe/src

dl-moe-test:
	PYTHONPATH=dl_moe/src .venv/bin/python -m pytest -q dl_moe/tests

dl-moe-validate:
	PYTHONPATH=dl_moe/src .venv/bin/python -m dl_moe.experiments.dl_moe_01.validate_preregistration
	PYTHONPATH=dl_moe/src .venv/bin/python -m dl_moe.experiments.dl_moe_01.run --dry-run

ci-build:
	.venv/bin/python -m pip wheel --no-deps --no-build-isolation --no-index ./dl_moe --wheel-dir results/validation/wheels
