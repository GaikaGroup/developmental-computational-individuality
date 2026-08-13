.PHONY: test pilot pilot-background analyze replication replication-background replication-analyze

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
