.PHONY: test pilot pilot-background analyze

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
