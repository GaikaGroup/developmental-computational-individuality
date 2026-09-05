.PHONY: test smoke analyze plot

test:
	PYTHONPATH=src pytest -q tests

smoke:
	PYTHONPATH=src python -m dl_moe.smoke

analyze:
	PYTHONPATH=src python -m dl_moe.analysis results/smoke

plot:
	PYTHONPATH=src python -m dl_moe.plots results/smoke results/smoke/specialization.png
