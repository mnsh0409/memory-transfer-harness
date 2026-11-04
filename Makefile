.PHONY: test smoke figures
PYTHON ?= python3
test:
	$(PYTHON) -m pytest tests/ -q
smoke:
	$(PYTHON) -m pytest tests/test_harness.py::test_replay_one_code_path_all_baselines -q
figures:
	$(PYTHON) -m scripts.collect_stats && $(PYTHON) -m scripts.make_figures
