.PHONY: run run-skip labels ui train resonance random dashboard dashboard-demo env

env:
	@bash -lc 'source scripts/activate_env.sh && exec $$SHELL -l'
run: ; python -m pipeline.run_main

run-skip: ; RUN_PROFILES=false BUILD_EMBEDDINGS=false python -m pipeline.run_main

embed: ; BUILD_EMBEDDINGS=true RUN_PROFILES=false python -m pipeline.run_main

labels: ; RUN_LABELS=true RUN_PROFILES=false BUILD_EMBEDDINGS=false python -m pipeline.run_main

ui: ; uvicorn profiling.label_ui.app:app --reload

dashboard: ; uvicorn resonance.dashboard.app:app --reload
dashboard-demo: ; DEMO_MODE=true RESONANCE_CACHE_PATH=data/demo/resonance_cache.json uvicorn resonance.dashboard.app:app --reload

train: ; RUN_TRAIN=true RUN_PROFILES=false BUILD_EMBEDDINGS=false python -m pipeline.run_main

resonance: ; RUN_RESONANCE_TEST=true RUN_PROFILES=false BUILD_EMBEDDINGS=false RESONANCE_WRITE_CACHE=true RESONANCE_CACHE_PATH=data/demo/resonance_cache.json python -m pipeline.run_main

random: ; PICK_RANDOM_TEST=true RUN_PROFILES=false BUILD_EMBEDDINGS=false python -m pipeline.run_main
