.PHONY: run run-skip labels ui train resonance random

run: ; python -m pipeline.run_main

run-skip: ; RUN_PROFILES=false BUILD_EMBEDDINGS=false python -m pipeline.run_main

labels: ; RUN_LABELS=true RUN_PROFILES=false BUILD_EMBEDDINGS=false python -m pipeline.run_main

ui: ; uvicorn profiling.label_ui.app:app --reload

dashboard: ; uvicorn resonance.dashboard.app:app --reload

train: ; RUN_TRAIN=true RUN_PROFILES=false BUILD_EMBEDDINGS=false python -m pipeline.run_main

resonance: ; RUN_RESONANCE_TEST=true RUN_PROFILES=false BUILD_EMBEDDINGS=false python -m pipeline.run_main

random: ; PICK_RANDOM_TEST=true RUN_PROFILES=false BUILD_EMBEDDINGS=false python -m pipeline.run_main
