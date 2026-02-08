.PHONY: run run-skip labels ui train resonance random

run: ; TRACE_VERBOSE=true python pipeline/run_main.py

run-skip: ; TRACE_VERBOSE=true RUN_PROFILES=false BUILD_EMBEDDINGS=false python pipeline/run_main.py

labels: ; TRACE_VERBOSE=true RUN_LABELS=true RUN_PROFILES=false BUILD_EMBEDDINGS=false python pipeline/run_main.py

ui: ; uvicorn profiling.label_ui.app:app --reload

train: ; TRACE_VERBOSE=true RUN_TRAIN=true RUN_PROFILES=false BUILD_EMBEDDINGS=false python pipeline/run_main.py

resonance: ; TRACE_VERBOSE=true RUN_RESONANCE_TEST=true RUN_PROFILES=false BUILD_EMBEDDINGS=false python pipeline/run_main.py

random: ; TRACE_VERBOSE=true PICK_RANDOM_TEST=true RUN_PROFILES=false BUILD_EMBEDDINGS=false python pipeline/run_main.py
