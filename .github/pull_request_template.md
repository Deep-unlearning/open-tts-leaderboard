## New model / backend

**Model(s):** <!-- e.g. org/model-name, with a link to the model card -->
**Backend folder:** <!-- e.g. my-backend/ -->
**Docker Space (public):** <!-- e.g. https://huggingface.co/spaces/YOUR_USERNAME/evals-my-backend -->

### Model info (for `model_info.csv`)
- License:
- Voice cloning: yes / no
- Batch inference: yes / no
- Number of supported languages:
- Model size (B params):
- Runs with `transformers`: yes / no

### Evaluated splits
<!-- e.g. Seed-TTS tts/en, tts/zh, CV3-Eval zero_shot/{en,zh,...}; with or without voice cloning -->

### Results
<!-- Paste the "Results per dataset" / "Composite Results" block printed at the end of submit_jobs.sh -->

```
```

### Checklist
- [ ] `run_eval.py` and `submit_jobs.sh` follow the structure of the existing backends (see README, "Adding a new model")
- [ ] Stage 1 runs on the default `h200` flavor (`FLAVOR` is not overridden)
- [ ] Generation time excludes model loading / warm-up
- [ ] Smoke test passed (`MAX_EVAL_SAMPLES=8 ONLY_LANGS="en" bash <backend>/submit_jobs.sh`)
- [ ] Full evaluation completed and results are in the bucket
- [ ] (Optional) `--ttfa_probe` implemented and target added to `submit_ttfa_jobs.sh`
