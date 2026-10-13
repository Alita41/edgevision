# Evaluation protocol and results

## What is actually verified here

- Regression tests validate region proposals, IoU tracking/confirmation, unknown suppression, event retention, API parameters, and quantized tensor conversion.
- A deterministic synthetic smoke scene validates the entire application path, with a moving red sign-shaped object. This is an architecture smoke test, **not accuracy evidence**.
- A local synthetic **computational** benchmark reports processing latency/throughput on the current CPU. This is *not* a Raspberry Pi figure, real-scene model accuracy, or real-time traffic-sign recognition evidence.

## Model classification evaluation

`train_gtsrb.py` trains a small CNN from *user-provided* labeled GTSRB sign crops and optionally exports a calibrated int8 TensorFlow Lite model. Without a supplied separately labeled official GTSRB test set, its stratified crop holdout is an internal validation result and may be optimistic because GTSRB includes correlated sequences. Prefer using the official Test.csv.

The script produces classification accuracy, macro F1, full per-class classification report, confusion matrix, and expected calibration error (ECE). It **does not** calculate detection mAP from crop-classification data; doing so would be scientifically invalid.

## Detection metrics on scene frames

Supply scene-level bounding-box ground truth and predictions as JSONL:

```json
{"frame_id":"video1:000123", "label":"stop", "box":[120,75,180,145]}
```

Predictions additionally have `confidence` and `abstained`, if applicable. Then run:

```bash
python scripts/evaluate_annotations.py \
  --ground-truth path/to/annotations.jsonl \
  --predictions path/to/predictions.jsonl \
  --out artifacts/scene-eval.json
```

The report includes localization-aware micro precision, recall, F1, per-class TP/FP/FN, and abstention rate at IoU 0.5. This is intentionally **not** presented as COCO mAP; the supplied implementation does not compute full AP curves.

## Edge-device performance

On the actual Raspberry Pi or target edge device, start the server with the camera and trained model, then:

```bash
python scripts/benchmark_device.py --url http://127.0.0.1:8080 --seconds 60 \
  --out artifacts/pi-measurement.json
```

Record CPU, OS, source resolution, model hash, backend, p50/p95 processing time, processed FPS, dropped-frame count, power profile, sustained thermal conditions, and frame-to-event latency. No device data are bundled in this repository.

## Next research milestone

A realistic ablation study would compare single-frame vs multi-frame confirmation, calibrated vs uncalibrated thresholds, FP32 vs INT8, contour proposals vs a learned detector, and daytime vs dusk/rain/night slices. Report false alerts/minute, per-class precision/recall, abstention, and processing latency on **held-out** scenes.
