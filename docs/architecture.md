# Architecture

EdgeVision is an edge-first **perception + confirmation + observation** pipeline. It is designed to be inspectable, instrumented, and runnable without external APIs.

```text
     Camera / video / synthetic source
                  │
             capture thread
                  │
          bounded queue (size 2)
           drop oldest on overflow
                  │
             worker thread
                  │
       HSV + contour region proposals
                  │
        ┌─────────┴─────────┐
        │                   │
  candidate-only       TensorFlow Lite
  heuristic demo       sign classifier
        │                   │
        │               confidence / margin gate
        │                   │
        └─────────┬─────────┘
                  │
          IoU identity tracking
           temporal consensus
                  │
         cooldown + severity tier
                  │
         SQLite metadata journal
                  │
        FastAPI + MJPEG + dashboard
```

## Why this architecture

A classifier cannot infer what a scene contains unless someone first proposes plausible regions. The detector here is intentionally simple: HSV red/blue segmentation followed by contour geometry and NMS. This is a constrained baseline with known weaknesses under low light, unusual materials, glare, and partially occluded signs. A full-scene detector could replace this module without changing tracking/storage/API contracts.

The default **candidate-only** backend does not recognize real traffic-sign classes: it emits `red sign-like candidate` or `blue sign-like candidate`. It exists to exercise camera capture, geometry, tracking, logging, dashboard, and backpressure on any machine without ML weights. No pretrained model is silently downloaded.

When supplied with a trained TensorFlow Lite model and matching labels, the neural classifier receives RGB crops resized to the training input shape. Float and int8 tensors are normalized/quantized using interpreter metadata. The classifier rejects predictions whose confidence or top-two margin fall below configured thresholds. Rejected predictions are explicitly called `uncertain sign` and cannot generate a verified event.

## Temporal confirmation

Region proposals are matched against tracks by IoU. Each track maintains the last seven labels and scores and requires repeated agreement on a class before confirmation. The default requirement is three matching observations. Unknown classifications do not cast positive class votes. Per-track cooldown avoids continuous repeat alerts while a tracked object stays visible.

**Limitation:** this is a pragmatic lightweight tracker, not DeepSORT/ByteTrack. Fast motion, occlusion, overlaps, and camera cuts may create ID switches. The tracker is benchmarkable and replaceable.

## Backpressure and observability

Acquisition and inference run on different threads. The queue is bounded at two frames. New frames supersede stale queued frames if processing slows; the application counts dropped frames and measures frame processing latency. The dashboard shows processing throughput, p50/p95 processing latency, active tracks, dropped frames, live source, and mode (heuristic vs TFLite).

Synthetic runtime throughput must never be presented as Raspberry Pi hardware throughput. A separate `scripts/benchmark_device.py` records **actual** processing frames/sec against a running service on the target device.

## Persistence and privacy

The event database stores timestamp, track identity, bounding box, label, severity, source, classifier artifact/version, and confidence. It does not store raw video or crops. Annotated JPEGs are created in memory for local MJPEG streaming only. The dashboard is served locally and makes no external analytics calls. The server binds to loopback by default; deliberately exposing it beyond localhost requires network security and authentication that are not implemented here.

SQLite uses WAL journal mode (separate from the C++ NovaKV WAL project) and an event retention cap. Events can be exported as JSONL for replay/audit.

## Trust boundary

This is an **observational developer prototype**, not a driver-assistance, autonomous-driving, safety-critical, or roadworthy system. Its alerts are informational. Performance, calibration, false alert rate, fairness across conditions, and on-device latency require independent evaluation before any operational use.
