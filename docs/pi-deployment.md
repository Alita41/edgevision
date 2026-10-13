# Raspberry Pi deployment

This is a *target setup guide*, not a claim that the repository was benchmarked on a Pi.

## Recommended setup

- Raspberry Pi 4/5 or another Linux ARM64 host, recent 64-bit OS
- Raspberry Pi Camera Module, compatible USB camera, or local test video
- Python 3.10+ and OpenCV; install TensorFlow Lite runtime matching your OS/architecture
- Consistent power and cooling, especially for repeatable latency tests

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip libgl1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Then install a compatible tflite-runtime wheel for your Pi/OS architecture.
```

Try an instrumented camera pipeline without a trained model:

```bash
python -m edgevision.cli serve --source camera:0 --host 127.0.0.1 --port 8080
```

For actual neural sign recognition, train/export the model separately (often much faster on a GPU machine). Copy only the `.tflite` file and `labels.json` to the Pi:

```bash
python -m edgevision.cli serve --source camera:0 \
  --model models/edgevision-int8.tflite \
  --labels models/labels.json --host 127.0.0.1
```

Do not expose this unauthenticated dashboard on a public IP address. For remote access use an authenticated VPN/reverse proxy.

Measure actual sustained device performance:

```bash
python scripts/benchmark_device.py --seconds 60 --out artifacts/my-pi.json
```

Document `vcgencmd measure_temp` / `vcgencmd get_throttled` if available, input resolution, camera FPS, model version, OS/kernel, and power mode with your benchmark report. Do not treat the synthetic demo FPS as hardware FPS.
