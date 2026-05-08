# Automobile Rear Camera — Flutter App

Real-time rear-obstacle detection and distance estimation for Android, built with Flutter and YOLOv8n TFLite. No internet connection required — fully on-device inference.

---

## Download APK

**[Download latest release APK](https://drive.google.com/drive/folders/1ygco5Dz-Q0LPf9s6RNa9fVoyhALlGxM6?usp=sharing)**

---

## Features

- Real-time object detection via **YOLOv8n** (TensorFlow Lite, float32)
- Dynamic distance estimation for cars, trucks, buses, motorcycles, and persons
- Three alert zones with color coding and haptic feedback:
  - **SAFE** (green) — > 3 m
  - **CAUTION** (yellow) — 1.5 – 3 m
  - **DANGER** (red) — < 1.5 m
- Persistent bounding boxes with corner-accent overlay
- Aspect-ratio heuristic to correct YOLOv8n car/truck/bus misclassification from rear views
- Exponential Moving Average (EMA) smoothing to stabilize distance readings
- 5 FPS inference cap with time-based throttling (no UI lag)

---

## Tech Stack

| Component | Technology |
|---|---|
| UI Framework | Flutter 3 (Dart) |
| Object Detection | YOLOv8n (Ultralytics) — TFLite float32 |
| Inference Runtime | tflite_flutter ^0.11 |
| Camera | camera ^0.11 |
| Compute | Flutter `compute` isolates |

---

## Distance Estimation Approach

Monocular distance is estimated from bounding box geometry using the pin-hole camera model:

```
distance = (real_object_width × focal_length_px) / bbox_width_px
```

Key design decisions:

1. **Resolution-invariant focal length** — `focalLengthPx` is calibrated at a 480 px reference width (70° H-FoV assumption) and scaled automatically to the actual source frame width at runtime, so the formula is correct regardless of camera resolution preset.

2. **Adaptive width/height blending** — At close range (bbox fills > 55% of frame) the formula switches to height-dominant estimation because only a partial car width may be visible; at far range width dominates.

3. **EMA smoothing** — New readings are blended with the previous smoothed value (`α = 0.35`) to suppress per-frame jitter.

4. **Detection persistence** — The last known bounding box and distance are held for up to 6 missed inference frames (~1.2 s) before clearing, avoiding flicker on brief occlusions.

5. **Class correction heuristic** — Bus/truck detections with a wider-than-tall bounding box are re-labelled as "car", fixing the common rear-view misclassification of YOLOv8n-nano.

---

## Project Structure

```
lib/
├── main.dart               # App entry point
├── camera_screen.dart      # Camera stream, inference loop, UI
├── yolo_decoder.dart       # YOLOv8n TFLite output → Detection objects + NMS
├── distance_estimator.dart # Focal-length formula, adaptive blending, zone logic
├── image_utils.dart        # YUV420 → letterboxed RGB tensor
├── tflite_yolo.dart        # TFLite model loader / runner
└── cv_config.dart          # Centralised CV constants (thresholds, class dims)

assets/
└── models/
    └── yolov8n_float32.tflite
```

---

## How to Build

Requirements: Flutter 3.x, Java 17, Android SDK 34.

```bash
flutter pub get
flutter build apk --release
# APK → build/app/outputs/flutter-apk/app-release.apk
```

---

## Permissions

- `CAMERA` — live viewfinder for obstacle detection

---

## Limitations

- Monocular distance is approximate (±20–30%) without camera calibration
- YOLOv8n-nano is a small model; detection confidence drops at night or in heavy rain
- Distance accuracy degrades when only a partial object is visible in frame
