/// Mirrors Python [config.py] for consistent project reporting.
library cv_config;

/// COCO class ids used by YOLOv8n for our targets.
const Map<String, int> targetClassIds = {
  'person': 0,
  'car': 2,
  'motorcycle': 3,
  'bus': 5,
  'truck': 7,
};

const Map<String, double> knownWidthsM = {
  'car': 1.8,
  'truck': 2.5,
  'bus': 2.6,
  'motorcycle': 0.7,
  'person': 0.5,
};

const Map<String, double> knownHeightsM = {
  'car': 1.5,
  'truck': 3.5,
  'bus': 3.2,
  'motorcycle': 1.1,
  'person': 1.7,
};

/// Focal length at [kFocalCalibrationWidthPx] source resolution, 70° H-FoV.
/// Scaled automatically to actual source width inside estimateDistanceM.
const double defaultFocalLengthPx = 343.0;

/// Reference source width the focal is calibrated at (px).
const double kFocalCalibrationWidthPx = 480.0;

const double zoneSafeThresholdM = 3.0;
const double zoneCautionThresholdM = 1.5;
const double maxRadarRangeM = 10.0;

/// Minimum physically-plausible distance (metres).
const double minDistanceM = 0.15;

/// Show "TOO CLOSE" label when smoothed distance is below this (metres).
const double tooCloseLabelThresholdM = 0.4;

const int inputSize = 640;

/// Per-COCO-id minimum confidence.
/// Raised from 0.12 → 0.40 to eliminate low-confidence false positives
/// (random objects hitting 13–33 % on YOLOv8n-nano noise floor).
const Map<int, double> classConfThresholds = {
  0: 0.40, // person
  2: 0.40, // car
  3: 0.40, // motorcycle
  5: 0.40, // bus
  7: 0.40, // truck
};
const double defaultConfThreshold = 0.40;
