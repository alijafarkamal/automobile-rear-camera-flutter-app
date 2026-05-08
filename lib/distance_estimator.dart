import 'dart:math' as math;

import 'cv_config.dart';

enum AlertZone { safe, caution, danger }

/// Port of [vision/distance.py] fused distance + alert helpers.
class DistanceEstimator {
  DistanceEstimator({
    this.focalLengthPx = defaultFocalLengthPx,
  });

  double focalLengthPx;

  void calibrateFromKnownObject({
    required double pixelWidth,
    required double knownDistanceM,
    required double realWidthM,
  }) {
    if (pixelWidth <= 0 || knownDistanceM <= 0 || realWidthM <= 0) {
      throw ArgumentError('Calibration inputs must be positive');
    }
    focalLengthPx = ((pixelWidth * knownDistanceM) / realWidthM * 100).round() / 100;
  }

  double estimateDistanceM({
    required double pixelWidth,
    required double pixelHeight,
    required double realWidth,
    required double realHeight,
    required List<double> bbox, // x1,y1,x2,y2 in original image px
    required int frameWidth,
    required int frameHeight,
  }) {
    if (pixelWidth <= 0) return double.infinity;

    // Scale focal length to the actual source frame width so the formula is
    // resolution-independent. defaultFocalLengthPx is calibrated at
    // kFocalCalibrationWidthPx (480 px, 70° H-FoV).
    final effectiveFocal = focalLengthPx *
        (frameWidth > 0
            ? frameWidth / kFocalCalibrationWidthPx
            : 1.0);

    final dWidth = (realWidth * effectiveFocal) / pixelWidth;
    final dHeight = pixelHeight > 0
        ? (realHeight * effectiveFocal) / pixelHeight
        : dWidth;

    final y2 = bbox[3];
    final dY = (frameHeight > 0 && y2 > 0)
        ? dWidth * (1.0 - 0.15 * (y2 / frameHeight))
        : dWidth;

    // Adaptive width/height blending.
    // At close range bbox fills most of the frame so the visible portion may
    // be less than the full object width — height is more reliable then.
    final widthFill = frameWidth > 0 ? pixelWidth / frameWidth : 0.0;
    double wW, wH;
    if (widthFill > 0.55) {
      wW = 0.10; wH = 0.75; // almost full-frame: rely almost entirely on height
    } else if (widthFill > 0.30) {
      wW = 0.25; wH = 0.60; // medium range
    } else {
      wW = 0.55; wH = 0.30; // far: width is stable
    }
    final fused = wW * dWidth + wH * dHeight + 0.15 * dY;
    return math.max(fused, minDistanceM).toDouble().clamp(minDistanceM, 999.0);
  }

  static AlertZone zoneForDistance(double? distanceM) {
    if (distanceM == null || distanceM.isInfinite) return AlertZone.safe;
    if (distanceM > zoneSafeThresholdM) return AlertZone.safe;
    if (distanceM >= zoneCautionThresholdM) return AlertZone.caution;
    return AlertZone.danger;
  }
}
