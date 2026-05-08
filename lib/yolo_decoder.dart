import 'dart:math' as math;

import 'package:flutter/foundation.dart';

import 'cv_config.dart';

class Detection {
  Detection({
    required this.className,
    required this.classId,
    required this.score,
    required this.xyxy,
  });

  final String className;
  final int classId;
  final double score;
  /// x1,y1,x2,y2 in **original** image pixel coordinates.
  final List<double> xyxy;

  int get bboxWidthPx => (xyxy[2] - xyxy[0]).round().abs().clamp(1, 100000);
  int get bboxHeightPx => (xyxy[3] - xyxy[1]).round().abs().clamp(1, 100000);

  double get centerY => (xyxy[1] + xyxy[3]) / 2.0;
}

/// Decode Ultralytics YOLOv8 TFLite output \[1,84,8400\] with NMS.
class YoloDecoder {
  YoloDecoder({
    this.iouThreshold = 0.45,
  });

  final double iouThreshold;

  static final List<String> _cocoNames = List<String>.generate(80, (i) => 'id_$i');

  static final Set<int> _targetClassSet = targetClassIds.values.toSet();

  static const Set<int> _vehicleClassIds = {2, 3, 5, 7};

  static String _classNameFromId(int id) {
    switch (id) {
      case 0:
        return 'person';
      case 2:
        return 'car';
      case 3:
        return 'motorcycle';
      case 5:
        return 'bus';
      case 7:
        return 'truck';
      default:
        return _cocoNames[id];
    }
  }

  List<Detection> decodeFlat({
    required Float32List output,
    required int numAnchors,
    required List<double> Function(double x1, double y1, double x2, double y2) toOriginal,
  }) {
    // Flat layout [84][N]: index = channel * numAnchors + i
    if (output.length < 84 * numAnchors) {
      debugPrint('Unexpected flat output length: ${output.length}');
      return [];
    }

    final candidates = <_RawBox>[];

    for (var i = 0; i < numAnchors; i++) {
      // YOLOv8n TFLite outputs normalized [0-1] coords relative to the
      // 640×640 model input. Multiply by inputSize to get pixel coordinates
      // before the letterbox un-mapping in toOriginal().
      final cx = output[0 * numAnchors + i] * inputSize;
      final cy = output[1 * numAnchors + i] * inputSize;
      final w  = output[2 * numAnchors + i] * inputSize;
      final h  = output[3 * numAnchors + i] * inputSize;

      var bestScore = 0.0;
      var bestId = 0;
      for (var c = 0; c < 80; c++) {
        final raw = output[(4 + c) * numAnchors + i];
        final s = _score(raw);
        if (s > bestScore) {
          bestScore = s;
          bestId = c;
        }
      }

      if (!_targetClassSet.contains(bestId)) continue;
      final minScore = classConfThresholds[bestId] ?? defaultConfThreshold;
      if (bestScore < minScore) continue;

      final x1 = cx - w / 2;
      final y1 = cy - h / 2;
      final x2 = cx + w / 2;
      final y2 = cy + h / 2;

      final xyxyOrig = toOriginal(x1, y1, x2, y2);
      candidates.add(
        _RawBox(
          classId: bestId,
          score: bestScore,
          xyxy: xyxyOrig,
        ),
      );
    }

    candidates.sort((a, b) => b.score.compareTo(a.score));
    final picked = <_RawBox>[];
    for (final b in candidates) {
      var keep = true;
      for (final p in picked) {
        if (p.classId != b.classId) continue;
        if (_iou(b.xyxy, p.xyxy) > iouThreshold) {
          keep = false;
          break;
        }
      }
      if (keep) picked.add(b);
    }

    return picked
        .map(
          (e) => Detection(
            className: _classNameFromId(e.classId),
            classId: e.classId,
            score: e.score,
            xyxy: e.xyxy,
          ),
        )
        .toList();
  }

  /// Widest vehicle (car/motorcycle/bus/truck) if any; otherwise widest target.
  static Detection? nearestObstacle(List<Detection> dets) {
    if (dets.isEmpty) return null;
    final vehicles =
        dets.where((d) => _vehicleClassIds.contains(d.classId)).toList();
    final pool = vehicles.isNotEmpty ? vehicles : dets;
    return pool.reduce((a, b) => a.bboxWidthPx >= b.bboxWidthPx ? a : b);
  }

  static double _sigmoid(double x) {
    if (x < -20) return 0;
    if (x > 20) return 1;
    return 1.0 / (1.0 + math.exp(-x));
  }

  /// Public wrapper for diagnostic scoring.
  static double scoreDiagnostic(double raw) => _score(raw);

  /// Handles model outputs that may be raw logits or sigmoid probabilities.
  static double _score(double raw) {
    if (raw < -0.1 || raw > 1.1) {
      return _sigmoid(raw);
    }
    return raw.clamp(0.0, 1.0);
  }

  static double _iou(List<double> a, List<double> b) {
    final xa1 = a[0], ya1 = a[1], xa2 = a[2], ya2 = a[3];
    final xb1 = b[0], yb1 = b[1], xb2 = b[2], yb2 = b[3];
    final interX1 = math.max(xa1, xb1);
    final interY1 = math.max(ya1, yb1);
    final interX2 = math.min(xa2, xb2);
    final interY2 = math.min(ya2, yb2);
    final iw = math.max(0.0, interX2 - interX1);
    final ih = math.max(0.0, interY2 - interY1);
    final inter = iw * ih;
    final areaA = math.max(0.0, xa2 - xa1) * math.max(0.0, ya2 - ya1);
    final areaB = math.max(0.0, xb2 - xb1) * math.max(0.0, yb2 - yb1);
    final union = areaA + areaB - inter + 1e-6;
    return inter / union;
  }
}

class _RawBox {
  _RawBox({
    required this.classId,
    required this.score,
    required this.xyxy,
  });

  final int classId;
  final double score;
  final List<double> xyxy;
}
