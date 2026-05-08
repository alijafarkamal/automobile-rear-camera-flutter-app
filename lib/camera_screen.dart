import 'dart:async';
import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:permission_handler/permission_handler.dart';

import 'cv_config.dart';
import 'distance_estimator.dart';
import 'image_utils.dart';
import 'tflite_yolo.dart';
import 'yolo_decoder.dart';

class ObstacleHomePage extends StatefulWidget {
  const ObstacleHomePage({super.key});

  @override
  State<ObstacleHomePage> createState() => _ObstacleHomePageState();
}

class _ObstacleHomePageState extends State<ObstacleHomePage> {
  static const _minInferenceIntervalMs = 200;
  static const _distanceEmaAlpha = 0.35;
  static const _maxNoDetFrames = 3;

  final TfliteYolo _yolo = TfliteYolo();
  final DistanceEstimator _distance = DistanceEstimator();

  CameraController? _camera;
  bool _busy = false;
  bool _modelReady = false;
  String? _bootstrapError;
  bool _streamOn = false;
  List<_UiDetection> _lastDets = [];
  double? _nearestM;
  double? _smoothedM;
  AlertZone _zone = AlertZone.safe;
  DateTime _lastInferenceAt = DateTime.fromMillisecondsSinceEpoch(0);
  int _inferenceCountForLog = 0;
  int _frameW = 1;
  int _frameH = 1;
  AlertZone? _lastHapticZone;
  int _noDetFrames = 0;

  @override
  void initState() {
    super.initState();
    _bootstrap();
  }

  Future<void> _bootstrap() async {
    setState(() {
      _bootstrapError = null;
      _modelReady = false;
    });

    try {
      final camStatus = await Permission.camera.request();
      if (!camStatus.isGranted) {
        throw StateError('Camera permission denied');
      }

      await _yolo.load();

      final cams = await availableCameras();
      if (cams.isEmpty) {
        throw StateError('No cameras available');
      }
      final back = cams.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cams.first,
      );

      _camera = CameraController(
        back,
        ResolutionPreset.low,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      await _camera!.initialize();

      if (!mounted) return;
      setState(() {
        _modelReady = true;
        _bootstrapError = null;
      });
    } catch (e, st) {
      debugPrint('Bootstrap failed: $e\n$st');
      if (!mounted) return;
      setState(() {
        _modelReady = false;
        _bootstrapError = e.toString();
      });
    }
  }

  @override
  void dispose() {
    _camera?.dispose();
    _yolo.close();
    super.dispose();
  }

  void _toggleStream() {
    if (!_modelReady || _camera == null) return;
    if (_streamOn) {
      _camera!.stopImageStream();
      setState(() {
        _streamOn = false;
        _lastDets = [];
        _nearestM = null;
        _smoothedM = null;
        _noDetFrames = 0;
      });
    } else {
      _camera!.startImageStream(_onFrame);
      setState(() => _streamOn = true);
    }
  }

  Future<void> _onFrame(CameraImage image) async {
    if (!_yolo.isLoaded || _busy || _camera == null) return;

    final now = DateTime.now();
    if (now.difference(_lastInferenceAt).inMilliseconds <
        _minInferenceIntervalMs) {
      return;
    }
    _lastInferenceAt = now;

    _busy = true;
    try {
      final packet = YuvPacket.fromCameraImage(
        image,
        _camera!.description.sensorOrientation,
      );

      final letterboxed = await compute(letterboxedTensorFromYuv, packet);
      final dets = _yolo.runLetterboxed(letterboxed);

      final nearestDet = YoloDecoder.nearestObstacle(dets);
      double? distM;
      if (nearestDet != null) {
        final fillRatio = nearestDet.bboxWidthPx / letterboxed.srcW;
        final wM = knownWidthsM[nearestDet.className] ?? 1.0;
        final hM = knownHeightsM[nearestDet.className] ?? 1.5;
        distM = _distance.estimateDistanceM(
          pixelWidth: nearestDet.bboxWidthPx.toDouble(),
          pixelHeight: nearestDet.bboxHeightPx.toDouble(),
          realWidth: wM,
          realHeight: hM,
          bbox: nearestDet.xyxy,
          frameWidth: letterboxed.srcW,
          frameHeight: letterboxed.srcH,
        );
        // #region agent log
        debugPrint(
          '[DBG-f421c0][post-fix] srcW=${letterboxed.srcW} srcH=${letterboxed.srcH} '
          'bboxW=${nearestDet.bboxWidthPx} bboxH=${nearestDet.bboxHeightPx} '
          'fill=${fillRatio.toStringAsFixed(3)} class=${nearestDet.className} '
          'distM=${distM.toStringAsFixed(2)}m',
        );
        // #endregion
      }

      // #region agent log
      if (distM != null || _smoothedM != null) {
        debugPrint('[DBG-f421c0][post-fix] EMA: prevSm=${_smoothedM?.toStringAsFixed(2) ?? "null"} raw=${distM?.toStringAsFixed(2) ?? "null"} noDetFr=$_noDetFrames');
      }
      // #endregion

      if (distM != null) {
        _noDetFrames = 0;
        final rawDist = distM;
        _smoothedM = _smoothedM == null
            ? rawDist
            : _distanceEmaAlpha * rawDist + (1 - _distanceEmaAlpha) * _smoothedM!;
      } else {
        _noDetFrames++;
        if (_noDetFrames > _maxNoDetFrames) {
          _smoothedM = null;
        }
      }

      double maxRaw = -1000.0;
      final outFlat = _yolo.outputFlat;
      if (outFlat != null) {
        for (int i = 0; i < outFlat.length; i++) {
          if (outFlat[i] > maxRaw) maxRaw = outFlat[i];
        }
      }

      final uiDets = dets
          .map(
            (d) => _UiDetection(
              xyxy: d.xyxy,
              label: '${_fixedClassName(d)} ${(d.score * 100).toStringAsFixed(0)}%',
            ),
          )
          .toList();

      // #region agent log
      for (final d in dets) {
        final fixed = _fixedClassName(d);
        if (fixed != d.className) {
          debugPrint(
            '[DBG-f421c0][H-B] relabel ${d.className}(id=${d.classId}) '
            'W=${d.bboxWidthPx} H=${d.bboxHeightPx} ratio=${(d.bboxWidthPx / d.bboxHeightPx).toStringAsFixed(2)} → $fixed',
          );
        }
      }
      debugPrint(
        '[DBG-f421c0][H-A] persist: noDetFr=$_noDetFrames thresh=${_maxNoDetFrames * 2} '
        'newDets=${uiDets.length} prevDets=${_lastDets.length}',
      );
      // #endregion

      if (!mounted) return;
      setState(() {
        _frameW = letterboxed.srcW;
        _frameH = letterboxed.srcH;
        // Persist bounding boxes for 2× the distance persistence window.
        // Only clear after sustained absence; avoids instant flicker on missed frames.
        if (uiDets.isNotEmpty) {
          _lastDets = uiDets;
        } else if (_noDetFrames > _maxNoDetFrames * 2) {
          _lastDets = [];
        }
        _nearestM = _smoothedM;
        _zone = DistanceEstimator.zoneForDistance(_smoothedM);
        _hapticForZone(_zone);
      });

      _inferenceCountForLog++;
      if (dets.isNotEmpty) {
        debugPrint(
          'Detections: ${dets.length} | Top Score: ${dets.first.score.toStringAsFixed(2)} | Dist(raw): ${distM?.toStringAsFixed(2)}m | sm: ${_smoothedM?.toStringAsFixed(2)}m',
        );
      } else if (_inferenceCountForLog % 10 == 0) {
        debugPrint('No detections. Max Raw: ${maxRaw.toStringAsFixed(3)}');
      }
    } catch (e, st) {
      debugPrint('Frame processing error: $e\n$st');
    } finally {
      _busy = false;
    }
  }

  /// Aspect-ratio heuristic: bus (id=5) / truck (id=7) seen from the rear
  /// are always taller than wide (doors, bodywork). If the bbox is wider than
  /// tall the model mis-fired — the object is almost certainly a car.
  static String _fixedClassName(Detection d) {
    if ((d.classId == 5 || d.classId == 7) && d.bboxWidthPx > d.bboxHeightPx) {
      return 'car';
    }
    return d.className;
  }

  void _hapticForZone(AlertZone z) {
    if (_lastHapticZone == z) return;
    _lastHapticZone = z;
    if (z == AlertZone.danger) {
      unawaited(HapticFeedback.heavyImpact());
    } else if (z == AlertZone.caution) {
      unawaited(HapticFeedback.mediumImpact());
    }
  }

  Color _zoneColor() {
    switch (_zone) {
      case AlertZone.safe:
        return const Color(0xFF4ECCA3);
      case AlertZone.caution:
        return const Color(0xFFFFD460);
      case AlertZone.danger:
        return const Color(0xFFE94560);
    }
  }

  String _zoneLabel() {
    switch (_zone) {
      case AlertZone.safe:
        return 'SAFE';
      case AlertZone.caution:
        return 'CAUTION';
      case AlertZone.danger:
        return 'DANGER';
    }
  }

  @override
  Widget build(BuildContext context) {
    final cam = _camera;
    final previewReady = cam != null && cam.value.isInitialized;

    return Scaffold(
      backgroundColor: const Color(0xFF0B1021),
      appBar: AppBar(
        title: const Text('Rear Obstacle Distance'),
        backgroundColor: const Color(0xFF16213E),
      ),
      body: Column(
        children: [
          if (_bootstrapError != null)
            MaterialBanner(
              content: Text(_bootstrapError!, style: const TextStyle(fontSize: 13)),
              backgroundColor: Colors.red.shade900,
              actions: [
                TextButton(
                  onPressed: _bootstrap,
                  child: const Text('Retry', style: TextStyle(color: Colors.white)),
                ),
              ],
            ),
          if (!previewReady && _bootstrapError == null)
            const Expanded(
              child: Center(child: CircularProgressIndicator()),
            )
          else if (_bootstrapError == null)
            Expanded(
              child: Stack(
                fit: StackFit.expand,
                children: [
                  CameraPreview(cam!),
                  CustomPaint(
                    painter: _GuidelinesPainter(),
                  ),
                  CustomPaint(
                    painter: _BoxPainter(
                      dets: _lastDets,
                      imageW: _frameW.toDouble(),
                      imageH: _frameH.toDouble(),
                    ),
                  ),
                ],
              ),
            )
          else
            const Expanded(
              child: Center(
                child: Padding(
                  padding: EdgeInsets.all(24),
                  child: Text(
                    'Fix the error above, then tap Retry.',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.white70),
                  ),
                ),
              ),
            ),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: const Color(0xFF16213E).withValues(alpha: 0.88),
              border: Border(
                top: BorderSide(color: Colors.white.withValues(alpha: 0.1), width: 1),
              ),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  _zoneLabel(),
                  style: TextStyle(
                    color: _zoneColor(),
                    fontWeight: FontWeight.w900,
                    fontSize: 28,
                    letterSpacing: 2,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  _nearestM == null
                      ? 'SCANNING...'
                      : (_nearestM! < tooCloseLabelThresholdM
                          ? 'TOO CLOSE  •  ${(_nearestM! * 100).toStringAsFixed(0)} cm'
                          : '${_nearestM!.toStringAsFixed(1)} METERS'),
                  style: TextStyle(
                    color: Colors.white.withValues(alpha: 0.7),
                    fontSize: 16,
                    fontWeight: FontWeight.w500,
                  ),
                ),
                const SizedBox(height: 20),
                Row(
                  children: [
                    Expanded(
                      child: Container(
                        height: 56,
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(16),
                          gradient: LinearGradient(
                            colors: _streamOn
                                ? [const Color(0xFFE94560), const Color(0xFFC62828)]
                                : [const Color(0xFF4ECCA3), const Color(0xFF45B39D)],
                          ),
                          boxShadow: [
                            BoxShadow(
                              color: (_streamOn ? const Color(0xFFE94560) : const Color(0xFF4ECCA3))
                                  .withValues(alpha: 0.3),
                              blurRadius: 12,
                              offset: const Offset(0, 4),
                            ),
                          ],
                        ),
                        child: ElevatedButton(
                          onPressed: previewReady && _modelReady ? _toggleStream : null,
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.transparent,
                            shadowColor: Colors.transparent,
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                          ),
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(
                                _streamOn ? Icons.stop_rounded : Icons.play_arrow_rounded,
                                color: Colors.white,
                                size: 28,
                              ),
                              const SizedBox(width: 8),
                              Text(
                                _streamOn ? 'STOP SYSTEM' : 'START SYSTEM',
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontWeight: FontWeight.bold,
                                  fontSize: 16,
                                  letterSpacing: 1.1,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _UiDetection {
  _UiDetection({required this.xyxy, required this.label});
  final List<double> xyxy;
  final String label;
}

class _BoxPainter extends CustomPainter {
  _BoxPainter({
    required this.dets,
    required this.imageW,
    required this.imageH,
  });

  final List<_UiDetection> dets;
  final double imageW;
  final double imageH;

  @override
  @override
  void paint(Canvas canvas, Size size) {
    if (imageW <= 1 || imageH <= 1) return;

    final scale = math.max(size.width / imageW, size.height / imageH);
    final dx = (size.width - imageW * scale) / 2;
    final dy = (size.height - imageH * scale) / 2;

    for (final d in dets) {
      final rect = Rect.fromLTRB(
        dx + d.xyxy[0] * scale,
        dy + d.xyxy[1] * scale,
        dx + d.xyxy[2] * scale,
        dy + d.xyxy[3] * scale,
      );

      // Outer glow
      final glowPaint = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 6
        ..color = Colors.lightGreenAccent.withValues(alpha: 0.2)
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 4);
      canvas.drawRRect(RRect.fromRectAndRadius(rect, const Radius.circular(8)), glowPaint);

      // Main box
      final boxPaint = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.5
        ..color = Colors.lightGreenAccent;
      canvas.drawRRect(RRect.fromRectAndRadius(rect, const Radius.circular(8)), boxPaint);

      // Corner accents
      final cornerPaint = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 5
        ..color = Colors.white;
      final cornerLen = rect.width * 0.15;
      
      // Top left corner
      canvas.drawLine(rect.topLeft, rect.topLeft + Offset(cornerLen, 0), cornerPaint);
      canvas.drawLine(rect.topLeft, rect.topLeft + Offset(0, cornerLen), cornerPaint);
      
      // Bottom right corner
      canvas.drawLine(rect.bottomRight, rect.bottomRight - Offset(cornerLen, 0), cornerPaint);
      canvas.drawLine(rect.bottomRight, rect.bottomRight - Offset(0, cornerLen), cornerPaint);

      final labelPainter = TextPainter(
        text: TextSpan(
          text: ' ${d.label.toUpperCase()} ',
          style: const TextStyle(
            color: Colors.black,
            fontSize: 11,
            fontWeight: FontWeight.w900,
            backgroundColor: Colors.lightGreenAccent,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      labelPainter.paint(canvas, rect.topLeft - const Offset(0, 16));
    }
  }

  @override
  bool shouldRepaint(covariant _BoxPainter oldDelegate) => true;
}

class _GuidelinesPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final w = size.width;
    final h = size.height;

    // Perspective parameters
    const bottomW = 0.90; // Bottom width fraction
    const topW = 0.45;    // Top width fraction
    const topY = 0.55;    // Horizon relative to screen height

    final bL = w * (1 - bottomW) / 2;
    final bR = w * (1 + bottomW) / 2;
    final tL = w * (1 - topW) / 2;
    final tR = w * (1 + topW) / 2;
    final tY = h * topY;
    final bY = h * 0.95;

    void drawSegment(double startT, double endT, Color color) {
      final p = Paint()
        ..color = color.withValues(alpha: 0.6)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 4
        ..strokeCap = StrokeCap.round;

      final p1L = Offset(tL + (bL - tL) * startT, tY + (bY - tY) * startT);
      final p2L = Offset(tL + (bL - tL) * endT, tY + (bY - tY) * endT);
      final p1R = Offset(tR + (bR - tR) * startT, tY + (bY - tY) * startT);
      final p2R = Offset(tR + (bR - tR) * endT, tY + (bY - tY) * endT);

      canvas.drawLine(p1L, p2L, p);
      canvas.drawLine(p1R, p2R, p);

      // Horizontal markers
      if (endT >= 0.99) {
         canvas.drawLine(p2L, p2R, p..strokeWidth = 2);
      }
    }

    // Red zone (closest)
    drawSegment(0.8, 1.0, const Color(0xFFE94560));
    // Yellow zone
    drawSegment(0.5, 0.8, const Color(0xFFFFD460));
    // Green zone
    drawSegment(0.2, 0.5, const Color(0xFF4ECCA3));

    // Decorative "Radar" arcs at top
    final radarPaint = Paint()
      ..color = Colors.white.withValues(alpha: 0.1)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1;
    for (var i = 1; i <= 3; i++) {
      canvas.drawArc(
        Rect.fromCenter(center: Offset(w/2, tY), width: w * 0.2 * i, height: h * 0.1 * i),
        math.pi, math.pi, false, radarPaint
      );
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => false;
}
