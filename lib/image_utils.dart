import 'dart:math' as math;
import 'dart:typed_data';

import 'package:camera/camera.dart';

import 'cv_config.dart';

/// Serializable camera frame + strides (for [compute] isolate).
class YuvPacket {
  YuvPacket({
    required this.rawWidth,
    required this.rawHeight,
    required this.sensorOrientation,
    required this.yBytes,
    required this.yRowStride,
    required this.uBytes,
    required this.vBytes,
    required this.uRowStride,
    required this.vRowStride,
    required this.uPixelStride,
    required this.vPixelStride,
    required this.isNv21,
  });

  final int rawWidth;
  final int rawHeight;
  final int sensorOrientation;
  final Uint8List yBytes;
  final int yRowStride;
  final Uint8List uBytes;
  final Uint8List vBytes;
  final int uRowStride;
  final int vRowStride;
  final int uPixelStride;
  final int vPixelStride;
  final bool isNv21;

  /// Copies (possibly heavy) — runs before isolate.
  factory YuvPacket.fromCameraImage(CameraImage image, int sensorOrientation) {
    final yPlane = image.planes[0];
    final yBytes = Uint8List.fromList(yPlane.bytes);

    if (image.planes.length >= 3) {
      final u = image.planes[1];
      final v = image.planes[2];
      return YuvPacket(
        rawWidth: image.width,
        rawHeight: image.height,
        sensorOrientation: sensorOrientation,
        yBytes: yBytes,
        yRowStride: yPlane.bytesPerRow,
        uBytes: Uint8List.fromList(u.bytes),
        vBytes: Uint8List.fromList(v.bytes),
        uRowStride: u.bytesPerRow,
        vRowStride: v.bytesPerRow,
        uPixelStride: u.bytesPerPixel ?? 1,
        vPixelStride: v.bytesPerPixel ?? 1,
        isNv21: false,
      );
    }

    final uv = image.planes[1];
    return YuvPacket(
      rawWidth: image.width,
      rawHeight: image.height,
      sensorOrientation: sensorOrientation,
      yBytes: yBytes,
      yRowStride: yPlane.bytesPerRow,
      uBytes: Uint8List.fromList(uv.bytes),
      vBytes: Uint8List(0),
      uRowStride: uv.bytesPerRow,
      vRowStride: 0,
      uPixelStride: uv.bytesPerPixel ?? 2,
      vPixelStride: uv.bytesPerPixel ?? 2,
      isNv21: true,
    );
  }
}

/// Letterboxed model input + mapping back to **oriented** image pixels (matches old `copyRotate` preview).
class LetterboxedTensorResult {
  LetterboxedTensorResult({
    required this.tensor,
    required this.scale,
    required this.padX,
    required this.padY,
    required this.srcW,
    required this.srcH,
  });

  final Float32List tensor;

  /// Model 640×640 letterbox params relative to oriented size [srcW]×[srcH].
  final double scale;
  final double padX;
  final double padY;
  final int srcW;
  final int srcH;

  List<double> toOriginalXyxy(double x1, double y1, double x2, double y2) {
    return <double>[
      ((x1 - padX) / scale).clamp(0.0, srcW.toDouble()),
      ((y1 - padY) / scale).clamp(0.0, srcH.toDouble()),
      ((x2 - padX) / scale).clamp(0.0, srcW.toDouble()),
      ((y2 - padY) / scale).clamp(0.0, srcH.toDouble()),
    ];
  }
}

/// Top-level for [compute] — must be static/top-level.
LetterboxedTensorResult letterboxedTensorFromYuv(YuvPacket p) {
  final oriented = _orientedSize(p.rawWidth, p.rawHeight, p.sensorOrientation);
  final srcW = oriented.$1;
  final srcH = oriented.$2;

  final scale = math.min(inputSize / srcW, inputSize / srcH);
  final newW = srcW * scale;
  final newH = srcH * scale;
  final padX = (inputSize - newW) / 2.0;
  final padY = (inputSize - newH) / 2.0;

  final out = Float32List(inputSize * inputSize * 3);
  const fill = 114 / 255.0;
  for (var i = 0; i < out.length; i++) {
    out[i] = fill;
  }

  for (var dy = 0; dy < inputSize; dy++) {
    for (var dx = 0; dx < inputSize; dx++) {
      final ox = (dx - padX) / scale;
      final oy = (dy - padY) / scale;
      if (ox < 0 || oy < 0 || ox >= srcW - 0.001 || oy >= srcH - 0.001) {
        continue;
      }
      final oxi = ox.floor().clamp(0, srcW - 1);
      final oyi = oy.floor().clamp(0, srcH - 1);

      final raw = _orientedToRaw(oxi, oyi, p.sensorOrientation, p.rawWidth, p.rawHeight);
      final rx = raw.$1;
      final ry = raw.$2;

      final yuv = _readYuv(p, rx, ry);
      final base = (dy * inputSize + dx) * 3;
      out[base] = yuv.$1;
      out[base + 1] = yuv.$2;
      out[base + 2] = yuv.$3;
    }
  }

  return LetterboxedTensorResult(
    tensor: out,
    scale: scale,
    padX: padX,
    padY: padY,
    srcW: srcW,
    srcH: srcH,
  );
}

(int, int) _orientedSize(int rawW, int rawH, int sensorOrientation) {
  switch (sensorOrientation) {
    case 90:
    case 270:
      return (rawH, rawW);
    default:
      return (rawW, rawH);
  }
}

/// Map oriented pixel (ox, oy) to raw sensor pixel (rx, ry).
/// Oriented size is (rawH, rawW) when [sensorOrientation] is 90/270.
(int, int) _orientedToRaw(int ox, int oy, int sensorOrientation, int rawW, int rawH) {
  switch (sensorOrientation) {
    case 90:
      final rx = rawW - 1 - oy;
      final ry = ox;
      return (rx.clamp(0, rawW - 1), ry.clamp(0, rawH - 1));
    case 270:
      final rx = oy;
      final ry = rawH - 1 - ox;
      return (rx.clamp(0, rawW - 1), ry.clamp(0, rawH - 1));
    case 180:
      return (
        (rawW - 1 - ox).clamp(0, rawW - 1),
        (rawH - 1 - oy).clamp(0, rawH - 1),
      );
    default:
      return (ox.clamp(0, rawW - 1), oy.clamp(0, rawH - 1));
  }
}

(double, double, double) _readYuv(YuvPacket p, int x, int y) {
  final yv = p.yBytes[y * p.yRowStride + x] & 0xff;

  late int u;
  late int v;
  if (p.isNv21) {
    final uvx = x >> 1;
    final uvy = y >> 1;
    final offset = uvy * p.uRowStride + uvx * p.uPixelStride;
    if (offset + 1 >= p.uBytes.length) {
      u = 128;
      v = 128;
    } else {
      v = p.uBytes[offset] & 0xff;
      u = p.uBytes[offset + 1] & 0xff;
    }
  } else {
    final uvx = x >> 1;
    final uvy = y >> 1;
    final ui = uvy * p.uRowStride + uvx * p.uPixelStride;
    final vi = uvy * p.vRowStride + uvx * p.vPixelStride;
    u = ui < p.uBytes.length ? (p.uBytes[ui] & 0xff) : 128;
    v = vi < p.vBytes.length ? (p.vBytes[vi] & 0xff) : 128;
  }

  final ud = u - 128.0;
  final vd = v - 128.0;
  final rf = (yv + 1.370705 * vd).clamp(0.0, 255.0) / 255.0;
  final gf = (yv - 0.337633 * ud - 0.698001 * vd).clamp(0.0, 255.0) / 255.0;
  final bf = (yv + 1.732446 * ud).clamp(0.0, 255.0) / 255.0;
  return (rf, gf, bf);
}
