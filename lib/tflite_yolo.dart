import 'package:flutter/foundation.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import 'cv_config.dart';
import 'image_utils.dart';
import 'yolo_decoder.dart';

/// Loads YOLOv8n TFLite and runs inference.
///
/// Uses raw [Uint8List] views for [Interpreter.run] — required by tflite_flutter's
/// float pipeline (avoids broken nested-List conversion).
class TfliteYolo {
  TfliteYolo();

  Interpreter? _interpreter;
  Float32List? _outputFlat;
  int _numAnchors = 8400;

  final YoloDecoder _decoder = YoloDecoder();

  bool get isLoaded => _interpreter != null;
  Float32List? get outputFlat => _outputFlat;

  Future<void> load() async {
    if (_interpreter != null) return;

    final options = InterpreterOptions()..threads = 2;
    _interpreter = await Interpreter.fromAsset(
      'assets/models/yolov8n_float32.tflite',
      options: options,
    );

    final outTensor = _interpreter!.getOutputTensor(0);
    final outElems = outTensor.numBytes() ~/ 4;
    _outputFlat = Float32List(outElems);
    if (outElems % 84 != 0) {
      debugPrint('Unexpected output element count $outElems (not multiple of 84)');
    } else {
      _numAnchors = outElems ~/ 84;
    }
    debugPrint('TFLite anchors: $_numAnchors (output floats: $outElems)');

    final inShape = _interpreter!.getInputTensor(0).shape;
    debugPrint('TFLite input shape: $inShape');
    if (inShape.length != 4 ||
        inShape[0] != 1 ||
        inShape[1] != inputSize ||
        inShape[2] != inputSize ||
        inShape[3] != 3) {
      debugPrint('Warning: expected input [1,$inputSize,$inputSize,3]');
    }

    const inBytes = inputSize * inputSize * 3 * 4;
    if (_interpreter!.getInputTensor(0).numBytes() != inBytes) {
      debugPrint(
        'Input tensor byte size mismatch: got ${_interpreter!.getInputTensor(0).numBytes()}, expected $inBytes',
      );
    }
    if (_interpreter!.getOutputTensor(0).numBytes() != _outputFlat!.length * 4) {
      debugPrint(
        'Output tensor byte size mismatch: got ${_interpreter!.getOutputTensor(0).numBytes()}, expected ${_outputFlat!.length * 4}',
      );
    }
  }

  void close() {
    _interpreter?.close();
    _interpreter = null;
  }

  /// Letterboxed NHWC float tensor + mapping; see [LetterboxedTensorResult].
  List<Detection> runLetterboxed(LetterboxedTensorResult r) {
    final interpreter = _interpreter;
    final outputFlat = _outputFlat;
    if (interpreter == null || outputFlat == null) {
      return [];
    }

    final inBytes = r.tensor.buffer.asUint8List();

    // 1. Direct input setting
    interpreter.getInputTensor(0).setTo(inBytes);

    // 2. Invoke inference
    interpreter.invoke();

    // 3. Direct output access (avoids nested List shaping crash / silent flat buffer failures)
    final outTensor = interpreter.getOutputTensor(0);
    final outBytes = outTensor.data;

    // Fast memory copy to our Float32List via UnmodifiableUint8ListView
    outputFlat.buffer.asUint8List().setRange(0, outBytes.length, outBytes);

    return _decoder.decodeFlat(
      output: outputFlat,
      numAnchors: _numAnchors,
      toOriginal: r.toOriginalXyxy,
    );
  }
}
