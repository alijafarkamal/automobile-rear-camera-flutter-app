import 'package:flutter_test/flutter_test.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:typed_data';

void main() {
  test('tflite run flat buffer', () async {
    final interpreter = await Interpreter.fromAsset('assets/models/yolov8n_float32.tflite');
    final inputShape = interpreter.getInputTensor(0).shape;
    final outputShape = interpreter.getOutputTensor(0).shape;
    print('input shape: $inputShape');
    print('output shape: $outputShape');
    
    final inputFlat = Float32List(640 * 640 * 3);
    final outputFlat = Float32List(84 * 8400);
    
    try {
      interpreter.run(inputFlat.buffer.asUint8List(), outputFlat.buffer.asUint8List());
      print('Run successful with Uint8List');
    } catch (e) {
      print('Run failed with Uint8List: $e');
    }

    try {
      interpreter.run(inputFlat, outputFlat);
      print('Run successful with Float32List');
    } catch (e) {
      print('Run failed with Float32List: $e');
    }
  });
}
