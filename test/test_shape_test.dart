import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

void main() {
  test('print shapes', () async {
    final interpreter = await Interpreter.fromFile(File('assets/models/yolov8n_float32.tflite'));
    print("Input shape: \${interpreter.getInputTensor(0).shape}");
    print("Output shape: \${interpreter.getOutputTensor(0).shape}");
  });
}
