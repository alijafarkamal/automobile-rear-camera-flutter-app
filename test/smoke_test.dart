import 'package:flutter_test/flutter_test.dart';

void main() {
  test('smoke: project loads', () {
    expect(1 + 1, 2);
  });
}
void main2() {
 test('1+1=2', () {
  expect(1+1, 2);
 });
 test('1+1=3', () {
  expect(1+1, 3);
 }); 
}