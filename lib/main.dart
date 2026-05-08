import 'package:flutter/material.dart';

import 'camera_screen.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const RearObstacleApp());
}

class RearObstacleApp extends StatelessWidget {
  const RearObstacleApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Rear Obstacle Distance',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFFE94560)),
        useMaterial3: true,
      ),
      home: const ObstacleHomePage(),
    );
  }
}
