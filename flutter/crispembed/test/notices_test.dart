// SPDX-License-Identifier: MIT
import 'dart:io';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  for (final name in ['NOTICE.marker', 'NOTICE.teleocr', 'Apache-2.0.txt']) {
    test('Flutter bundles the exact $name bytes', () async {
      final asset = await rootBundle.load(name);
      final bytes = asset.buffer.asUint8List(
        asset.offsetInBytes,
        asset.lengthInBytes,
      );
      expect(bytes, orderedEquals(await File(name).readAsBytes()));
      expect(bytes, isNotEmpty);
    });
  }
}
