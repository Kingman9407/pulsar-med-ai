import 'package:flutter_test/flutter_test.dart';
import 'package:medical/main.dart';
import 'package:medical/features/chat/views/chat_view.dart';

void main() {
  testWidgets('MedicalApp loads chat screen smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const MedicalApp());

    // Verify that the ChatView loads
    expect(find.byType(ChatView), findsOneWidget);
  });
}
