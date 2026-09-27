import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:medical/core/database/app_database.dart';
import 'package:medical/features/chat/models/chat_message.dart';

void main() {
  late AppDatabase db;

  setUp(() {
    // Use an in-memory SQLite database for fast, isolated tests
    db = AppDatabase(NativeDatabase.memory());
  });

  tearDown(() async {
    await db.close();
  });

  test('Database initializes with default medical consultation session', () async {
    final sessions = await db.watchSessions().first;
    expect(sessions.length, 1);
    expect(sessions.first.id, 'default_session');
    expect(sessions.first.title, 'Medical Consultation');
  });

  test('Inserts and retrieves user and AI doctor messages in chronological order', () async {
    final now = DateTime.now();

    final userMsg = ChatMessage(
      id: 'msg_1',
      text: 'What are the symptoms of acute sinusitis?',
      timestamp: now.subtract(const Duration(seconds: 10)),
      sender: MessageSender.user,
      status: MessageStatus.sent,
    );

    final aiMsg = ChatMessage(
      id: 'msg_2',
      text: 'Common symptoms include facial pressure, nasal congestion, and headache.',
      timestamp: now,
      sender: MessageSender.doctor,
      status: MessageStatus.read,
      tokensPerSecond: 24.5,
      tokenCount: 68,
    );

    await db.upsertMessage(userMsg);
    await db.upsertMessage(aiMsg);

    final messages = await db.getMessagesForSession('default_session');
    expect(messages.length, 2);

    expect(messages[0].id, 'msg_1');
    expect(messages[0].sender, MessageSender.user);
    expect(messages[0].text, 'What are the symptoms of acute sinusitis?');

    expect(messages[1].id, 'msg_2');
    expect(messages[1].sender, MessageSender.doctor);
    expect(messages[1].tokensPerSecond, 24.5);
    expect(messages[1].tokenCount, 68);
  });

  test('Updates existing message on conflict (e.g. streaming accumulation)', () async {
    final now = DateTime.now();

    final partialAiMsg = ChatMessage(
      id: 'msg_stream',
      text: 'Partial response...',
      timestamp: now,
      sender: MessageSender.doctor,
      status: MessageStatus.sending,
    );

    await db.upsertMessage(partialAiMsg);

    final updatedAiMsg = partialAiMsg.copyWith(
      text: 'Complete response from local LLM.',
      status: MessageStatus.read,
      tokensPerSecond: 28.2,
      tokenCount: 110,
    );

    await db.upsertMessage(updatedAiMsg);

    final messages = await db.getMessagesForSession('default_session');
    expect(messages.length, 1);
    expect(messages.first.text, 'Complete response from local LLM.');
    expect(messages.first.status, MessageStatus.read);
    expect(messages.first.tokensPerSecond, 28.2);
    expect(messages.first.tokenCount, 110);
  });

  test('Clears session messages successfully', () async {
    final msg = ChatMessage(
      id: 'msg_temp',
      text: 'Temporary query',
      timestamp: DateTime.now(),
      sender: MessageSender.user,
    );

    await db.upsertMessage(msg);
    expect((await db.getMessagesForSession('default_session')).length, 1);

    await db.clearSessionMessages('default_session');
    expect((await db.getMessagesForSession('default_session')).length, 0);
  });
}
