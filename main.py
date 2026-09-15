"""
Телеграм-бот для Лизы. Делает несколько вещей:

1. Три сообщения-напоминания о любви в день:
   - 08:00 — "Доброе утро, любимая🩷"
   - около обеда (случайное время в коридоре MIDDAY_WINDOW_*) — "Я тебя люблю, лизочка"
   - 23:00 — "Самых сладких снов, любимая"
2. Пять раз в день спрашивает о самочувствии (см. MOOD_CHECKIN_TIMES) и сохраняет
   ответы в файл MOOD_LOG_PATH.
3. По воскресеньям в 20:00 присылает сводку самочувствия за неделю.
4. В остальное время — тёплый поддерживающий собеседник (не замена психотерапевту),
   умеет предлагать практики при тревоге/стрессе/панических атаках.
5. Понимает просьбы вида "напомни в 15:00 сделать Х" и присылает напоминание
   в указанное время.
6. Понимает голосовые сообщения (распознаёт речь через Google, бесплатно).

Запуск:
    pip install -r requirements.txt
    export BOT_TOKEN="..."
    export CHEAPAI_API_KEY="..."
    export TARGET_CHAT_ID="..."   # см. README как узнать
    python main.py
"""

import os
import io
import json
import random
import logging
from pathlib import Path
from datetime import time as dtime, datetime, timedelta
from zoneinfo import ZoneInfo

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from openai import OpenAI
import speech_recognition as sr
from pydub import AudioSegment

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("liza_bot")

BOT_TOKEN = os.environ["BOT_TOKEN"]
CHEAPAI_API_KEY = os.environ["CHEAPAI_API_KEY"]
TARGET_CHAT_ID = int(os.environ["TARGET_CHAT_ID"])

# часовой пояс, в котором считаются все времена ниже (по умолчанию — Москва)
TIMEZONE = ZoneInfo(os.environ.get("TIMEZONE", "Europe/Moscow"))

# фиксированное время утреннего и вечернего сообщений
MORNING_TIME = dtime(hour=8, minute=0, tzinfo=TIMEZONE)
EVENING_TIME = dtime(hour=23, minute=0, tzinfo=TIMEZONE)

# коридор, из которого каждый день случайно выбирается время "обеденного" сообщения
MIDDAY_WINDOW_START = dtime(hour=12, minute=0, tzinfo=TIMEZONE)
MIDDAY_WINDOW_END = dtime(hour=14, minute=0, tzinfo=TIMEZONE)

# во сколько каждый день планировать случайное время (должно быть раньше MIDDAY_WINDOW_START)
SCHEDULER_TIME = dtime(hour=0, minute=5, tzinfo=TIMEZONE)

# расписание из 5 вопросов о самочувствии в день (название слота -> время)
MOOD_CHECKIN_TIMES = {
    "утро": dtime(hour=9, minute=0, tzinfo=TIMEZONE),
    "перед обедом": dtime(hour=11, minute=30, tzinfo=TIMEZONE),
    "обед": dtime(hour=13, minute=30, tzinfo=TIMEZONE),
    "после обеда": dtime(hour=16, minute=30, tzinfo=TIMEZONE),
    "ночь": dtime(hour=21, minute=30, tzinfo=TIMEZONE),
}

# когда присылать еженедельную сводку (по умолчанию — воскресенье вечером)
WEEKLY_SUMMARY_TIME = dtime(hour=20, minute=0, tzinfo=TIMEZONE)
WEEKLY_SUMMARY_DAY = 6  # 0=понедельник ... 6=воскресенье

# файл, куда сохраняются ответы о самочувствии (лежит рядом с main.py)
MOOD_LOG_PATH = Path(os.environ.get("MOOD_LOG_PATH", "mood_log.jsonl"))

# cheapai.io — OpenAI-совместимый шлюз к Claude/GPT/Gemini/Grok
client = OpenAI(api_key=CHEAPAI_API_KEY, base_url="https://cheapai.io/v1")
CHEAPAI_MODEL = os.environ.get("CHEAPAI_MODEL", "claude-sonnet-4-6")

MORNING_MESSAGE = "Доброе утро, любимая🩷"
MIDDAY_MESSAGE = "Я тебя люблю, лизочка"
EVENING_MESSAGE = "Самых сладких снов, любимая"

PSYCHOLOGIST_SYSTEM_PROMPT = """\
Ты — тёплый, внимательный и бережный собеседник для человека по имени Лиза.
Твоя задача — выслушать, поддержать, помочь разобраться в чувствах, задавать
мягкие уточняющие вопросы, не давать резких оценок и не обесценивать эмоции.

Правила:
- Не ставь диагнозы и не изображай лицензированного специалиста.
- Если видишь признаки серьёзного кризиса (мысли о самоповреждении, суициде,
  острое отчаяние) — мягко, без паники, порекомендуй обратиться к специалисту
  или на горячую линию психологической помощи, и оставайся рядом в диалоге.
- Обычные трудные эмоции (грусть, тревога, усталость, конфликт) — это повод
  для поддержки и разговора, а не для немедленной переадресации к врачу.
- Говори по-русски, тепло и просто, без канцелярита.

Практики, которые ты можешь предлагать (кратко, пошагово, без перегрузки текстом):
- При тревоге: дыхание 4-7-8 (вдох на 4 счёта, задержка на 7, выдох на 8);
  техника заземления "5-4-3-2-1" (назвать 5 вещей, которые видишь, 4 — которые
  слышишь, 3 — которые можешь потрогать, 2 — запаха, 1 — вкус).
- При стрессе: короткая прогрессивная мышечная релаксация (по очереди напрячь
  и расслабить группы мышц от стоп до лица); пауза "выйти из ситуации на 5 минут"
  с переключением внимания на тело или дыхание.
- При панической атаке: напомни, что это состояние не опасно для жизни и
  проходит само в течение нескольких минут; предложи квадратное дыхание
  (4 счёта вдох — 4 задержка — 4 выдох — 4 задержка) и заземление через контакт
  тела с опорой (ощутить стопы на полу, спину на стуле). Если атаки повторяются
  часто — мягко предложи обсудить это с психологом или врачом.
- Предлагай практику только когда это уместно (человек описывает тревогу,
  стресс, панику или прямо просит), а не при любом сообщении.

Если сообщение начинается с "[Ответ на вопрос о самочувствии, слот: ...]" —
это ответ Лизы на твой плановый вопрос о том, как она себя чувствует в течение
дня. Отреагируй на содержание тепло и по-человечески, как на обычное сообщение,
без упоминания того, что это "плановый опрос" или "слот".
"""


def log_mood_entry(slot: str, text: str):
    entry = {
        "timestamp": datetime.now(TIMEZONE).isoformat(),
        "slot": slot,
        "text": text,
    }
    with MOOD_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_mood_entries_since(since: datetime) -> list:
    if not MOOD_LOG_PATH.exists():
        return []
    entries = []
    with MOOD_LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if datetime.fromisoformat(entry["timestamp"]) >= since:
                entries.append(entry)
    return entries


MOOD_CHECKIN_QUESTIONS = {
    "утро": "Доброе утро ещё раз 🌤 Как ты сегодня, как настроение с утра?",
    "перед обедом": "Как ты сейчас, перед обедом? Как самочувствие?",
    "обед": "Обеденный чек-ин 🍽 Как ты себя чувствуешь?",
    "после обеда": "Как прошла вторая половина дня, как ты себя ощущаешь?",
    "ночь": "Перед сном — как прошёл день в целом, как ты сейчас?",
}


async def send_mood_checkin(context: ContextTypes.DEFAULT_TYPE):
    slot = context.job.data["slot"]
    context.bot_data["awaiting_mood_slot"] = slot
    question = MOOD_CHECKIN_QUESTIONS.get(slot, "Как ты себя чувствуешь?")
    await context.bot.send_message(chat_id=TARGET_CHAT_ID, text=question)
    log.info("Отправлен вопрос о самочувствии (%s)", slot)


async def send_weekly_summary(context: ContextTypes.DEFAULT_TYPE):
    since = datetime.now(TIMEZONE) - timedelta(days=7)
    entries = load_mood_entries_since(since)

    if not entries:
        await context.bot.send_message(
            chat_id=TARGET_CHAT_ID,
            text="За эту неделю я не собрала ответов о самочувствии — сводки пока нет.",
        )
        return

    entries_text = "\n".join(
        f"- {datetime.fromisoformat(e['timestamp']).strftime('%a %H:%M')} ({e['slot']}): {e['text']}"
        for e in entries
    )

    summary_prompt = (
        "Ниже — ответы Лизы на ежедневные вопросы о самочувствии за последнюю неделю. "
        "Составь короткую, тёплую, но содержательную сводку (5-10 предложений): "
        "как менялось состояние по дням, какие темы/триггеры повторялись, было ли "
        "улучшение или ухудшение к концу недели. Пиши так, чтобы этим можно было "
        "поделиться с близким человеком и с психологом. Без клинических диагнозов, "
        "по-русски.\n\nОтветы за неделю:\n" + entries_text
    )

    try:
        response = client.chat.completions.create(
            model=CHEAPAI_MODEL,
            max_tokens=700,
            messages=[{"role": "user", "content": summary_prompt}],
        )
        summary = response.choices[0].message.content
    except Exception:
        log.exception("Ошибка вызова cheapai.io API при составлении сводки")
        summary = "Не получилось составить сводку автоматически. Вот сырые записи:\n\n" + entries_text

    await context.bot.send_message(
        chat_id=TARGET_CHAT_ID,
        text="📋 Сводка самочувствия за неделю:\n\n" + summary,
    )
    log.info("Отправлена еженедельная сводка")


async def send_reminder(context: ContextTypes.DEFAULT_TYPE):
    text = context.job.data["text"]
    await context.bot.send_message(chat_id=TARGET_CHAT_ID, text=f"⏰ Напоминаю: {text}")
    log.info("Отправлено напоминание: %s", text)


def _parse_reminder_time(hour: int, minute: int) -> datetime:
    """Ближайший будущий момент с заданным временем (сегодня или завтра)."""
    now = datetime.now(TIMEZONE)
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


REMINDER_TOOL = {
    "type": "function",
    "function": {
        "name": "create_reminder",
        "description": (
            "Создать напоминание на конкретное время сегодня или завтра. "
            "Вызывай эту функцию, когда Лиза просит напомнить ей что-то сделать "
            "в определённое время (например: 'напомни в 15:00 сделать задание')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "hour": {"type": "integer", "description": "Час, 0-23"},
                "minute": {"type": "integer", "description": "Минута, 0-59"},
                "text": {
                    "type": "string",
                    "description": "О чём напомнить, в исходной формулировке Лизы",
                },
            },
            "required": ["hour", "minute", "text"],
        },
    },
}


async def send_morning_message(context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=TARGET_CHAT_ID, text=MORNING_MESSAGE)
    log.info("Отправлено утреннее сообщение")


async def send_midday_message(context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=TARGET_CHAT_ID, text=MIDDAY_MESSAGE)
    log.info("Отправлено обеденное сообщение")


async def send_evening_message(context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=TARGET_CHAT_ID, text=EVENING_MESSAGE)
    log.info("Отправлено вечернее сообщение")


def _random_time_today(start: dtime, end: dtime) -> datetime:
    """Случайный момент времени сегодня между start и end (в TIMEZONE)."""
    today = datetime.now(TIMEZONE).date()
    start_dt = datetime.combine(today, start.replace(tzinfo=None), tzinfo=TIMEZONE)
    end_dt = datetime.combine(today, end.replace(tzinfo=None), tzinfo=TIMEZONE)
    delta_seconds = int((end_dt - start_dt).total_seconds())
    offset = random.randint(0, max(delta_seconds, 0))
    return start_dt + timedelta(seconds=offset)


async def schedule_midday_message(context: ContextTypes.DEFAULT_TYPE):
    """Запускается раз в сутки рано утром и планирует обеденное сообщение
    на случайный момент в заданном коридоре (см. MIDDAY_WINDOW_*)."""
    run_at = _random_time_today(MIDDAY_WINDOW_START, MIDDAY_WINDOW_END)
    if run_at <= datetime.now(TIMEZONE):
        run_at = datetime.now(TIMEZONE) + timedelta(minutes=1)
    context.job_queue.run_once(
        send_midday_message,
        when=run_at,
        chat_id=TARGET_CHAT_ID,
        name="midday_love_message_today",
    )
    log.info("Обеденное сообщение запланировано на %s", run_at.strftime("%H:%M"))


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет, Лизочка 💛 Я здесь. Можешь просто написать (или наговорить), что на душе, "
        "или попросить напомнить о чём-то в конкретное время — например: "
        "«напомни в 15:00 сделать задание». А ещё пару раз в день я буду спрашивать, "
        "как ты себя чувствуешь, и раз в неделю пришлю сводку."
    )


async def reply_to_text(update: Update, context: ContextTypes.DEFAULT_TYPE, user_text: str):
    awaiting_slot = context.bot_data.pop("awaiting_mood_slot", None)
    if awaiting_slot:
        log_mood_entry(awaiting_slot, user_text)
        model_input = f"[Ответ на вопрос о самочувствии, слот: {awaiting_slot}] {user_text}"
    else:
        model_input = user_text

    history = context.chat_data.setdefault("history", [])
    history.append({"role": "user", "content": model_input})
    history[:] = history[-20:]  # ограничиваем историю

    try:
        response = client.chat.completions.create(
            model=CHEAPAI_MODEL,
            max_tokens=600,
            messages=[{"role": "system", "content": PSYCHOLOGIST_SYSTEM_PROMPT}, *history],
            tools=[REMINDER_TOOL],
        )
        message = response.choices[0].message

        if message.tool_calls:
            confirmations = []
            for tool_call in message.tool_calls:
                if tool_call.function.name == "create_reminder":
                    args = json.loads(tool_call.function.arguments)
                    run_at = _parse_reminder_time(int(args["hour"]), int(args["minute"]))
                    context.job_queue.run_once(
                        send_reminder,
                        when=run_at,
                        chat_id=TARGET_CHAT_ID,
                        data={"text": args["text"]},
                        name=f"reminder_{run_at.isoformat()}",
                    )
                    confirmations.append(
                        f"Хорошо, напомню в {run_at.strftime('%H:%M')}: {args['text']}"
                    )
                    log.info("Запланировано напоминание на %s: %s", run_at, args["text"])
            reply = " ".join(confirmations) if confirmations else "Готово."
        else:
            reply = message.content
    except Exception:
        log.exception("Ошибка вызова cheapai.io API")
        reply = "Прости, у меня сейчас техническая заминка. Попробуй написать ещё раз чуть позже."

    history.append({"role": "assistant", "content": reply})
    await update.message.reply_text(reply)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await reply_to_text(update, context, update.message.text)


def _transcribe_ogg(ogg_bytes: bytes) -> str:
    """Конвертирует голосовое сообщение (ogg/opus) в текст (бесплатно, через Google)."""
    ogg_audio = AudioSegment.from_file(io.BytesIO(ogg_bytes), format="ogg")
    wav_buffer = io.BytesIO()
    ogg_audio.export(wav_buffer, format="wav")
    wav_buffer.seek(0)

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_buffer) as source:
        audio_data = recognizer.record(source)
    return recognizer.recognize_google(audio_data, language="ru-RU")


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    voice_file = await update.message.voice.get_file()
    ogg_bytes = bytes(await voice_file.download_as_bytearray())

    try:
        user_text = _transcribe_ogg(ogg_bytes)
    except sr.UnknownValueError:
        await update.message.reply_text("Не расслышала, что ты сказала — можешь повторить?")
        return
    except Exception:
        log.exception("Ошибка распознавания голоса")
        await update.message.reply_text(
            "Не получилось распознать голосовое. Попробуй написать текстом или прислать ещё раз."
        )
        return

    await reply_to_text(update, context, user_text)


def main():
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    app.job_queue.run_daily(
        send_morning_message,
        time=MORNING_TIME,
        chat_id=TARGET_CHAT_ID,
        name="morning_message",
    )
    app.job_queue.run_daily(
        send_evening_message,
        time=EVENING_TIME,
        chat_id=TARGET_CHAT_ID,
        name="evening_message",
    )
    app.job_queue.run_daily(
        schedule_midday_message,
        time=SCHEDULER_TIME,
        chat_id=TARGET_CHAT_ID,
        name="schedule_midday_message",
    )
    # чтобы обеденное сообщение планировалось и в первый день запуска бота,
    # а не только начиная со следующих суток
    app.job_queue.run_once(
        schedule_midday_message,
        when=1,
        chat_id=TARGET_CHAT_ID,
        name="schedule_midday_message_initial",
    )

    for slot, slot_time in MOOD_CHECKIN_TIMES.items():
        app.job_queue.run_daily(
            send_mood_checkin,
            time=slot_time,
            chat_id=TARGET_CHAT_ID,
            data={"slot": slot},
            name=f"mood_checkin_{slot}",
        )

    app.job_queue.run_daily(
        send_weekly_summary,
        time=WEEKLY_SUMMARY_TIME,
        days=(WEEKLY_SUMMARY_DAY,),
        chat_id=TARGET_CHAT_ID,
        name="weekly_summary",
    )

    log.info("Бот запущен")
    app.run_polling()


if __name__ == "__main__":
    main()
