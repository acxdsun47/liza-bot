"""
Telegram-бот для Лизы.

Возможности:
1. Три тёплых сообщения в день.
2. Пять check-in о самочувствии в день + сохранение ответов.
3. Еженедельная сводка по воскресеньям.
4. Поддерживающий AI-диалог и короткие практики при тревоге/стрессе.
5. Отдельный режим «мне тревожно / паническая атака».
6. Напоминания: «в 15:00», «завтра в 9:30», «через 40 минут»,
   «каждый день в 18:00».
7. Напоминания сохраняются в reminders.json и восстанавливаются после перезапуска.
8. Голосовые сообщения распознаются через Google Speech Recognition.

Переменные окружения:
    BOT_TOKEN
    CHEAPAI_API_KEY
    TARGET_CHAT_ID
    TIMEZONE             (по умолчанию Europe/Moscow)
    CHEAPAI_MODEL        (по умолчанию claude-sonnet-4-6)
    MOOD_LOG_PATH        (по умолчанию mood_log.jsonl)
    REMINDER_LOG_PATH    (по умолчанию reminders.json)
"""

import io
import json
import logging
import os
import random
import re
import uuid
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

from openai import OpenAI
import speech_recognition as sr
from pydub import AudioSegment
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("liza_bot")

BOT_TOKEN = os.environ["BOT_TOKEN"]
CHEAPAI_API_KEY = os.environ["CHEAPAI_API_KEY"]
TARGET_CHAT_ID = int(os.environ["TARGET_CHAT_ID"])

TIMEZONE = ZoneInfo(os.environ.get("TIMEZONE", "Europe/Moscow"))

MORNING_TIME = dtime(hour=8, minute=0, tzinfo=TIMEZONE)
EVENING_TIME = dtime(hour=23, minute=0, tzinfo=TIMEZONE)

MIDDAY_WINDOW_START = dtime(hour=12, minute=0, tzinfo=TIMEZONE)
MIDDAY_WINDOW_END = dtime(hour=14, minute=0, tzinfo=TIMEZONE)
SCHEDULER_TIME = dtime(hour=0, minute=5, tzinfo=TIMEZONE)

MOOD_CHECKIN_TIMES = {
    "утро": dtime(hour=9, minute=0, tzinfo=TIMEZONE),
    "перед обедом": dtime(hour=11, minute=30, tzinfo=TIMEZONE),
    "обед": dtime(hour=13, minute=30, tzinfo=TIMEZONE),
    "после обеда": dtime(hour=16, minute=30, tzinfo=TIMEZONE),
    "ночь": dtime(hour=21, minute=30, tzinfo=TIMEZONE),
}

WEEKLY_SUMMARY_TIME = dtime(hour=20, minute=0, tzinfo=TIMEZONE)
WEEKLY_SUMMARY_DAY = 6

MOOD_LOG_PATH = Path(os.environ.get("MOOD_LOG_PATH", "mood_log.jsonl"))
REMINDER_LOG_PATH = Path(os.environ.get("REMINDER_LOG_PATH", "reminders.json"))

MOOD_RESPONSE_TTL = timedelta(hours=2)

client = OpenAI(
    api_key=CHEAPAI_API_KEY,
    base_url="https://cheapai.io/v1",
)
CHEAPAI_MODEL = os.environ.get("CHEAPAI_MODEL", "claude-sonnet-4-6")

MORNING_MESSAGE = "Доброе утро, любимая🩷"
MIDDAY_MESSAGE = "Я тебя люблю, лизочка"
EVENING_MESSAGE = "Самых сладких снов, любимая"

MOOD_CHECKIN_QUESTIONS = {
    "утро": "Доброе утро ещё раз 🌤 Как ты сегодня, как настроение с утра?",
    "перед обедом": "Как ты сейчас, перед обедом? Как самочувствие?",
    "обед": "Обеденный чек-ин 🍽 Как ты себя чувствуешь?",
    "после обеда": "Как прошла вторая половина дня, как ты себя ощущаешь?",
    "ночь": "Перед сном — как прошёл день в целом, как ты сейчас?",
}

PSYCHOLOGIST_SYSTEM_PROMPT = """\
Ты — тёплый, внимательный и бережный собеседник для человека по имени Лиза.
Ты не врач и не психотерапевт. Твоя задача — поддержать, помочь назвать эмоции,
спокойно разобраться в ситуации и предложить небольшую безопасную практику,
если она уместна.

Правила:
- Не ставь диагнозы и не утверждай, что точно знаешь причину симптомов.
- Не обесценивай чувства фразами вроде «не переживай» или «всё ерунда».
- Пиши по-русски, тепло, естественно и коротко. Не превращай каждый ответ в длинную лекцию.
- Если Лиза просто хочет поговорить — разговаривай, а не навязывай упражнения.
- При тревоге можно предложить одну практику за раз: медленное дыхание,
  заземление 5-4-3-2-1, ощущение стоп на полу, расслабление мышц,
  переключение внимания на окружающие предметы.
- При панике не утверждай категорически, что любые симптомы безопасны:
  сильная боль в груди, обморок, выраженная одышка, новые или необычные симптомы
  требуют медицинской оценки. Если человек в непосредственной опасности,
  предложи обратиться за экстренной помощью.
- Если появляются мысли о самоубийстве или самоповреждении, спокойно выясни,
  есть ли непосредственная опасность, предложи не оставаться одной и обратиться
  к близкому человеку/экстренной или кризисной помощи. Не оставляй это без внимания.
- Не выдавай себя за психолога.

Если сообщение начинается с "[Ответ на вопрос о самочувствии, слот: ...]",
ответь тепло и по-человечески, не упоминай технические детали и слово "слот".
"""

CRISIS_KEYWORDS = (
    "не хочу жить",
    "не хочу больше жить",
    "хочу умереть",
    "убить себя",
    "покончить с собой",
    "самоубий",
    "суицид",
    "причинить себе вред",
    "навредить себе",
    "порезать себя",
)

ANXIETY_KEYWORDS = (
    "тревож",
    "паничес",
    "панич",
    "паническая атака",
    "паническая",
    "стресс",
    "накрыло",
    "не могу успоко",
    "сильное сердцебиение",
)

PRACTICE_KEYWORDS = (
    "успоко",
    "практик",
    "упражнен",
    "что делать",
    "помоги мне",
)


def now_local() -> datetime:
    return datetime.now(TIMEZONE)


def atomic_write_json(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def log_mood_entry(slot: str, text: str, source: str = "text", mood_label: str | None = None):
    entry = {
        "timestamp": now_local().isoformat(),
        "slot": slot,
        "text": text,
        "source": source,
    }
    if mood_label:
        entry["mood_label"] = mood_label

    with MOOD_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_mood_entries_since(since: datetime) -> list:
    if not MOOD_LOG_PATH.exists():
        return []

    entries = []
    with MOOD_LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                timestamp = datetime.fromisoformat(entry["timestamp"])
                if timestamp >= since:
                    entries.append(entry)
            except (json.JSONDecodeError, KeyError, ValueError):
                log.warning("Пропущена повреждённая запись mood_log")
    return entries


def load_reminders() -> list[dict]:
    if not REMINDER_LOG_PATH.exists():
        return []

    try:
        data = json.loads(REMINDER_LOG_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        log.exception("Не удалось загрузить reminders.json")
        return []


def save_reminders(reminders: list[dict]):
    atomic_write_json(REMINDER_LOG_PATH, reminders)


def add_reminder(run_at: datetime, text: str, repeat_daily: bool = False) -> dict:
    reminder = {
        "id": uuid.uuid4().hex,
        "run_at": run_at.isoformat(),
        "text": text,
        "repeat_daily": repeat_daily,
        "created_at": now_local().isoformat(),
    }
    reminders = load_reminders()
    reminders.append(reminder)
    save_reminders(reminders)
    return reminder


def remove_reminder(reminder_id: str):
    reminders = load_reminders()
    reminders = [r for r in reminders if r.get("id") != reminder_id]
    save_reminders(reminders)


def _parse_clock(hour: int, minute: int) -> tuple[int, int] | None:
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour, minute


def _candidate_at(hour: int, minute: int, day_offset: int = 0) -> datetime:
    base = now_local() + timedelta(days=day_offset)
    return base.replace(
        hour=hour,
        minute=minute,
        second=0,
        microsecond=0,
    )


def _parse_reminder_time(hour: int, minute: int, day_offset: int = 0) -> datetime:
    """Возвращает указанное время сегодня/завтра."""
    parsed = _parse_clock(hour, minute)
    if parsed is None:
        raise ValueError("Некорректное время")

    candidate = _candidate_at(hour, minute, day_offset)

    if day_offset == 0 and candidate <= now_local():
        candidate += timedelta(days=1)

    return candidate


def parse_reminder(text: str) -> dict | None:
    """
    Поддерживаемые формы:
      напомни в 15:00 сделать задание
      напомни в 20/00 забрать вб
      напомни завтра в 09:30 позвонить
      напомни сегодня в 18:00 купить молоко
      напомни через 40 минут выйти
      каждый день в 18:00 принимать душ

    Возвращает:
      {"run_at": datetime, "text": str, "repeat_daily": bool}
    """
    original = text.strip()
    lower = original.lower()

    repeat_daily = bool(
        re.search(r"\b(?:каждый день|ежедневно|каждый вечер|каждое утро)\b", lower)
    )

    # Относительное время: «через 40 минут/час/2 часа»
    relative = re.search(
        r"(?:напомни\s+)?через\s+(\d+)\s*"
        r"(мин(?:ут[а-я]*)?|час(?:а|ов)?|ч)\s+(.+)$",
        lower,
        re.IGNORECASE,
    )
    if relative:
        amount = int(relative.group(1))
        unit = relative.group(2)
        reminder_text = original[relative.start(3):].strip()

        if "час" in unit or unit == "ч":
            delta = timedelta(hours=amount)
        else:
            delta = timedelta(minutes=amount)

        if amount <= 0:
            return None

        return {
            "run_at": now_local() + delta,
            "text": reminder_text,
            "repeat_daily": False,
        }

    # Время HH:MM, HH.MM или HH/MM.
    time_match = re.search(r"\b(\d{1,2})\s*[:./]\s*(\d{2})\b", lower)
    if not time_match:
        return None

    hour = int(time_match.group(1))
    minute = int(time_match.group(2))
    if _parse_clock(hour, minute) is None:
        return None

    before = lower[:time_match.start()]
    after_original = original[time_match.end():].strip()
    after_lower = lower[time_match.end():].strip()

    # Только фразы, похожие на просьбу о напоминании.
    has_reminder_marker = any(
        marker in lower
        for marker in (
            "напомни",
            "напомнить",
            "напоминай",
            "поставь напоминание",
            "каждый день",
            "ежедневно",
        )
    )
    if not has_reminder_marker:
        return None

    day_offset = 1 if "завтра" in before else 0
    if "сегодня" in before:
        day_offset = 0

    # Для «напомни в 15:00 сделать X» всё после времени — задача.
    # Для «напомни сделать X в 15:00» берём текст до времени.
    if after_original:
        reminder_text = after_original
    else:
        prefix = original[:time_match.start()]
        prefix = re.sub(
            r"\b(?:напомни|напомнить|напоминай|поставь|мне|сегодня|завтра|каждый день|ежедневно|в)\b",
            " ",
            prefix,
            flags=re.IGNORECASE,
        )
        reminder_text = re.sub(r"\s+", " ", prefix).strip(" ,.-")

    if not reminder_text:
        return None

    # Повторяющееся напоминание всегда привязываем к ближайшему будущему времени.
    if repeat_daily:
        run_at = _candidate_at(hour, minute, 0)
        if run_at <= now_local():
            run_at += timedelta(days=1)
    else:
        run_at = _parse_reminder_time(hour, minute, day_offset)

    return {
        "run_at": run_at,
        "text": reminder_text,
        "repeat_daily": repeat_daily,
    }


def format_reminder_confirmation(reminder: dict) -> str:
    run_at = datetime.fromisoformat(reminder["run_at"])
    if reminder.get("repeat_daily"):
        return (
            f"Хорошо ❤️ Буду напоминать каждый день в "
            f"{run_at.strftime('%H:%M')}: {reminder['text']}"
        )

    day_text = "сегодня"
    if run_at.date() != now_local().date():
        day_text = "завтра"

    return f"Хорошо ❤️ Напомню {day_text} в {run_at.strftime('%H:%M')}: {reminder['text']}"


async def schedule_reminder_job(
    context: ContextTypes.DEFAULT_TYPE,
    reminder: dict,
):
    run_at = datetime.fromisoformat(reminder["run_at"])
    delay = max(1, (run_at - now_local()).total_seconds())

    context.job_queue.run_once(
        send_reminder,
        when=delay,
        chat_id=TARGET_CHAT_ID,
        data=reminder,
        name=f"reminder_{reminder['id']}",
    )


async def send_reminder(context: ContextTypes.DEFAULT_TYPE):
    reminder = context.job.data
    text = reminder["text"]

    await context.bot.send_message(
        chat_id=TARGET_CHAT_ID,
        text=f"⏰ Напоминаю: {text}",
    )

    if reminder.get("repeat_daily"):
        run_at = datetime.fromisoformat(reminder["run_at"]) + timedelta(days=1)
        reminder["run_at"] = run_at.isoformat()

        reminders = load_reminders()
        for item in reminders:
            if item.get("id") == reminder.get("id"):
                item["run_at"] = reminder["run_at"]
                break
        save_reminders(reminders)

        await schedule_reminder_job(context, reminder)
        log.info("Повторное напоминание перенесено на %s", run_at)
    else:
        remove_reminder(reminder["id"])
        log.info("Отправлено одноразовое напоминание: %s", text)


async def restore_reminders(application: Application):
    """Восстанавливает напоминания после перезапуска бота."""
    reminders = load_reminders()
    changed = False
    now = now_local()

    for reminder in reminders[:]:
        try:
            run_at = datetime.fromisoformat(reminder["run_at"])
        except (KeyError, ValueError):
            reminders.remove(reminder)
            changed = True
            continue

        if run_at <= now:
            if reminder.get("repeat_daily"):
                while run_at <= now:
                    run_at += timedelta(days=1)
                reminder["run_at"] = run_at.isoformat()
                changed = True
            else:
                # Одноразовое напоминание уже пропущено во время выключения.
                reminders.remove(reminder)
                changed = True
                continue

        await schedule_reminder_job(application, reminder)

    if changed:
        save_reminders(reminders)

    log.info("Восстановлено напоминаний: %d", len(reminders))


def is_crisis_text(text: str) -> bool:
    lower = text.lower()
    return any(keyword in lower for keyword in CRISIS_KEYWORDS)


def is_anxiety_text(text: str) -> bool:
    lower = text.lower()
    return any(keyword in lower for keyword in ANXIETY_KEYWORDS)


def is_practice_request(text: str) -> bool:
    lower = text.lower()
    return any(keyword in lower for keyword in PRACTICE_KEYWORDS)


async def send_morning_message(context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=TARGET_CHAT_ID, text=MORNING_MESSAGE)


async def send_midday_message(context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=TARGET_CHAT_ID, text=MIDDAY_MESSAGE)


async def send_evening_message(context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=TARGET_CHAT_ID, text=EVENING_MESSAGE)


def _random_time_today(start: dtime, end: dtime) -> datetime:
    today = now_local().date()
    start_dt = datetime.combine(today, start.replace(tzinfo=None), tzinfo=TIMEZONE)
    end_dt = datetime.combine(today, end.replace(tzinfo=None), tzinfo=TIMEZONE)
    delta_seconds = int((end_dt - start_dt).total_seconds())
    offset = random.randint(0, max(delta_seconds, 0))
    return start_dt + timedelta(seconds=offset)


async def schedule_midday_message(context: ContextTypes.DEFAULT_TYPE):
    run_at = _random_time_today(MIDDAY_WINDOW_START, MIDDAY_WINDOW_END)

    if run_at <= now_local():
        run_at = now_local() + timedelta(minutes=1)

    context.job_queue.run_once(
        send_midday_message,
        when=max(1, (run_at - now_local()).total_seconds()),
        chat_id=TARGET_CHAT_ID,
        name="midday_love_message_today",
    )


async def send_mood_checkin(context: ContextTypes.DEFAULT_TYPE):
    slot = context.job.data["slot"]

    # Состояние хранится в chat_data, а не в глобальном bot_data.
    context.chat_data["awaiting_mood"] = {
        "slot": slot,
        "created_at": now_local().isoformat(),
    }

    question = MOOD_CHECKIN_QUESTIONS.get(
        slot,
        "Как ты себя чувствуешь?",
    )

    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("😊 Хорошо", callback_data=f"mood:{slot}:good"),
                InlineKeyboardButton("😐 Нормально", callback_data=f"mood:{slot}:ok"),
            ],
            [
                InlineKeyboardButton("😔 Плохо", callback_data=f"mood:{slot}:bad"),
                InlineKeyboardButton("😰 Тревожно", callback_data=f"mood:{slot}:anxious"),
            ],
        ]
    )

    await context.bot.send_message(
        chat_id=TARGET_CHAT_ID,
        text=question + "\n\nМожно выбрать вариант ниже или просто написать своими словами.",
        reply_markup=keyboard,
    )
    log.info("Отправлен check-in: %s", slot)


async def handle_mood_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data or ""
    parts = data.split(":", 2)
    if len(parts) != 3:
        return

    _, slot, mood_label = parts

    labels = {
        "good": "😊 Хорошо",
        "ok": "😐 Нормально",
        "bad": "😔 Плохо",
        "anxious": "😰 Тревожно",
    }

    label = labels.get(mood_label, mood_label)
    log_mood_entry(slot, label, source="button", mood_label=label)

    context.chat_data.pop("awaiting_mood", None)

    await query.edit_message_reply_markup(reply_markup=None)
    await query.message.reply_text(
        f"Записала ❤️ {label}\nЕсли хочешь, можешь написать пару слов о том, что повлияло на состояние."
    )


async def send_weekly_summary(context: ContextTypes.DEFAULT_TYPE):
    since = now_local() - timedelta(days=7)
    entries = load_mood_entries_since(since)

    if not entries:
        await context.bot.send_message(
            chat_id=TARGET_CHAT_ID,
            text=(
                "📋 За эту неделю пока нет записей о самочувствии. "
                "Если захочешь, начнём собирать их с сегодняшнего дня ❤️"
            ),
        )
        return

    entries_text = "\n".join(
        f"- {datetime.fromisoformat(e['timestamp']).strftime('%d.%m %H:%M')} "
        f"({e.get('slot', '—')}): {e.get('text', '')}"
        for e in entries
    )

    summary_prompt = (
        "Составь короткую, тёплую и полезную недельную сводку самочувствия Лизы. "
        "Опирайся только на предоставленные записи. Укажи: "
        "1) как менялось состояние; "
        "2) какие эмоции/темы повторялись; "
        "3) что было самым тяжёлым или, наоборот, поддерживающим; "
        "4) какие вопросы имеет смысл обсудить с близким человеком или психологом. "
        "Не ставь диагнозов и не придумывай фактов. 6-10 предложений, по-русски.\n\n"
        "Записи:\n" + entries_text
    )

    try:
        response = client.chat.completions.create(
            model=CHEAPAI_MODEL,
            max_tokens=700,
            messages=[{"role": "user", "content": summary_prompt}],
        )
        summary = (response.choices[0].message.content or "").strip()
    except Exception:
        log.exception("Ошибка cheapai.io при недельной сводке")
        summary = (
            "Автоматически составить сводку не получилось. "
            "Ниже оставляю записи за неделю:\n\n" + entries_text
        )

    await context.bot.send_message(
        chat_id=TARGET_CHAT_ID,
        text="📋 Сводка самочувствия за неделю:\n\n" + summary,
    )


def mood_state_is_fresh(context: ContextTypes.DEFAULT_TYPE) -> bool:
    state = context.chat_data.get("awaiting_mood")
    if not state:
        return False

    try:
        created_at = datetime.fromisoformat(state["created_at"])
    except (KeyError, ValueError):
        context.chat_data.pop("awaiting_mood", None)
        return False

    if now_local() - created_at > MOOD_RESPONSE_TTL:
        context.chat_data.pop("awaiting_mood", None)
        return False

    return True


async def generate_ai_reply(context: ContextTypes.DEFAULT_TYPE, model_input: str) -> str:
    history = context.chat_data.setdefault("history", [])
    history.append({"role": "user", "content": model_input})
    history[:] = history[-20:]

    response = client.chat.completions.create(
        model=CHEAPAI_MODEL,
        max_tokens=600,
        messages=[
            {"role": "system", "content": PSYCHOLOGIST_SYSTEM_PROMPT},
            *history,
        ],
    )

    reply = (response.choices[0].message.content or "").strip()
    if not reply:
        reply = "Я здесь ❤️ Расскажи мне чуть подробнее, что сейчас происходит."

    history.append({"role": "assistant", "content": reply})
    history[:] = history[-20:]
    return reply


async def send_crisis_support(update: Update):
    await update.message.reply_text(
        "Лиза, я рядом ❤️ Мне важно сейчас не оставлять тебя с этим одной.\n\n"
        "Если есть риск, что ты можешь причинить себе вред прямо сейчас, "
        "пожалуйста, отойди от всего, чем можно навредить себе, и позови человека, "
        "которому доверяешь, чтобы он побыл рядом. При непосредственной опасности "
        "обратись в местную экстренную службу.\n\n"
        "Если непосредственной опасности нет, можешь написать мне, что именно сейчас "
        "происходит. Я побуду с тобой и помогу пройти ближайшие несколько минут."
    )


async def send_anxiety_practice(update: Update):
    await update.message.reply_text(
        "Давай сначала просто немного снизим напряжение. Не нужно делать идеально ❤️\n\n"
        "🌿 **Заземление 5–4–3–2–1**\n"
        "1. Назови 5 вещей, которые видишь.\n"
        "2. 4 вещи, которых можешь коснуться.\n"
        "3. 3 звука, которые слышишь.\n"
        "4. 2 запаха или ощущения запаха.\n"
        "5. 1 вкус или ощущение во рту.\n\n"
        "Потом сделай несколько спокойных, обычных вдохов и длиннее выдыхай, "
        "не заставляя себя дышать глубоко.\n\n"
        "Когда закончишь, напиши мне одним словом: **«готово»** — и посмотрим, "
        "стало ли хоть немного легче."
    )


async def reply_to_text(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user_text: str,
    source: str = "text",
):
    if not user_text.strip():
        return

    # ВАЖНО: сначала обрабатываем напоминание и сразу выходим.
    # Поэтому «мне плохо» после напоминания больше не превращается в повтор напоминания.
    reminder_data = parse_reminder(user_text)
    if reminder_data:
        reminder = add_reminder(
            reminder_data["run_at"],
            reminder_data["text"],
            reminder_data["repeat_daily"],
        )
        await schedule_reminder_job(context, reminder)

        # Если в этот момент был активный mood check-in, просьба о напоминании
        # не записывается как ответ о самочувствии.
        await update.message.reply_text(format_reminder_confirmation(reminder))
        return

    # Сохраняем свободный текст как ответ на недавний check-in.
    model_input = user_text
    if mood_state_is_fresh(context):
        state = context.chat_data.pop("awaiting_mood")
        slot = state["slot"]
        log_mood_entry(slot, user_text, source=source)
        model_input = f"[Ответ на вопрос о самочувствии, слот: {slot}] {user_text}"

    if is_crisis_text(user_text):
        await send_crisis_support(update)
        return

    # Быстрый локальный сценарий: не надо тратить API-запрос на простую практику.
    if is_anxiety_text(user_text) and (
        is_practice_request(user_text)
        or "паничес" in user_text.lower()
        or "панич" in user_text.lower()
    ):
        await send_anxiety_practice(update)
        return

    try:
        reply = await generate_ai_reply(context, model_input)
    except Exception:
        log.exception("Ошибка вызова cheapai.io API")
        reply = (
            "Прости, у меня сейчас техническая заминка ❤️ "
            "Но я здесь. Попробуй написать ещё раз через минутку."
        )

    await update.message.reply_text(reply)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await reply_to_text(
        update,
        context,
        update.message.text or "",
        source="text",
    )


def _transcribe_ogg(ogg_bytes: bytes) -> str:
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
        await update.message.reply_text(
            "Не расслышала, что ты сказала — можешь повторить? ❤️"
        )
        return
    except Exception:
        log.exception("Ошибка распознавания голоса")
        await update.message.reply_text(
            "Не получилось распознать голосовое. "
            "Попробуй написать текстом или прислать ещё раз."
        )
        return

    await reply_to_text(
        update,
        context,
        user_text,
        source="voice",
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет, Лизочка 💛 Я здесь.\n\n"
        "Можешь просто писать или отправлять голосовые. "
        "Если тревожно — скажи мне, и я помогу пройти через это.\n\n"
        "Для напоминания можно написать, например:\n"
        "• «напомни в 15:00 сделать задание»\n"
        "• «напомни завтра в 09:30 позвонить»\n"
        "• «напомни через 40 минут выйти»\n"
        "• «каждый день в 18:00 принять лекарство»\n\n"
        "А ещё я буду несколько раз в день спрашивать, как ты себя чувствуешь, "
        "и по воскресеньям присылать недельную сводку ❤️"
    )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    log.exception(
        "Необработанная ошибка Telegram",
        exc_info=context.error,
    )


async def post_init(application: Application):
    await restore_reminders(application)
    log.info("Напоминания восстановлены после запуска")


def main():
    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(handle_mood_button, pattern=r"^mood:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_error_handler(error_handler)

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

    # Планируем обеденное сообщение сразу после запуска.
    app.job_queue.run_once(
        schedule_midday_message,
        when=2,
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

    log.info(
        "Бот запущен. Часовой пояс: %s | chat_id: %s",
        TIMEZONE,
        TARGET_CHAT_ID,
    )
    app.run_polling()


if __name__ == "__main__":
    main()
