import json, time, math, statistics
import ollama
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from RB_beginner import classify_question
from dataset import DEV, TEST
from collections import defaultdict

DATA = TEST

LABELS = ["definitional_q", "comparative_q", "enumerative_q", "none"]

PROMPT_V1 = """Ты классифицируешь вопросы пользователя о кроссовках.
Определи класс вопроса и ответь ТОЛЬКО в формате JSON:
{
  "класс": "<класс вопроса>"
}

Классы:
- definitional_q — что это такое, что означает характеристика
- comparative_q — чем отличается одно от другого
- enumerative_q — какие бывают, перечисли

Если вопрос не про характеристики кроссовок — верни {"класс": null}"""

PROMPT_V2 = """Ты классифицируешь вопросы пользователя о кроссовках.
Определи класс вопроса и ответь ТОЛЬКО в формате JSON:
{
  "класс": "<класс вопроса>"
}
Классы:
- definitional_q — пользователь просит объяснить, что такое термин или что означает характеристика.
- comparative_q — пользователь спрашивает, чем одно отличается от другого.
- enumerative_q — пользователь просит перечислить, какие бывают варианты или значения.
Если вопрос не относится ни к одному из классов (поиск по фильтрам, просьба о рекомендации, вопрос о программе, вопрос не про кроссовки) — верни {"класс": null}

Примеры:
Вопрос: Что значит высокие кеды?
Ответ: {"класс": "definitional_q"}
Вопрос: Чем отличаются кеды от кроссовок?
Ответ: {"класс": "comparative_q"}
Вопрос: Какие бывают цвета кед?
Ответ: {"класс": "enumerative_q"}
Вопрос: Покажи белые Nike до 8000
Ответ: {"класс": null}
Вопрос: Какая погода в Москве?
Ответ: {"класс": null}"""

PROMPT_V3 = """Ты классифицируешь вопросы пользователя о кроссовках.
Определи класс вопроса и ответь ТОЛЬКО в формате JSON:
{
  "класс": "<класс вопроса>"
}
Классы:
- definitional_q — что это такое, что означает характеристика
- comparative_q — чем отличается одно от другого
- enumerative_q — какие бывают, перечисли
- other — всё остальное: вопрос не про характеристики кроссовков"""

PROMPT_V4 = """Ты классифицируешь вопросы пользователя о кроссовках.
Определи класс вопроса и ответь ТОЛЬКО в формате JSON:
{
  "класс": "<класс вопроса>"
}
Классы:
- definitional_q — пользователь просит объяснить, что такое термин или что означает характеристика.
- comparative_q — пользователь спрашивает, чем одно отличается от другого.
- enumerative_q — пользователь просит перечислить, какие бывают варианты или значения.
- other — всё остальное (поиск по фильтрам, просьба о рекомендации, вопрос о программе, вопрос не про кроссовки).

Примеры:
Вопрос: Что значит высокие кеды?
Ответ: {"класс": "definitional_q"}
Вопрос: Чем отличаются кеды от кроссовок?
Ответ: {"класс": "comparative_q"}
Вопрос: Какие бывают цвета кед?
Ответ: {"класс": "enumerative_q"}
Вопрос: Покажи белые Nike до 8000
Ответ: {"класс": "other"}
Вопрос: Какая погода в Москве?
Ответ: {"класс": "other"}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "класс": {"type": "string",
                  "enum": ["definitional_q", "comparative_q", "enumerative_q", "other"]}
    },
    "required": ["класс"],
}

PROMPTS = [
    ("V1", PROMPT_V1, None),
    ("V2", PROMPT_V2, None),
    ("V3", PROMPT_V3, SCHEMA),
    ("V4", PROMPT_V4, SCHEMA),
]
MODELS = ["qwen3.5:0.8b", "qwen3.5:2b"]


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d

def parse_llm(raw):
    try:
        label = json.loads(raw)["класс"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return "invalid"                # сломанный формат = отдельная ошибка
    if label in (None, "other", "null"):
        return "none"
    return label if label in LABELS[:3] else "invalid"

def make_llm_predictor(model, system_prompt, schema=None, temperature=0.0):
    def predict(q):
        try:
            resp = ollama.chat(
                model=model,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": q}],
                think=False, format=schema,
                options={"temperature": temperature, "seed": 42},
            )
            raw = resp["message"]["content"]
            predict.raws.append(raw)
            return parse_llm(raw)
        except Exception:
            predict.raws.append("СИСТЕМНАЯ_ОШИБКА_OLLAMA")
            return "invalid"
    predict.raws = []
    return predict

def evaluate(name, predict, data):
    gold = [g for _, g, _ in data]
    preds, times = [], []
    for q, _, _ in data:
        t = time.perf_counter()
        preds.append(predict(q))
        times.append(time.perf_counter() - t)
    n = len(gold)
    ok = sum(p == g for p, g in zip(preds, gold))
    lo, hi = wilson(ok, n)

    by_type = defaultdict(lambda: [0, 0])
    for (_, g, t), p in zip(data, preds):
        by_type[t][1] += 1
        by_type[t][0] += (p == g)

    print(f"\n=== {name} ===")
    print(f"accuracy {ok/n:.3f}  95% ДИ [{lo:.3f}; {hi:.3f}]  "
          f"invalid {preds.count('invalid')}/{n}  "
          f"время ср. {statistics.mean(times):.4f} с, медиана {statistics.median(times):.4f} с")
    print("по типам:", {t: f"{k}/{m}" for t, (k, m) in by_type.items()})
    lab = LABELS + ["invalid"]
    print(classification_report(gold, preds, labels=lab, zero_division=0))
    print(confusion_matrix(gold, preds, labels=lab))

    print("Ошибки:")
    for (q, g, t), p in zip(data, preds):
        if p != g:
            print(f"  [{t}] {q!r}: эталон={g}, получено={p}")

    raws = getattr(predict, "raws", None)
    if raws:
        print("Сырые ответы для INVALID:")
        for (q, _, _), p, r in zip(data, preds, raws):
            if p == "invalid":
                print(f"  INVALID {q!r}: {r!r}")

    return {
        "name": name, "acc": ok / n, "lo": lo, "hi": hi,
        "f1": f1_score(gold, preds, labels=LABELS, average="macro", zero_division=0),
        "invalid": preds.count("invalid") / n,
        "time": statistics.mean(times),
    }

if __name__ == "__main__":
    results = []
    results.append(
        evaluate("rule-based, default вкл", lambda q: classify_question(q, use_default=True, verbose=False) or "none",
                 DATA))
    results.append(
        evaluate("rule-based, default выкл", lambda q: classify_question(q, use_default=False, verbose=False) or "none",
                 DATA))

    for model in MODELS:
        for pname, prompt, schema in PROMPTS:
            results.append(evaluate(f"{model}, {pname}",
                                    make_llm_predictor(model, prompt, schema), DATA))

    print("\n\n===== СВОДНАЯ ТАБЛИЦА =====")
    print(f"{'конфигурация':<24}{'accuracy':>9}{'95% ДИ':>18}{'macro-F1':>10}{'invalid':>9}{'время, с':>10}")
    for r in results:
        print(f"{r['name']:<24}{r['acc']:>9.3f}  [{r['lo']:.2f}; {r['hi']:.2f}]"
              f"{r['f1']:>10.3f}{r['invalid']:>9.1%}{r['time']:>10.3f}")