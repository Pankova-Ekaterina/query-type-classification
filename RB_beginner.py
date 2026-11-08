import re
import pymorphy3


morph = pymorphy3.MorphAnalyzer()

lexicon_beginner = {
    "INTERROGATIVE": ["что", "зачем", "почему", "как", "какой", "какие",
                      "чем", "для чего", "в чём"],
    "SUBJECT": ["кроссовка", "кроссовок", "кед", "обувь", "модель"],
    "NOUN_DOMAIN": ["стиль", "водостойкость", "водостойкий", "подошва",
                    "толщина", "вес", "высокий", "низкий", "характеристика",
                    "параметр", "разница", "отличие", "цвет", "бренд"],
    "ACTION_FUNCTION": ["означать", "значить", "отличаться", "бывать",
                        "делиться", "нужный", "описать"],
    "ADJECTIVE_QUAL": ["обычный", "невысокий", "высокий", "основной"],
    "PARTICLE": ["это", "такой", "же"],
    "PREPOSITION": ["для", "в", "с", "по", "на", "у", "от", "между"],
    "BRAND": ["converse", "vans", "nike", "adidas", "asics", "puma", "dickies",
          "champion", "balenciaga", "louboutin", "numeris", "doublet",
          "mmy", "bbs", "dc", "owens", "margiela", "zanotti", "simons", "golden", "goose"],
}

def normalize(word):
    return morph.parse(word)[0].normal_form

def tokenize(text):
    words = re.findall(r"[а-яёa-z0-9]+", text.lower())
    return [normalize(w) for w in words]

def get_token_class(token):
    for category, words in lexicon_beginner.items():
        if token in words:
            return category
    return None


DEF_ACTIONS  = {"означать", "значить"}
COMP_ACTIONS = {"отличаться", "различаться", "разница", "отличие"}
ENUM_ACTIONS = {"бывать", "делиться", "описать"}

def to_symbol(token):
    """Терминал грамматики: уточнённый класс лексемы."""
    if token in DEF_ACTIONS:
        return "DEF_ACT"
    if token in COMP_ACTIONS:
        return "COMP_ACT"
    if token in ENUM_ACTIONS:
        return "ENUM_ACT"
    return get_token_class(token)

Q = {"INTERROGATIVE"}
PART = {"PARTICLE"}
DEF = {"DEF_ACT"}
COMP = {"COMP_ACT"}
ENUM = {"ENUM_ACT"}
DOMAIN = {"NOUN_DOMAIN", "SUBJECT", "ADJECTIVE_QUAL", "BRAND"}

# Продукции: каждая - последовательность символов в заданном порядке
GRAMMAR = {
    "comparative_q": [
        [Q, COMP, DOMAIN],      # Чем отличаются высокие кроссовки от обычных?
        [Q, DOMAIN, COMP],      # Чем кроссовки отличаются от кед?
    ],
    "enumerative_q": [
        [Q, ENUM, DOMAIN],      # Какие бывают стили кроссовок?
        [Q, DOMAIN, ENUM],      # Какие характеристики кроссовок бывают?
    ],
    "definitional_q": [
        [Q, PART, DOMAIN],      # Что такое водостойкие кроссовки?
        [Q, DEF, DOMAIN],       # Что означает толщина подошвы?
        [DOMAIN, Q, PART],      # Водостойкие кроссовки - что это?
    ],
}

def match_order(symbols, pattern):
    """True, если символы pattern встречаются в symbols в этом порядке
    (между ними могут стоять другие слова)."""
    i = 0
    for s in symbols:
        if i < len(pattern) and s in pattern[i]:
            i += 1
    return i == len(pattern)

def classify_question(text, verbose=False, use_default=True):
    tokens = tokenize(text)
    symbols = [to_symbol(t) for t in tokens]

    if verbose:
        print(f"Токены:  {tokens}")
        print(f"Символы: {symbols}")

    for label, productions in GRAMMAR.items():
        for pattern in productions:
            if match_order(symbols, pattern):
                return label


    if use_default and match_order(symbols, [Q, DOMAIN]):
        return "definitional_q"

    return None

if __name__ == "__main__":
    questions = [
        ("Что такое водостойкие кроссовки?",            "definitional_q"),
        ("Чем отличаются высокие кроссовки от обычных?", "comparative_q"),
        ("Какие бывают стили кроссовок?",               "enumerative_q"),
        ("Что означает толщина подошвы у кроссовок?",   "definitional_q"),
        ("Какие характеристики кроссовок бывают?",      "enumerative_q"),
        ("Что такое подводная лодка?",                  None),
    ]

    correct = 0
    for q, gold in questions:
        result = classify_question(q, verbose=True)
        mark = "OK" if result == gold else "ОШИБКА"
        if result == gold:
            correct += 1
        print(f"Вопрос: {q}")
        print(f"Эталон: {gold} | Получено: {result} -> {mark}\n")

    print(f"Верно: {correct}/{len(questions)}")