import random
import json  # загрузка датасета
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # векторайзер
from sklearn.linear_model import LogisticRegression  # классификатор
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split  # разбиение на тренировочную и тестовую выборки


# импортируем библиотеки


def clean(text):
    cleaned_text = ''
    for char in text.lower():
        if char in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяabcdefghijklmnopqrstuvwxyz ':
            cleaned_text = cleaned_text + char
    return cleaned_text


with open('/home/sergey/PycharmProjects/ChatBot_Telegram/BOT_CONFIG.json') as f:
    BOT_CONFIG = json.load(f)
len(BOT_CONFIG['intents'])  # загрузка датасета, 147 интентов

corpus = []
y = []
for intent in BOT_CONFIG['intents'].keys():
    for example in BOT_CONFIG['intents'][intent]['examples']:
        corpus.append(example)
        y.append(intent)
len(corpus), len(y)  # список текстов и список их интентов в том одинаковом порядке

corpus_train, corpus_test, y_train, y_test = train_test_split(corpus, y, test_size=0.2,
                                                              random_state=42)  # разбиение на тренировочную и тестовую выборки

vectorizer = CountVectorizer(ngram_range=(1, 5),
                             analyzer='char_wb')  # TfidfVectorizer() preprocessor=clean # векторизуем тексты (на тренировочной создаем словарь, к тестовой только применяем)
X_train = vectorizer.fit_transform(corpus_train)
X_test = vectorizer.transform(corpus_test)
# print(len(vectorizer.get_feature_names_out()))  # длина словаря - 425 слов

model = RandomForestClassifier(n_estimators=300)  # LogisticRegression(max_iter=200)#
model.fit(X_train, y_train)  # учим модель на тренировочной части
print(model.score(X_train, y_train), model.score(X_test, y_test))  # оцениваем качество на тренировочной и на тестовой


def clean(text):
    cleaned_text = ''
    for char in text.lower():
        if char in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя':
            cleaned_text = cleaned_text + char
    return cleaned_text


def get_intent_by_model(text):
    return model.predict(vectorizer.transform([text]))[0]


def bot(text):
    intent = get_intent_by_model(text)
    return random.choice(BOT_CONFIG['intents'][intent]['responses'])


while True:
    text = input()
    if text == 'пока':
        print('И тебе пока')
        break
    answer = bot(text)
    print(answer)
