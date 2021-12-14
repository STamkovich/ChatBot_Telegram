import random
import nltk

BOT_CONFIG = {
    'intents': {
        'Hello': {
            'examples': ['Привет Друг', 'Приветики', 'Здравствуй', 'Здарова', 'Привет'],
            'responses': ['Хай', 'Прив!', 'Хелоу', 'hi']
        },
        'Bye': {
            'examples': ['Пока', 'Покедова', 'Goggd Bye!', 'Bay'],
            'responses': ['До свидания', 'Увидимся', 'Пока-пока)']
        },
        'howdoyoudo': {
            'examples': ['Как дела?', 'Как поживаешь?', 'How are you?'],
            'responses': ['Классно!', 'Я живу в компьютере', 'Прекрасно']
        },
    }
}


def clean(text):
    cleaned_text = ''
    for char in text.lower():
        if char in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяabcdefghijklmnopqrstuvwxyz':
            cleaned_text = cleaned_text + char
    return cleaned_text


def get_intent(text):
    for intent in BOT_CONFIG['intents'].keys():
        for example in BOT_CONFIG['intents'][intent]['examples']:
            clean_example = clean(example)
            clean_text = clean(text)
            if nltk.edit_distance(clean_example, clean_text) / max(len(clean_example), len(clean_text)) < 0.4:
                return intent
    return 'intent not found'


def bot(text):
    intent = get_intent(text)
    if intent == 'intent not found':
        return 'Я ничего не понял'
    else:
        return random.choice(BOT_CONFIG['intents'][intent]['responses'])


while True:
    text = input()
    answer = bot(text)
    print(answer)
