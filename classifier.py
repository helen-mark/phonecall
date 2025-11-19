from transformers import pipeline
import json
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

file_path = 'результат_анализа_large.json'
text = json.load(open(file_path, 'r', encoding='utf-8'))
text = text['transcription']['text']
print(text[:300])
# Шаг 2: Классификация текста
classifier = pipeline("text-classification",
                     model="joeddav/xlm-roberta-large-xnli",
                      max_length=512,
                      truncation=True)

# # Для детекции эмоций в русском тексте
# models = [
#     "Aniemore/rubert-tiny2-russian-emotion-detection",  # 6 эмоций
#     # "cointegrated/rubert-tiny2-cedr-emotion-detection", # 7 эмоций
#     # "sismetanin/rubert-ru-emotion-detection",          # 5 эмоций
# ]
#
# # Для интентов/тематики
# intent_models = [
#     "cointegrated/rubert-tiny-toxicity",  # для детекции негатива
#     "UrukHan/t5-russian-summarization",   # можно адаптировать
# ]

# candidate_labels = [
#     "жалоба на качество",
#     "техническая консультация",
#     "опрос удовлетворенности",
#     "оформление заказа",
#     "претензия по доставке",
#     "консультация по услугам"
# ]
result = classifier(text, candidate_labels= ["радость", "злость", "страх", "удивление", "нейтрально"], hypothesis_template="This call is about {}")
# print(f"Вероятность жалобы: {result['scores'][0]:.2%}")
# print(f"Вероятность консультации: {result['scores'][1]:.2%}")
# print(f"Вероятность недовольства: {result['scores'][2]:.2%}")
top_k = 3
top_emotions = list(zip(result['labels'][:top_k], result['scores'][:top_k]))
print(f"Top {top_k} emotions:", top_emotions)