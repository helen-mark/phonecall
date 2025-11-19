import re
from transformers import pipeline
from collections import defaultdict
import json


class ContextAwareRussianAnalyzer:
    def __init__(self):
        # Инициализируем модель для анализа тональности
        self.sentiment_analyzer = pipeline(
            "text-classification",
            model="blanchefort/rubert-base-cased-sentiment",
            return_all_scores=True,
            max_length=512,
            truncation=True
        )

        # Словари для определения контекста
        self.neutral_indicators = {
            'questions': [
                'скажите', 'подскажите', 'объясните', 'помогите', 'прошу',
                'как', 'что', 'где', 'когда', 'почему', 'сколько', 'можно ли'
            ],
            'technical_terms': [
                'счет', 'договор', 'документ', 'номер', 'дата', 'расчет',
                'сумма', 'дней', 'месяц', 'умножение', 'разделить', 'период'
            ],
            'formal_requests': [
                'можно', 'возможно', 'нужно', 'необходимо', 'требуется',
                'следует', 'хотел бы', 'хотела бы'
            ],
            'neutral_phrases': [
                'перед глазами', 'в системе', 'в договоре', 'в документах',
                'по базе', 'по данным', 'у меня', 'у вас'
            ]
        }

        self.true_negative_indicators = {
            'emotional_words': [
                'ужас', 'кошмар', 'безобразие', 'возмущен', 'разочарован',
                'бесит', 'раздражает', 'нервы', 'терпение', 'злюсь', 'сердит'
            ],
            'complaint_patterns': [
                'почему опять', 'опять не', 'снова проблемы', 'опять ошибка',
                'достало уже', 'надоело', 'хуже некуда', 'как такое возможно',
                'это неправильно', 'так нельзя', 'недопустимо'
            ],
            'intensifiers': [
                'очень', 'крайне', 'совершенно', 'абсолютно', 'совсем',
                'полностью', 'просто', 'уже', 'всегда', 'никогда'
            ],
            'negative_verbs': [
                'нарушили', 'испортили', 'подвели', 'обманули', 'забыли',
                'опоздали', 'потеряли', 'сломали', 'не работае'
            ]
        }

    def calculate_sentiment_confidence(self, sentence, sentiment_result):
        """Рассчитываем уверенность с учетом контекста"""
        sentence_lower = sentence.lower()
        top_sentiment = max(sentiment_result, key=lambda x: x['score'])

        base_confidence = top_sentiment['score']

        # Считаем индикаторы
        neutral_indicators_count = 0
        negative_indicators_count = 0

        # Нейтральные индикаторы
        for category, indicators in self.neutral_indicators.items():
            for indicator in indicators:
                if indicator in sentence_lower:
                    neutral_indicators_count += 1

        # Негативные индикаторы
        for category, indicators in self.true_negative_indicators.items():
            for indicator in indicators:
                if indicator in sentence_lower:
                    negative_indicators_count += 1

        # Логика коррекции для негативных оценок
        if top_sentiment['label'] == 'NEGATIVE':
            if neutral_indicators_count > negative_indicators_count:
                # Сильный признак ложного негатива
                correction_factor = min(0.7, neutral_indicators_count * 0.2)
                final_confidence = base_confidence * (1 - correction_factor)
                is_corrected = True
            elif neutral_indicators_count == negative_indicators_count and neutral_indicators_count > 0:
                # Неопределенная ситуация - умеренная коррекция
                final_confidence = base_confidence * 0.7
                is_corrected = True
            else:
                # Вероятно настоящий негатив
                final_confidence = base_confidence
                is_corrected = False
        else:
            # Для позитивных и нейтральных не корректируем
            final_confidence = base_confidence
            is_corrected = False

        return final_confidence, is_corrected

    def analyze_with_context(self, long_text):
        """Основной метод анализа текста"""
        # Разбиваем на предложения
        sentences = re.split(r'[.!?]+', long_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

        analyzed_sentences = []

        for i, sentence in enumerate(sentences):
            try:
                # Анализируем каждое предложение
                result = self.sentiment_analyzer(sentence[:400])[0]
                top_sentiment = max(result, key=lambda x: x['score'])

                # Применяем контекстную коррекцию
                confidence, needs_correction = self.calculate_sentiment_confidence(sentence, result)

                # Определяем финальный сентимент
                if needs_correction and top_sentiment['label'] == 'NEGATIVE':
                    if confidence < 0.5:  # Сильная коррекция
                        final_sentiment = 'NEUTRAL'
                    else:  # Умеренная коррекция
                        final_sentiment = 'NEGATIVE'  # Но с пониженной уверенностью
                else:
                    final_sentiment = top_sentiment['label']

                analyzed_sentences.append({
                    'sentence': sentence,
                    'original_sentiment': top_sentiment['label'],
                    'original_confidence': top_sentiment['score'],
                    'final_sentiment': final_sentiment,
                    'context_confidence': confidence,
                    'needs_correction': needs_correction,
                    'sentence_index': i
                })

            except Exception as e:
                print(f"Ошибка при анализе предложения {i}: {e}")
                continue

        return self.generate_final_report(analyzed_sentences)

    def generate_final_report(self, analyzed_sentences):
        """Генерация итогового отчета"""
        sentiment_stats = {
            'POSITIVE': 0,
            'NEUTRAL': 0,
            'NEGATIVE': 0
        }

        corrections_count = sum(1 for s in analyzed_sentences if s['needs_correction'])

        for sentence_data in analyzed_sentences:
            sentiment_stats[sentence_data['final_sentiment']] += 1

        total_sentences = len(analyzed_sentences)

        if total_sentences > 0:
            complaint_score = sentiment_stats['NEGATIVE'] / total_sentences
        else:
            complaint_score = 0

        return {
            'detailed_analysis': analyzed_sentences,
            'summary': {
                'total_sentences': total_sentences,
                'positive_ratio': sentiment_stats['POSITIVE'] / total_sentences if total_sentences > 0 else 0,
                'neutral_ratio': sentiment_stats['NEUTRAL'] / total_sentences if total_sentences > 0 else 0,
                'negative_ratio': sentiment_stats['NEGATIVE'] / total_sentences if total_sentences > 0 else 0,
                'corrections_applied': corrections_count,
                'correction_rate': corrections_count / total_sentences if total_sentences > 0 else 0,
                'final_complaint_score': complaint_score
            },
            'complaint_level': self.assess_complaint_level(complaint_score)
        }

    def assess_complaint_level(self, complaint_score):
        """Оценка уровня жалобы"""
        if complaint_score > 0.3:
            return "ВЫСОКИЙ_УРОВЕНЬ_ЖАЛОБЫ"
        elif complaint_score > 0.15:
            return "СРЕДНИЙ_УРОВЕНЬ_ЖАЛОБЫ"
        else:
            return "НИЗКИЙ_УРОВЕНЬ_ЖАЛОБЫ"

def simple_usage(text):
    # Инициализируем анализатор
    analyzer = ContextAwareRussianAnalyzer()


    # Анализируем текст
    result = analyzer.analyze_with_context(text)

    return result


file_path = 'результат_анализа_medium_long_talk.json'
text = json.load(open(file_path, 'r', encoding='utf-8'))
text = text['transcription']['text']
print(text[:300])
result = simple_usage(text)
for i in result['detailed_analysis']:
    print(i['sentence'])
    print(i['final_sentiment'])