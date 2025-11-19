import ollama
import chromadb
from sentence_transformers import SentenceTransformer
import re
from typing import List, Dict, Optional
import numpy as np


class EnhancedUniversalAnalyzer(UniversalCallAnalyzer):
    def ask_with_evidence(self, question: str) -> Dict:
        """Возвращает ответ с доказательствами"""

        relevant_calls = self._find_relevant_calls(question)

        if not relevant_calls:
            return {
                "answer": "В базе данных нет информации для ответа на этот вопрос.",
                "evidence": [],
                "confidence": 0
            }

        analysis = self._analyze_with_evidence(question, relevant_calls)

        return {
            "answer": analysis,
            "evidence": relevant_calls[:3],  # Топ-3 доказательства
            "confidence": self._calculate_confidence(question, relevant_calls),
            "sources_count": len(relevant_calls)
        }

    def _analyze_with_evidence(self, question: str, relevant_calls: List[str]) -> str:
        """Анализ с явным указанием доказательств"""

        context = "\n".join([
            f"Доказательство {i + 1}: {call}"
            for i, call in enumerate(relevant_calls[:5])  # Ограничиваем для читаемости
        ])

        prompt = f"""
ВОПРОС: {question}

НАЙДЕННЫЕ ДАННЫЕ:
{context}

Проанализируй данные и ответь:
1. Есть ли прямая информация для ответа?
2. Если есть - приведи конкретные цитаты
3. Если нет - укажи, что именно не найдено
4. Будь максимально точен

АНАЛИЗ И ОТВЕТ:
"""
        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            options={'temperature': 0.1}
        )

        return response['response']

    def _calculate_confidence(self, question: str, relevant_calls: List[str]) -> float:
        """Оценивает уверенность в ответе"""
        if not relevant_calls:
            return 0.0

        # Простая эвристика: чем больше релевантных результатов, тем выше уверенность
        max_results = 10
        confidence = min(len(relevant_calls) / max_results, 1.0)

        # Увеличиваем уверенность при точных совпадениях
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        for call in relevant_calls[:3]:
            call_words = set(re.findall(r'\b\w+\b', call.lower()))
            if question_words.intersection(call_words):
                confidence += 0.2

        return min(confidence, 1.0)


# Тестирование улучшенной версии
def test_enhanced_analyzer():
    analyzer = EnhancedUniversalAnalyzer()

    # Загрузка тестовых данных
    call_texts = [...]  # ваши звонки

    analyzer.index_calls(call_texts)

    hard_questions = [
        "Называл ли менеджер клиента дураком вчера?",
        "Кто из клиентов жаловался именно на Петрова?",
        "Были ли угрозы в адрес клиентов?",
    ]

    for question in hard_questions:
        print(f"\n🔍 Анализируем: {question}")
        result = analyzer.ask_with_evidence(question)

        print(f"✅ Ответ: {result['answer']}")
        print(f"📊 Уверенность: {result['confidence']:.2f}")
        print(f"📎 Источников: {result['sources_count']}")
        if result['evidence']:
            print("🔎 Доказательства:")
            for i, evidence in enumerate(result['evidence'][:2], 1):
                print(f"   {i}. {evidence[:100]}...")

class UniversalCallAnalyzer:
    def __init__(self, model_name: str = "llama3:8b"):
        self.model_name = model_name
        self.client = ollama.Client()

        # Универсальная модель для эмбеддингов
        self.embedding_model = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        )

        # Векторная БД для семантического поиска
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(
            name="calls_universal",
            metadata={"description": "Универсальная база звонков для любых вопросов"}
        )

        # Дополнительно: инвертированный индекс для точного поиска
        self.keyword_index = {}

    def index_calls(self, call_texts: List[str]):
        """Индексирует звонки для универсального поиска"""
        print(f"📚 Индексирую {len(call_texts)} звонков...")

        for i, text in enumerate(call_texts):
            # Семантическое индексирование
            embedding = self.embedding_model.encode(text).tolist()

            self.collection.add(
                documents=[text],
                embeddings=[embedding],
                ids=[f"call_{i}"],
                metadatas=[{"call_id": i, "length": len(text)}]
            )

            # Точное текстовое индексирование (для имен, конкретных фраз)
            self._build_keyword_index(text, i)

        print("✅ База знаний готова для любых вопросов!")

    def _build_keyword_index(self, text: str, call_id: int):
        """Строит индекс ключевых слов для точного поиска"""
        words = re.findall(r'\b\w+\b', text.lower())
        for word in words:
            if len(word) > 3:  # Игнорируем короткие слова
                if word not in self.keyword_index:
                    self.keyword_index[word] = []
                self.keyword_index[word].append(call_id)

    def ask_anything(self, question: str, max_results: int = 10) -> str:
        """Задает любой вопрос по базе звонков"""
        print(f"🔍 Ищу ответ на: '{question}'")

        # Стратегия 1: Семантический поиск
        semantic_results = self._semantic_search(question, max_results)

        # Стратегия 2: Точный поиск по ключевым словам
        keyword_results = self._keyword_search(question)

        # Объединяем результаты
        all_relevant_calls = self._merge_results(semantic_results, keyword_results)

        if not all_relevant_calls:
            return "❌ В базе звонков не найдено информации для ответа на этот вопрос."

        print(f"📞 Найдено релевантных фрагментов: {len(all_relevant_calls)}")

        # Анализируем найденное
        return self._analyze_with_context(question, all_relevant_calls)

    def _semantic_search(self, question: str, max_results: int) -> List[str]:
        """Семантический поиск по смыслу"""
        question_embedding = self.embedding_model.encode(question).tolist()

        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=max_results
        )

        return results['documents'][0] if results['documents'] else []

    def _keyword_search(self, question: str) -> List[str]:
        """Точный поиск по ключевым словам"""
        relevant_call_ids = set()
        words = re.findall(r'\b\w+\b', question.lower())

        for word in words:
            if word in self.keyword_index:
                relevant_call_ids.update(self.keyword_index[word])

        # Получаем тексты найденных звонков
        keyword_results = []
        for call_id in list(relevant_call_ids)[:5]:  # Ограничиваем количество
            results = self.collection.get(ids=[f"call_{call_id}"])
            if results['documents']:
                keyword_results.extend(results['documents'])
                return keyword_results

    def _merge_results(self, semantic_results: List[str], keyword_results: List[str]) -> List[str]:
        """Объединяет результаты разных стратегий поиска"""
        all_results = semantic_results + keyword_results
        # Убираем дубликаты (простейшим способом)
        unique_results = []
        seen_texts = set()

        for result in all_results:
            text_hash = hash(result[:100])  # Хешируем начало для дедупликации
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_results.append(result)

        return unique_results[:15]  # Ограничиваем общее количество

    def _analyze_with_context(self, question: str, relevant_calls: List[str]) -> str:
        """Анализирует контекст для ответа на произвольный вопрос"""

        # Формируем контекст для модели
        context = "\n\n".join([
            f"[Фрагмент {i + 1}]: {call}"
            for i, call in enumerate(relevant_calls)
        ])

        prompt = f"""
    Ты — ассистент, который анализирует базу телефонных разговоров. 
    Отвечай ТОЛЬКО на основе предоставленных фрагментов разговоров. 
    Если информации для ответа нет - говори "В базе данных нет информации об этом".

    ВОПРОС: {question}

    БАЗА ДАННЫХ РАЗГОВОРОВ:
    {context}

    ИНСТРУКЦИИ:
    1. Отвечай точно на заданный вопрос
    2. Если нужно - цитируй конкретные фрагменты
    3. Если информации недостаточно - так и скажи
    4. Будь максимально конкретен

    ОТВЕТ:
    """
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.1,  # Минимум креативности для точности
                    'num_predict': 1000
                }
            )
            return response['response']
        except Exception as e:
            return f"Ошибка при анализе: {str(e)}"

# Пример использования с произвольными вопросами
def demo_universal_questions():
    analyzer = UniversalCallAnalyzer()

    # Пример базы звонков (в реальности - тысячи)
    call_texts = [
        "Менеджер: Здравствуйте, чем могу помочь? Клиент: У меня проблема с интернетом. Менеджер: Сейчас посмотрим... Вы правы, есть сбои.",
        "Клиент: Ваш сотрудник назвал меня глупым! Менеджер: Извините, такого быть не должно. Как звали сотрудника? Клиент: Не помню, но это было вчера.",
        "Менеджер Петров: Алло? Клиент: Здравствуйте, я хочу пожаловаться. Менеджер Петров: Слушаю вас. Клиент: Меня только что назвали дураком вашим оператором!",
        "Клиент: Мне нужна помощь с настройкой роутера. Менеджер Сидоров: Конечно, помогу. Сначала проверьте подключение кабеля. Клиент: Уже проверял.",
        "Менеджер: Добрый день! Клиент: Я в ярости! Ваш сотрудник Иванов нахамил мне! Менеджер: Приносим извинения, разберемся.",
        "Клиент: Подскажите тарифы. Менеджер: Есть пакет за 500 рублей. Клиент: Спасибо, подумаю.",
        "Менеджер: Чем могу помочь? Клиент: У меня медленный интернет. Менеджер: Возможно, проблемы на линии.",
    ]

    # Индексируем звонки
    analyzer.index_calls(call_texts)

    print("🚀 Система готова к ЛЮБЫМ вопросам!")
    print("=" * 60)

    # Примеры произвольных вопросов
    test_questions = [
        "Называл ли менеджер клиента дураком за последний месяц?",
        "Кто из менеджеров упоминался в жалобах?",
        "Были ли случаи хамства от сотрудников?",
        "Что клиенты говорят про тарифы?",
        "Упоминался ли менеджер Петров в негативном контексте?",
        "Сколько раз клиенты жаловались на медленный интернет?",
        "Кто такой менеджер Иванов и что о нем говорят?",
        "Какие именно технические проблемы упоминались?",
    ]

    for question in test_questions:
        print(f"\n🤔 ВОПРОС: {question}")
        print("─" * 50)

        answer = analyzer.ask_anything(question)
        print(f"📝 ОТВЕТ: {answer}")

        print("─" * 50)

if __name__ == "__main__":
    demo_universal_questions()