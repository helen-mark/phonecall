import ollama
import json
from typing import List, Dict
import os


class CallAnalyzer:
    def __init__(self, model_name: str = "llama3:8b"):
        self.model_name = model_name
        self.client = ollama.Client()

    def analyze_calls(self, call_texts: List[str], question: str) -> str:
        """
        Анализирует тексты звонков и отвечает на вопрос

        Args:
            call_texts: список текстов звонков
            question: вопрос для анализа

        Returns:
            ответ от модели
        """
        # Объединяем все тексты звонков
        all_calls_text = "\n\n".join([f"Звонок {i + 1}: {text}" for i, text in enumerate(call_texts)])

        # Формируем промпт
        prompt = f"""
Ты - аналитик колл-центра. Проанализируй следующие расшифровки телефонных разговоров и ответь на вопрос.

РАСШИФРОВКИ ЗВОНКОВ:
{all_calls_text}

ВОПРОС: {question}

Ответь максимально подробно и информативно. Если в данных нет информации для ответа - так и скажи.
ОТВЕТ:
"""
        try:
            # Отправляем запрос к модели
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.3,  # Меньше креативности, больше фактов
                    'num_predict': 1000  # Максимальная длина ответа
                }
            )

            return response['response']

        except Exception as e:
            return f"Ошибка при обращении к модели: {str(e)}"


def main():
    # Создаем экземпляр анализатора
    transcriptions_path = 'transcriptions'
    analyzer = CallAnalyzer()
    call_filepaths = [os.path.join(transcriptions_path, filename) for filename in os.listdir(transcriptions_path) if 'json' in filename]

    # Пример текстов звонков (замените на реальные данные)
    call_texts = [
        json.load(open(file_path, 'r', encoding='utf-8'))['transcription']['text'] for file_path in call_filepaths
    ]

    # Вопрос для анализа
    question = "На что чаще всего жалуются клиенты?"

    print("🔍 Анализирую звонки...")
    print(f"📞 Количество звонков: {len(call_texts)}")
    print(f"❓ Вопрос: {question}")
    print("-" * 50)

    # Получаем ответ от модели
    answer = analyzer.analyze_calls(call_texts, question)

    print("📊 Результат анализа:")
    print(answer)
    print("-" * 50)


if __name__ == "__main__":
    main()