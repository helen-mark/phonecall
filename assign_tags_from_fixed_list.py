import json
import os
import re
from typing import List, Dict, Any
import ollama  # Или openai, если используете OpenAI API
from typing import Union
#from llama_cpp import Llama


class JsonFileTaggingAgent:
    def __init__(self, model, node_url=None, tags_list: List[str] = None):
        """
        Инициализация агента для тегирования звонков

        Args:
            model_name: название модели Ollama
            tags_list: список тегов для классификации
        """

        self.is_local = False #isinstance(model, Llama)
        if self.is_local:
            self.model_name = 'local'
            self.model = model
        elif node_url:
            self.client = ollama.Client(host=node_url)
            self.model_name = 'from_yandex_node'
        else:
            self.client = ollama.Client()
            self.model_name = model

        # Стандартные теги (расширьте под свою предметную область)
        self.tags_list = tags_list

        # Кэш для избежания повторной обработки
        self.processed_files = set()

    def process_directory(self, input_dir: str, output_dir: str = None):
        """
        Обработка всех JSON файлов в директории

        Args:
            input_dir: путь к папке с JSON файлами
            output_dir: путь для сохранения (если None, перезаписывает исходные)
        """
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        print(f"📁 Найдено {len(json_files)} JSON файлов для обработки")

        n = 0
        for i, filename in enumerate(json_files, 1):
            if filename in os.listdir(output_dir):
                n += 1
                continue
        print(f'{n} files already processed')

        for i, filename in enumerate(json_files, 1):
            print(f"\n[{i}/{len(json_files)}] Обрабатываю {filename}...")
            if filename in os.listdir(output_dir):
                continue

            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename) if output_dir else input_path

            # Пропускаем уже обработанные
            if filename in self.processed_files:
                print(f"   ⏭️  Пропущен (уже обработан)")
                continue

            try:
                result = self.tag_single_file(input_path, output_path)
                if result:
                    self.processed_files.add(filename)
                    print(f"   ✅ Успешно тегирован. Теги: {result.get('tags', [])}")
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")

    def tag_single_file(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """
        Тегирование одного JSON файла

        Returns:
            Словарь с результатами тегирования
        """
        # Загрузка JSON
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Извлечение текста
        if 'transcription' in data and 'text' in data['transcription']:
            text = data['transcription']['text']
        elif 'text' in data:
            text = data['text']
        else:
            raise ValueError(f"В файле {input_path} не найден текст звонка")

        # Получение тегов от LLM
        tags_result = self.get_tags_from_llm(text)

        # Сохранение результатов в структуру данных
        if 'tags' not in data:
            data['tags'] = {}

        data['tags']['fixed_tags'] = tags_result.get('result', [])

        data.pop('segments', None)

        # Сохранение обновленного JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return tags_result

    def get_tags_from_llm(self, text: str) -> Dict[str, Any]:
        """
        Получение тегов от LLM с защитой от галлюцинаций

        Returns:
            Словарь с выбранными и дополнительными тегами
        """
        # Урезаем текст для экономии токенов (но оставляем суть)
        truncated_text = text[:3000] + "..." if len(text) > 3000 else text

        # Строгий промпт с требованием JSON формата
        prompt = f"""Ты — специалист по категоризации телефонных разговоров.
Есть записи телефонным разговоров с клиентами, которые берут в аренду ковры и получают услуги по их доставке (замене) и чистке.

Вот текст одного разговора:
{truncated_text}

ТВОЕ ЗАДАНИЕ: Проанализируй этот разговорный текст и присвой ему от 1 до 3 тегов, наиболее хорошо характеризующих причины обращения клиента.
Например, клиент долго не получает ответ на его заявку о том, что ему не доставили вовремя ковер. Тогда будет два основных тега: про долгое ожидание ответа и про недоставку (несвоевременную замену).
Либо клиент хочет возобновить услуги И при этом добавить больше ковров, чем было у него раньше. Тогда должен быть тег про возобновление и тег про добавление ковров.
И так далее.

Теги можно брать строго из этого списка:
{', '.join(self.tags_list)}
Не придумывай другие теги!

ВЕРНИ ОТВЕТ ТОЛЬКО В ФОРМАТЕ JSON:
{{
  "result": ["tag1", "tag2"],
}}
Если текст не содержит ясной причины обращения - верни пустой массив
"""

        try:
            if self.is_local:
                response = self.model(prompt,
                                      format='json',
                                      temperature=0.3,
                                      top_p=0.9)
            else:
                response = self.client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    format="json",  # Критически важно для парсинга!
                    options={
                        'temperature': 0.3,  # Минимум креативности для консистентности
                        'num_predict': 150,
                        'top_p': 0.9
                    }
                )

            # Извлечение JSON из ответа
            response_text = response['response']

            # Ищем JSON в ответе (на случай, если модель добавила текст)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())

                # Валидация: проверяем, что выбранные теги действительно из нашего списка
                valid_selected = []
                for tag in result.get('result', []):
                    if tag in self.tags_list:
                        valid_selected.append(tag)
                    else:
                        print(f"   ⚠️  Модель придумала тег '{tag}', игнорирую")

                result['result'] = valid_selected

                return result
            else:
                raise ValueError("LLM не вернула JSON")

        except Exception as e:
            print(f"   ⚠️  Ошибка при запросе к LLM: {e}")
            # Возвращаем заглушку в случае ошибки
            return {
                "result": [],
                "additional_tags": [],
                "reasoning": f"Ошибка: {str(e)}"
            }

    def validate_tags_consistency(self, input_dir: str, sample_size: int = 20):
        """
        Проверка консистентности тегирования (качество контроля)

        Args:
            input_dir: путь к папке с обработанными файлами
            sample_size: количество файлов для проверки
        """
        print("\n" + "=" * 60)
        print("🔍 ПРОВЕРКА КОНСИСТЕНТНОСТИ ТЕГИРОВАНИЯ")
        print("=" * 60)

        json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        sample_files = json_files[:min(sample_size, len(json_files))]

        tag_counts = {}
        additional_counts = {}

        for filename in sample_files:
            with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'tags' in data and 'fixed' in data['tags']:
                for tag in data['tags']['fixed']:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

        print("\n📊 СТАТИСТИКА ТЕГОВ (выборка):")
        for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {tag}: {count}")

        if additional_counts:
            print("\n💡 ПРЕДЛОЖЕННЫЕ ДОПОЛНИТЕЛЬНЫЕ ТЕГИ:")
            for tag, count in sorted(additional_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"   '{tag}': {count}")

        print(f"\n✅ Проверено файлов: {len(sample_files)}")
        print(f"✅ Уникальных тегов: {len(tag_counts)}")


# Пример использования
def main():
    # Инициализация агента с вашим списком тегов
    my_tags = [
        "низкое_качество_стирки_или_чистки",
        "не_заменили_ковры_вовремя",
        "клиент_хочет_добавить_ковры",
        "клиент_хочет_меньше_ковров",
        "погашение_долга",
        "расторжение_договора",
        "возобновление_услуг",
        "долго_нет_ответа_на_заявку",
        "лишняя_доставка",
        "доставили_не_те_ковры",
        "не_выставлен_вовремя_счет",
        "неверная_сумма_в_счете",
        "ковер_забрали_без_причины",
        "забрали_не_тот_ковер",
        "менеджер_нагрубил_клиенту",
        "неоправданно_высокие_цены",
        "неоправданный_рост_цен",
        "новый_клиент_заключение_договора",
        "консультация_или_уточнение_деталей",
        "поменять_спецификации",
        "менеджер_обещал_но_не_связался_с_клиентом",
        "клиент_уходит_к_конкурентам",
        "приостановить_услуги",
        "ошибка_в_документах"
    ]

    tagger = JsonFileTaggingAgent(
        model="mistral-nemo:12b",  # или "mistral", "qwen2.5:7b" и т.д.
        tags_list=my_tags
    )

    # Обработка всей директории
    input_directory = "transcriptions/"
    output_directory = "transcriptions_with_tags_strict_deepseek/"  # или None для перезаписи

    # Основная обработка
    tagger.process_directory(input_directory, output_directory)

    # Проверка консистентности
    result_dir = output_directory or input_directory
    tagger.validate_tags_consistency(result_dir)

    print("\n" + "=" * 60)
    print("🎉 ТЕГИРОВАНИЕ ЗАВЕРШЕНО!")
    print("=" * 60)


if __name__ == "__main__":
    main()