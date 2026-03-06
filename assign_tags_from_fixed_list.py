import json
import os
import re
from typing import List, Dict, Any
import ollama
from typing import Union
#from llama_cpp import Llama


class JsonFileTaggingAgent:
    def __init__(self, model, node_url=None, tags_list: List[str] = None, mail=False):
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

        self.tags_list = tags_list
        self.mail = mail
        self.processed_files = set()

    def process_directory(self, input_dir: str, output_dir: str = None):
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        print(f" Найдено {len(json_files)} JSON файлов для обработки")

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
                print(f"    Пропущен (уже обработан)")
                continue

            try:
                result = self.tag_single_file(input_path, output_path)
                if result:
                    self.processed_files.add(filename)
                    print(f"   Успешно тегирован. Теги: {result.get('tags', [])}")
            except Exception as e:
                print(f"   Ошибка: {e}")

    def tag_single_file(self, input_path: str, output_path: str) -> Dict[str, Any]:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'transcription' in data and 'text' in data['transcription']:
            text = data['transcription']['text']
        elif 'text' in data:
            text = data['text']
        else:
            raise ValueError(f"В файле {input_path} не найден текст звонка")

        tags_result = self.get_tags_from_llm(text)

        if 'tags' not in data:
            data['tags'] = {}

        data['tags']['fixed_tags'] = tags_result.get('result', [])

        data.pop('segments', None)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return tags_result

    def get_tags_from_llm(self, text: str) -> Dict[str, Any]:
        truncated_text = text[:3000] + "..." if len(text) > 3000 else text

        prompt = f"""Ты — специалист по категоризации телефонных разговоров.
Есть записи телефонных разговоров менеджеров с клиентами, которые берут в аренду грязезащитные ковры и получают услуги по их доставке (замене) и чистке.

Вот текст одного разговора:
{truncated_text}

ТВОЕ ЗАДАНИЕ: Во-первых, Проанализируй этот разговорный текст и верни краткое summary, характеризующие 1-2 главных причины обращения клиента.

Если текст не содержит ясной причины обращения - верни только слово "нет", и ничего больше.

Во-вторых, Проанализируй этот разговорный текст. Ознакомься со списком заранее сформированных описаний: 
{', '.join(self.tags_list)}. Подходят ли какие-нибудь из них этому тексту?
Присвой тексту от 0 до 3 описаний из фиксированного списка, только если они действительно хорошо характеризуют причины обращения клиента.
Например, клиент долго не получает ответ на его заявку о том, что ему не доставили вовремя ковер. Тогда присвой два описания: про долгое ожидание ответа и про недоставку (несвоевременную замену).
Либо клиент хочет возобновить услуги И при этом добавить больше ковров, чем было у него раньше. Тогда подойдет описание про возобновление услуг и описание про добавление ковров. И так далее.

Если клиент выражает недовольство ценами или непонимание, почему цены неожиданно выросли, - выбирай описание "клиент недоволен ценами". Но если клиент просто запрашивает информацию о планируемом росте цен, или уточняет, когда будет индексация, не выражая непонимания или недовольства, - то выбирай описание, связанное с уточнением деталей. Сам факт роста цен в связи с инфляцией - нормален.
Аналогично с другими проблемами: если клиент упоминает ключевые слова, связанные с какой-либо проблемой, - это не всегда означает факт возникновения проблемы. Будь внимателен!

Выбирай описание "консультация или уточнение деталей" только в случае, если нет никакой другой причины обращения!
Если ни одно описание не подходит, - просто не присваивай никаких описаний.

ВЕРНИ ОТВЕТ ТОЛЬКО В ФОРМАТЕ JSON:
{{
  "result": ["описание1", "описание2"],
  "summary": "причины обращения клиента своими словами"
}}
Если текст не содержит ясной причины обращения - верни пустой json.
"""
        
        print(prompt)

        
        prompt_mail = f"""Ты — специалист по категоризации писем электронной почты.
Есть емейл сообщения от клиентов, которые берут в аренду грязезащитные ковры и получают услуги по их доставке (замене) и чистке.

Вот один текст:
{truncated_text}

ТВОЕ ЗАДАНИЕ: Проанализируй этот текст и присвой ему от 1 до 3 тегов, наиболее хорошо характеризующих причины обращения клиента.
Например, клиент долго не получает ответ на его заявку о том, что ему не доставили вовремя ковер. Тогда будет два основных тега: про долгое ожидание ответа и про недоставку (несвоевременную замену).
Либо клиент хочет возобновить услуги И при этом добавить больше ковров, чем было у него раньше. Тогда должен быть тег про возобновление услуг и тег про добавление ковров. И так далее.

Если клиент выражает недовольство ценами или непонимание, почему цены неожиданно выросли, - выбирай тег клиент_недоволен_ценами. Но если клиент просто запрашивает информацию о планируемом росте цен, или уточняет, когда будет индексация, не выражая непонимания или недовольства, - то выбирай тег, связанный с уточнением деталей. Сам факт роста цен в связи с инфляцией - нормален.

Выбирай тег консультация_или_уточнение_деталей, ТОЛЬКО если нет никакой другой причины обращения!

Теги можно брать строго из этого списка:
{', '.join(self.tags_list)}
Не придумывай другие теги!

ВЕРНИ ОТВЕТ ТОЛЬКО В ФОРМАТЕ JSON:
{{
  "result": ["tag1", "tag2"]
}}
Если текст не содержит ясной причины обращения - верни пустой массив
"""


        empty_response = {
                "result": [],
                "summary": '',
                "additional_tags": [],
                "reasoning": "Ошибка"
            }
        
        try:
            if len(truncated_text) < 30:
                print('Text is too short')
                return empty_response
            if self.is_local:
                response = self.model(prompt_mail if self.mail else prompt,
                                      temperature=0.3,
                                      top_p=0.9,
                                      num_gpu=80,
                                      n_ctx=8000)
            else:
                print('Getting response ...')
                response = self.client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        'temperature': 0.3,
                        'top_p': 0.9,
                        "num_gpu": 80,
                        'n_ctx': 8000
                    }
                )

            response_text = response['response']
            print('response + ctx8: ', response_text)
            token_count = response['prompt_eval_count']
            print(f"Real n tokens in prompt: {token_count}")
            
            
            prompt_selfcheck = f"""Ты — специалист по категоризации телефонных разговоров.
Есть записи телефонных разговоров менеджеров с клиентами, которые берут в аренду грязезащитные ковры и получают услуги по их доставке (замене) и чистке.

Вот текст одного разговора:
{truncated_text}

Ты присвоил ему следующие теги:
{response_text}, которые ты выбрал из списка:
{', '.join(self.tags_list)}

Проверь себя! Точно ли каждый из выбранных тобой тегов отражает реальную проблему / причину обращения клиента, а не просто содержит те же ключевые слова, что встречаются в тексте разговора?
Не забыл ли ты добавить какие-нибудь теги?
Если нужно - исправь свой ответ. Если не нашел неточностей - оставь ответ тем же.
Выбирай тег консультация_или_уточнение_деталей, ТОЛЬКО если нет никакой другой причины обращения!

ВЕРНИ ОТВЕТ ТОЛЬКО В ФОРМАТЕ JSON (от 0 до 3 тегов):
{{
  "result": ["tag1", "tag2"]
}}
Если текст не содержит ясной причины обращения - верни пустой json
"""
            
#             print('Getting response - try 2 ...')
#             response = self.client.generate(
#                 model=self.model_name,
#                 prompt=prompt_selfcheck,
#                 options={
#                     'temperature': 0.3,
#                     'top_p': 0.8,
#                     'num_ctx': 3000
#                 }
#             )
#             response_text = response['response']
#             print('response (corrected): ', response_text)

            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())

                valid_selected = []
                for tag in result.get('result', []):
                    if tag in self.tags_list:
                        valid_selected.append(tag)
                    else:
                        print(f"    Модель придумала тег '{tag}', игнорирую")

                result['result'] = valid_selected

                return result
            else:
                raise ValueError("LLM не вернула JSON")

        except Exception as e:
            print(f"    Ошибка при запросе к LLM: {e}")
            # Возвращаем заглушку в случае ошибки
            return empty_response
        
    def get_summary_from_llm(self, text: str) -> Dict[str, Any]:
        truncated_text = text[:3000] + "..." if len(text) > 3000 else text

        prompt = f"""Ты — специалист по категоризации телефонных разговоров.
Есть записи телефонных разговоров менеджеров с клиентами, которые берут в аренду грязезащитные ковры и получают услуги по их доставке (замене) и чистке.

Вот текст одного разговора:
{truncated_text}

ТВОЕ ЗАДАНИЕ: Проанализируй этот разговорный текст и верни краткое summary, характеризующие 1-2 главных причины обращения клиента.

Если текст не содержит ясной причины обращения - верни только слово "нет", и ничего больше.

Если текст содержит жалобу на долгое отсутствие обратной связи (письменной или устной) от менеджеров, то, если это возможно, оцени задержку (в днях) и верни число дней. Например, клиент жалуется: я жду ответ со вчерашнего дня! тогда ответ = 1. Если клиент говорит: никто не перезванивает мне с самого утра... то ответ = 0.5.

Форма твоего ответа: текст summary, затем после запятой напиши число задержки (если она была и вычисляема) либо 0 в противном случае.
"""


        empty_response = ['', 0]
        
        try:
            if len(truncated_text) < 30:
                print('Text is too short')
                return empty_response
            if self.is_local:
                response = self.model(prompt_mail if self.mail else prompt,
                                      temperature=0.3,
                                      top_p=0.9,
                                      n_ctx=25000)
            else:
                print('Getting response ...')
                response = self.client.generate(
                    model=self.model_name,
                    prompt=prompt_mail if self.mail else prompt,
                    options={
                        'temperature': 0.3,
                        'top_p': 0.9,
                        'n_ctx': 25000
                    }
                )

            response_text = response['response']
            print('response + ctx35: ', response_text)
            token_count = response['prompt_eval_count']
            print(f"Real n tokens in prompt: {token_count}")
        
            summary, delay = response_text.strip().rsplit(',')
            return [summary, float(delay)]

        except Exception as e:
            print(f"    Ошибка при запросе к LLM: {e}")
            # Возвращаем заглушку в случае ошибки
            return empty_response

    def validate_tags_consistency(self, input_dir: str, sample_size: int = 20):
        print("\n" + "=" * 60)
        print(" ПРОВЕРКА КОНСИСТЕНТНОСТИ ТЕГИРОВАНИЯ")
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

        print("\n СТАТИСТИКА ТЕГОВ (выборка):")
        for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {tag}: {count}")

        if additional_counts:
            print("\n ПРЕДЛОЖЕННЫЕ ДОПОЛНИТЕЛЬНЫЕ ТЕГИ:")
            for tag, count in sorted(additional_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"   '{tag}': {count}")

        print(f"\n Проверено файлов: {len(sample_files)}")
        print(f" Уникальных тегов: {len(tag_counts)}")

