import json
import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, Counter
import sqlite3
from contextlib import contextmanager
import pandas as pd
from typing import Union
import ollama
#from llama_cpp import Llama


# ==================== Структуры данных ====================

class MetricType(Enum):
    """Типы метрик для анализа"""
    COUNT_BY_TAG = "count_by_tag"
    TOP_N_TAGS = "top_n_tags"
    TAG_TRENDS = "tag_trends"
    COMPARISON = "comparison"
    SUMMARY_STATS = "summary_stats"


@dataclass
class AnalysisPlan:
    """План анализа от LLM"""
    time_period: Dict[str, Any]  # start, end, description
    target_tags: List[str]
    metrics: List[MetricType]
    grouping: str = "month"
    comparison_tags: List[str] = None
    additional_filters: Dict = None

    def to_dict(self):
        """Конвертирует в словарь для JSON"""
        return {
            'time_period': self.time_period,
            'target_tags': self.target_tags,
            'metrics': [m.value for m in self.metrics],
            'grouping': self.grouping,
            'comparison_tags': self.comparison_tags,
            'filters': self.additional_filters or {}
        }


# ==================== Google Drive Data Loader ====================

class DriveDataLoader:
    """Загружает и управляет данными из Google Drive JSON файлов"""

    def __init__(self, json_directory: str, drive_path: str = None):
        self.csv_dir = json_directory
        self.drive_path = drive_path
        self.calls_cache = None
        self.conn = None
        self._check_drive_access()

    def _check_drive_access(self):
        """Проверяет доступ к Google Drive"""
        if self.drive_path and 'drive' in self.csv_dir:
            print(f"🌐 Использую Google Drive: {self.csv_dir}")

            # Проверяем существование директории
            if not os.path.exists(self.csv_dir):
                print(f"⚠️  Директория не найдена в Drive: {self.csv_dir}")
                print("ℹ️  Создаю директорию...")
                os.makedirs(self.csv_dir, exist_ok=True)

                # Создаем README файл
                readme_path = os.path.join(self.csv_dir, "README.txt")
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write("Директория для JSON файлов телефонных звонков\n")
                    f.write("Загрузите сюда файлы в формате JSON\n")
                    f.write(f"Создано: {datetime.now()}")

                print(f"✅ Создана новая директория в Google Drive")
            else:
                print(f"✅ Директория найдена в Google Drive")

    def load_all_calls(self, limit: int = None) -> List[Dict]:
        """Загружает все звонки из CSV файла"""
        if self.calls_cache is not None:
            return self.calls_cache[:limit] if limit else self.calls_cache

        all_calls = []

        # Проверяем существование директории
        if not os.path.exists(self.csv_dir):
            print(f"❌ Директория не найдена: {self.csv_dir}")
            if self.drive_path:
                print(f"ℹ️  Убедитесь, что папка существует в Google Drive")
                print(f"📍 Ожидаемый путь: {self.csv_dir}")
            return []

        # Ищем CSV файлы
        try:
            csv_files = [f for f in os.listdir(self.csv_dir) if f.endswith('.csv')]
        except Exception as e:
            print(f"❌ Ошибка чтения директории: {e}")
            return []

        if not csv_files:
            print(f"⚠️  В директории {self.csv_dir} нет CSV файлов")
            print("ℹ️  Ожидаемый формат CSV: колонки 'date', 'text', 'tags'")
            return []

        # Берем первый CSV файл (можно расширить для нескольких)
        csv_file = csv_files[0]
        filepath = os.path.join(self.csv_dir, csv_file)

        print(f"📂 Читаю данные из CSV файла: {csv_file}")

        try:
            # Читаем CSV файл
            df = pd.read_csv(
                filepath,
                encoding='utf-8',
                parse_dates=['date'],  # Автоматически парсим дату
                converters={
                    'tags': lambda x: eval(x) if isinstance(x, str) else []  # Конвертируем строку в список
                }
            )

            # Проверяем необходимые колонки
            required_columns = ['date', 'text', 'tags']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                print(f"❌ В CSV файле отсутствуют колонки: {missing_columns}")
                print(f"   Доступные колонки: {list(df.columns)}")
                return []

            print(f"✅ Загружено {len(df)} строк из CSV")

            # Конвертируем DataFrame в список словарей
            for idx, row in df.iterrows():
                # Получаем дату (уже в формате datetime благодаря parse_dates)
                call_date = pd.to_datetime(row['date'])

                # Если tags - строка, конвертируем в список
                tags = row['tags']
                if isinstance(tags, str):
                    try:
                        # Обрабатываем формат: "['tag1', 'tag2']"
                        tags = eval(tags) if tags.startswith('[') else tags.split(',')
                    except:
                        tags = []

                # Формируем структурированную запись
                call_record = {
                    'id': f"call_{idx}",
                    'file_name': csv_file,
                    'call_date': call_date,
                    'year': call_date.year if pd.notna(call_date) else None,
                    'month': call_date.month if pd.notna(call_date) else None,
                    'day': call_date.day if pd.notna(call_date) else None,
                    'full_text': str(row['text']) if pd.notna(row['text']) else '',
                    'summary': row.get('summary', '') if 'summary' in df.columns else '',
                    'tags': tags if isinstance(tags, list) else [tags],
                    'text_length': len(str(row['text'])) if pd.notna(row['text']) else 0,
                    'source_file': filepath,
                    'drive_path': self.drive_path if self.drive_path else None
                }

                all_calls.append(call_record)

                # Ограничение по количеству записей
                if limit and idx + 1 >= limit:
                    break

            self.calls_cache = all_calls

            # Выводим статистику
            print(f"✅ Преобразовано {len(all_calls)} записей звонков")

            if self.drive_path:
                print(f"🌐 Данные загружены из Google Drive")

            # Дополнительная информация о данных
            if all_calls:
                dates = [c['call_date'] for c in all_calls if c['call_date']]
                if dates:
                    min_date = min(dates)
                    max_date = max(dates)
                    print(f"📅 Диапазон дат: {min_date.strftime('%d.%m.%Y')} - {max_date.strftime('%d.%m.%Y')}")

                # Подсчет уникальных тегов
                all_tags = []
                for call in all_calls:
                    all_tags.extend(call['tags'])
                unique_tags = set(all_tags)
                print(f"🏷️  Уникальных тегов: {len(unique_tags)}")

            return all_calls

        except pd.errors.EmptyDataError:
            print(f"❌ CSV файл {csv_file} пустой")
            return []
        except Exception as e:
            print(f"❌ Ошибка чтения CSV файла {csv_file}: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _extract_date_from_filename(self, filename: str) -> datetime:
        """Извлекает дату из имени файла"""
        # Паттерны для поиска даты в имени файла
        patterns = [
            r'(\d{4})-(\d{2})-(\d{2})',  # YYYY-MM-DD
            r'(\d{2})\.(\d{2})\.(\d{4})',  # DD.MM.YYYY
            r'(\d{4})(\d{2})(\d{2})',  # YYYYMMDD
        ]

        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    if pattern == patterns[0]:  # YYYY-MM-DD
                        year, month, day = map(int, groups)
                        return datetime(year, month, day)
                    elif pattern == patterns[1]:  # DD.MM.YYYY
                        day, month, year = map(int, groups)
                        return datetime(year, month, day)
                    elif pattern == patterns[2]:  # YYYYMMDD
                        year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                        return datetime(year, month, day)

        # Если дата не найдена, используем дату изменения файла
        filepath = os.path.join(self.csv_dir, filename)
        if os.path.exists(filepath):
            try:
                return datetime.fromtimestamp(os.path.getmtime(filepath))
            except:
                pass

        # Fallback: текущая дата
        return datetime.now()

    def setup_in_memory_db(self):
        """Создает in-memory SQLite базу для быстрых запросов"""
        if self.conn is not None:
            return self.conn

        self.conn = sqlite3.connect(':memory:')
        cursor = self.conn.cursor()

        # Создаем таблицы
        cursor.execute("""
        CREATE TABLE calls (
            id TEXT PRIMARY KEY,
            file_name TEXT,
            call_date TEXT,
            year INTEGER,
            month INTEGER,
            day INTEGER,
            full_text TEXT,
            summary TEXT,
            tags_json TEXT,
            text_length INTEGER,
            source_file TEXT,
            drive_path TEXT
        )
        """)

        cursor.execute("""
        CREATE TABLE call_tags (
            call_id TEXT,
            tag TEXT,
            FOREIGN KEY (call_id) REFERENCES calls(id)
        )
        """)

        # Загружаем данные
        calls = self.load_all_calls()
        for call in calls:
            cursor.execute("""
            INSERT INTO calls (id, file_name, call_date, year, month, day, 
                              full_text, summary, tags_json, text_length, source_file, drive_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                call['id'],
                call['file_name'],
                call['call_date'].isoformat(),
                call['year'],
                call['month'],
                call['day'],
                call['full_text'],
                call['summary'],
                json.dumps(call['tags'], ensure_ascii=False),
                call['text_length'],
                call['source_file'],
                call.get('drive_path', '')
            ))

            # Вставляем теги
            for tag in call['tags']:
                cursor.execute(
                    "INSERT INTO call_tags (call_id, tag) VALUES (?, ?)",
                    (call['id'], tag)
                )

        self.conn.commit()

        source = "Google Drive" if self.drive_path else "локальной папки"
        print(f"✅ Данные загружены в in-memory SQLite ({len(calls)} записей из {source})")
        return self.conn

    @contextmanager
    def get_cursor(self):
        """Контекстный менеджер для курсора"""
        if self.conn is None:
            self.setup_in_memory_db()

        cursor = self.conn.cursor()
        try:
            yield cursor
        finally:
            cursor.close()


# ==================== DeepSeek Planner ====================

class DeepSeekPlanner:
    """LLM планировщик запросов"""

    def __init__(self, model, datasphere_node_url=None, drive_path=None):
        self.is_local = False  #isinstance(model, Llama)

        self.drive_path = drive_path
        self.available_tags = self._load_available_tags()

        if self.is_local:
            self.model = model
            self.model_name = 'local'
        elif datasphere_node_url:
            self.client = ollama.Client(host=datasphere_node_url, timeout=300)
            self.model_name = 'from_yandex_node'
            print(f"Mode: Yandex DataSphere (node url: {datasphere_node_url})")
        else:
            self.model_name = model
            self._setup_ollama_client()



    def _setup_ollama_client(self):
        """Настраивает клиент Ollama с учетом Google Drive"""
        try:
            # Настройка для Colab
            host = "http://localhost:11434"

            # Если есть Google Drive, можно кэшировать модели
            if self.drive_path:
                models_cache_dir = os.path.join(self.drive_path, "models_cache")
                os.makedirs(models_cache_dir, exist_ok=True)
                print(f"🌐 Кэш моделей Ollama в Google Drive: {models_cache_dir}")

            self.client = ollama.Client(host=host, timeout=60.0)

            # Проверяем доступность
            try:
                self.client.list()
                print(f"✅ Ollama подключен, модель: {self.model_name}")
            except Exception as e:
                print(f"⚠️  Ошибка подключения к Ollama: {e}")
                print("ℹ️  Убедитесь, что Ollama запущен в Colab")

        except ImportError:
            print("❌ Ollama не установлен")
            raise

    def _load_available_tags(self) -> List[str]:
        """Загружает все уникальные теги из JSON файлов"""
        # В реальности нужно загрузить из данных
        return [
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

    def create_analysis_plan(self, user_query: str) -> AnalysisPlan:
        """Создает план анализа на основе запроса пользователя"""

        prompt = self._build_planner_prompt(user_query)

        try:
            if self.is_local:
                response = self.model(
                    prompt,
                    max_tokens=500,
                    temperature=0.1)
            else:
                print('Use timeout 250')
                response = self.client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    format="json",
                    options={'temperature': 0.1, 'num_predict': 250, 'timeout': 250}
                )

            plan_data = json.loads(response['response'])

            # Парсим временной период
            time_period = self._parse_time_period(plan_data.get('time_period', {}))

            # Валидируем теги
            target_tags = self._validate_tags(plan_data.get('target_tags', []))

            # Парсим метрики
            metrics = self._parse_metrics(plan_data.get('metrics', []))

            return AnalysisPlan(
                time_period=time_period,
                target_tags=target_tags,
                metrics=metrics,
                grouping=plan_data.get('grouping', 'month'),
                comparison_tags=plan_data.get('comparison_tags', []),
                additional_filters=plan_data.get('filters', {})
            )

        except Exception as e:
            print(f"❌ Ошибка планировщика: {e}")
            # Возвращаем план по умолчанию
            return self._create_default_plan(user_query)

    def _build_planner_prompt(self, user_query: str) -> str:
        """Строит промпт для планировщика"""
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Добавляем информацию о источнике данных
        data_source = "Google Drive" if self.drive_path else "локальной базы данных"

        return f"""Ты — аналитик базы телефонных звонков компании по аренде ковров.

ИСТОЧНИК ДАННЫХ: {data_source}

ЗАПРОС: "{user_query}"

ТВОЯ ЗАДАЧА: Создать план анализа.
Система будет обращаться по твоему плану к текстам с записями телефонных звонков клиентов за несколько последних лет, содержащими описательные теги каждого звонка.

ДОСТУПНЫЕ ТЕГИ:
{', '.join(self.available_tags)}

МЕТРИКИ, которые система может посчитать для тебя для ответа на запрос, если это необходимо:
1. count_by_tag - подсчет звонков с заданным тегом за период
2. top_n_tags - самые частые теги звонков за период
3. tag_trends - динамика тега по времени: стал ли чаще или реже встречаться за период?
Сегодняшняя дата: {current_date} - используй ее, чтобы правильно определить временной период из запроса в случае, если в запросе временной период указан относительно сегодняшнего дня (например, "в прошлом году" и т.п.)

ВЕРНИ JSON с планом того, что системе нужно извлечь из данных для ответа на запрос, а именно: за какой период понадобятся данные? По каким имено тегам выбирать данные для ответа на данный запрос? Какие метрики подсчитать по этим данным для ответа на данный запрос?

{{
  "time_period": {{
    "description": "описание периода",
    "start": "YYYY-MM-DD или null",
    "end": "YYYY-MM-DD или null"
  }},
  "target_tags": ["тег1", "тег2", ... (1 or more tags)],
  "metrics": ["count_by_tag" and/or "tag_trends" and/or "top_n_tags" (necessary metrics)],
  "grouping": "month/week/day"
  }}

Ответ:
<|think|>false<|end|>"""

    def _parse_time_period(self, period_data: Dict) -> Dict[str, Any]:
        """Парсит временной период"""
        today = datetime.now()
        description = period_data.get('description', '')

        # Начальные значения
        start = today - timedelta(days=30)  # По умолчанию последний месяц
        end = today

        # Определяем период на основе описания
        if 'последние 6 месяцев' in description.lower():
            start = today - timedelta(days=30 * 6)
        elif 'этот месяц' in description.lower():
            start = datetime(today.year, today.month, 1)
        elif 'этот год' in description.lower():
            start = datetime(today.year, 1, 1)
        elif 'первый квартал 2024' in description.lower():
            start = datetime(2024, 1, 1)
            end = datetime(2024, 3, 31)
        elif 'прошлый год' in description.lower():
            start = datetime(today.year - 1, 1, 1)
            end = datetime(today.year - 1, 12, 31)

        # Переопределяем если указаны точные даты
        if period_data.get('start'):
            try:
                start = datetime.fromisoformat(period_data['start'])
            except:
                pass
        if period_data.get('end'):
            try:
                end = datetime.fromisoformat(period_data['end'])
            except:
                pass

        return {
            'start': start,
            'end': end,
            'description': description or f"с {start.strftime('%d.%m.%Y')} по {end.strftime('%d.%m.%Y')}"
        }

    def _validate_tags(self, tags: List[str]) -> List[str]:
        """Фильтрует и нормализует теги"""
        valid_tags = []
        for tag in tags:
            # Ищем похожие теги
            for available_tag in self.available_tags:
                if tag.lower() in available_tag.lower() or available_tag.lower() in tag.lower():
                    valid_tags.append(available_tag)
                    break

        return valid_tags or ['низкое_качество_стирки_или_чистки']  # Fallback

    def _parse_metrics(self, metrics: List[str]) -> List[MetricType]:
        """Парсит метрики"""
        metric_map = {
            'count_by_tag': MetricType.COUNT_BY_TAG,
            'top_n_tags': MetricType.TOP_N_TAGS,
            'tag_trends': MetricType.TAG_TRENDS,
            'comparison': MetricType.COMPARISON
        }

        result = []
        for metric in metrics:
            if metric in metric_map:
                result.append(metric_map[metric])

        return result or [MetricType.COUNT_BY_TAG]

    def _create_default_plan(self, user_query: str) -> AnalysisPlan:
        """Создает план по умолчанию при ошибке"""
        today = datetime.now()

        # Эвристики для определения тега
        target_tags = []
        if 'качеств' in user_query.lower() or 'стирк' in user_query.lower():
            target_tags.append('низкое_качество_стирки_или_чистки')
        if 'цен' in user_query.lower() or 'дорог' in user_query.lower():
            target_tags.append('неоправданно_высокие_цены')
        if 'консульт' in user_query.lower() or 'уточн' in user_query.lower():
            target_tags.append('консультация_или_уточнение_деталей')

        return AnalysisPlan(
            time_period={
                'start': today - timedelta(days=30 * 6),
                'end': today,
                'description': 'последние 6 месяцев'
            },
            target_tags=target_tags or ['низкое_качество_стирки_или_чистки'],
            metrics=[MetricType.COUNT_BY_TAG, MetricType.TAG_TRENDS],
            grouping='month'
        )


# ==================== Query Executor ====================

class JSONQueryExecutor:
    """Выполняет аналитические запросы к JSON данным"""

    def __init__(self, data_loader: DriveDataLoader):
        self.data_loader = data_loader

    def execute_plan(self, plan: AnalysisPlan) -> Dict[str, Any]:
        """Выполняет план анализа"""

        results = {}

        # Получаем данные за период
        all_calls = self.data_loader.load_all_calls()

        if not all_calls:
            print("⚠️  Нет данных для анализа")
            return {
                'error': 'Нет данных для анализа',
                'summary_stats': {
                    'total_calls': 0,
                    'period': plan.time_period['description'],
                    'date_range': f"{plan.time_period['start'].strftime('%Y-%m-%d')} - {plan.time_period['end'].strftime('%Y-%m-%d')}"
                }
            }

        print(f'{len(all_calls)} звонков всего')
        filtered_calls = self._filter_calls_by_period(all_calls, plan.time_period)
        print(f'{len(filtered_calls)} звонков после фильтрации по периоду')

        # Выполняем метрики
        for metric in plan.metrics:
            if metric == MetricType.COUNT_BY_TAG:
                results['count_by_tag'] = self._count_by_tag(filtered_calls, plan.target_tags)

            elif metric == MetricType.TAG_TRENDS:
                results['tag_trends'] = self._tag_trends(
                    filtered_calls,
                    plan.target_tags,
                    plan.grouping
                )

            elif metric == MetricType.TOP_N_TAGS:
                results['top_n_tags'] = self._top_n_tags(filtered_calls, n=5)

            elif metric == MetricType.COMPARISON:
                results['comparison'] = self._compare_tags(
                    filtered_calls,
                    plan.comparison_tags or plan.target_tags[:2]
                )

        # Добавляем общую статистику
        results['summary_stats'] = {
            'total_calls': len(filtered_calls),
            'period': plan.time_period['description'],
            'date_range': f"{plan.time_period['start'].strftime('%Y-%m-%d')} - {plan.time_period['end'].strftime('%Y-%m-%d')}",
            'data_source': 'Google Drive' if self.data_loader.drive_path else 'Local'
        }

        return results

    def _filter_calls_by_period(self, calls: List[Dict], period: Dict) -> List[Dict]:
        """Фильтрует звонки по временному периоду"""
        start_date = period['start']
        end_date = period['end']

        filtered = []
        for call in calls:
            call_date = call['call_date']
            if start_date <= call_date <= end_date:
                filtered.append(call)

        return filtered

    def _count_by_tag(self, calls: List[Dict], target_tags: List[str]) -> Dict[str, int]:
        """Подсчет звонков по тегам"""
        counts = defaultdict(int)

        for call in calls:
            for tag in call['tags']:
                # Проверяем, совпадает ли тег с целевыми
                for target in target_tags:
                    if target.lower() in tag.lower() or tag.lower() in target.lower():
                        counts[target] += 1
                        break

        return dict(counts)

    def _tag_trends(self, calls: List[Dict], target_tags: List[str], grouping: str) -> Dict[str, List]:
        """Динамика тегов по времени"""
        if not target_tags or not calls:
            return {}

        # Группируем по месяцам/неделям
        trends = defaultdict(lambda: defaultdict(int))

        for call in calls:
            # Определяем ключ группировки
            if grouping == 'month':
                period_key = call['call_date'].strftime('%Y-%m')
            elif grouping == 'week':
                year, week, _ = call['call_date'].isocalendar()
                period_key = f"{year}-W{week:02d}"
            else:  # day
                period_key = call['call_date'].strftime('%Y-%m-%d')

            # Считаем теги
            for tag in call['tags']:
                for target in target_tags:
                    if target.lower() in tag.lower() or tag.lower() in target.lower():
                        trends[target][period_key] += 1
                        break

        # Преобразуем в список для каждого тега
        result = {}
        for tag, period_counts in trends.items():
            result[tag] = [
                {'period': period, 'count': count}
                for period, count in sorted(period_counts.items())
            ]

        return result

    def _top_n_tags(self, calls: List[Dict], n: int = 5) -> List[Dict]:
        """Топ-N самых частых тегов"""
        tag_counter = Counter()

        for call in calls:
            tag_counter.update(call['tags'])

        return [
            {'tag': tag, 'count': count}
            for tag, count in tag_counter.most_common(n)
        ]

    def _compare_tags(self, calls: List[Dict], tags: List[str]) -> Dict[str, Any]:
        """Сравнивает два тега"""
        if len(tags) < 2:
            tags = tags + [None] * (2 - len(tags))

        counts = self._count_by_tag(calls, tags[:2])

        return {
            'tag1': {'name': tags[0], 'count': counts.get(tags[0], 0)},
            'tag2': {'name': tags[1], 'count': counts.get(tags[1], 0)},
            'total_calls': len(calls),
            'ratio': counts.get(tags[0], 0) / counts.get(tags[1], 1) if counts.get(tags[1], 0) > 0 else 0
        }


# ==================== DeepSeek Analyzer ====================

class DeepSeekAnalyzer:
    """LLM для анализа результатов и генерации ответов"""

    def __init__(self, model, datasphere_node_url = None, drive_path: str = None):
        self.is_local = False # isinstance(model, Llama)

        if self.is_local:
            self.model_name = 'local'
            self.model = model
        elif datasphere_node_url:
            self.client = ollama.Client(host=datasphere_node_url, timeout=300)
            self.model_name = 'from_yandex_node'
            print(f"Mode: Yandex DataSphere (node url: {datasphere_node_url})")
        else:
            self.model_name = model
            try:
                self.client = ollama.Client(
                    host="http://localhost:11434",
                    timeout=90.0  # Увеличенный таймаут для больших моделей
                )
            except ImportError:
                print("❌ Ollama не установлен")
                raise

        self.drive_path = drive_path



    def generate_answer(self, user_query: str, results: Dict, plan: AnalysisPlan) -> str:
        """Генерирует итоговый ответ на основе результатов"""

        prompt = self._build_analyzer_prompt(user_query, results, plan)

        try:
            if self.is_local:
                response = self.model(prompt,
                                      temperature=0.3,
                                      num_predict=1000)
            else:
                response = self.client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={'temperature': 0.3, 'num_predict': 1000}
                )

            return response['response'].strip()

        except Exception as e:
            print(f"❌ Ошибка анализатора: {e}")
            return self._generate_fallback_answer(results, plan)

    def _build_analyzer_prompt(self, user_query: str, results: Dict, plan: AnalysisPlan) -> str:
        """Строит промпт для анализатора"""

        # Форматируем результаты для промпта
        results_str = json.dumps(results, ensure_ascii=False, indent=2, default=str)

        # Добавляем информацию об источнике
        data_source = "Google Drive" if self.drive_path else "локальной базы"

        return f"""Ты — старший аналитик компании по аренде ковров.

ИСТОЧНИК ДАННЫХ: Анализ выполнен на данных из {data_source}

ЗАПРОС КЛИЕНТА: "{user_query}"

Для ответа на запрос система выбрала тексты обращений клиентов за нужный период и посчитала нужные метрики.
- Период, которым интересовался клиент: {plan.time_period['description']}
- Подходящие теги, которые выбрала система для выбора обращений для анализа данного запроса: {', '.join(plan.target_tags)}
- Метрики, которые система подсчитала для выполнения данного запроса, на основании текстов обращений, отобранных по этим тегам: {[m.value for m in plan.metrics]}

Вот результаты, которые выдала система по подсчетам метрик для этих тегов:
{results_str}

ТВОЯ ЗАДАЧА:
1. Проанализировать цифры в этих результатах (если результат не пустой!)
2. Ответить на запрос клиента
3. Выделить ключевые инсайты
4. Говорить конкретно, с цифрами

ФОРМАТ:
- Краткий вывод
- Детальный анализ
- Рекомендации (если есть)

Если ты видишь, что система дала тебе пустые метрики, или информации в результатах не достаточно для ответа на запрос клиента, - так и напиши.

ОТВЕТ НА РУССКОМ:"""

    def _generate_fallback_answer(self, results: Dict, plan: AnalysisPlan) -> str:
        """Генерирует ответ если LLM не сработала"""

        answer_parts = []

        # Краткий вывод
        answer_parts.append(f"📊 Анализ за период: {plan.time_period['description']}")

        # Количество по тегам
        if 'count_by_tag' in results and results['count_by_tag']:
            answer_parts.append("\n📈 Количество звонков по тегам:")
            for tag, count in results['count_by_tag'].items():
                answer_parts.append(f"  • {tag}: {count}")
        else:
            answer_parts.append("\n⚠️  Нет данных по указанным тегам")

        # Динамика
        if 'tag_trends' in results:
            for tag, trends in results['tag_trends'].items():
                if trends:
                    first = trends[0]['count']
                    last = trends[-1]['count']
                    change = ((last - first) / first * 100) if first > 0 else 0
                    trend_desc = "📈 рост" if change > 0 else "📉 снижение" if change < 0 else "➡️ без изменений"
                    answer_parts.append(f"\n📅 Динамика '{tag}': {trend_desc} ({abs(change):.1f}%)")

        # Рекомендации
        if 'count_by_tag' in results and results['count_by_tag']:
            max_tag = max(results['count_by_tag'].items(), key=lambda x: x[1])[0] if results['count_by_tag'] else None
            if max_tag and ('жалоба' in max_tag or 'низкое' in max_tag):
                answer_parts.append(
                    f"\n💡 Рекомендация: Обратите внимание на тег '{max_tag}' - это самая частая категория обращений")

        return "\n".join(answer_parts)


# ==================== Главная MCP система ====================

class JSONCallAnalyticsMCP:
    """Главная MCP система для работы с Google Drive JSON файлами"""

    def __init__(self, json_directory: str, model, node_url=None, drive_path: str = None):
        self.drive_path = drive_path
        self.data_loader = DriveDataLoader(json_directory, drive_path)
        self.planner = DeepSeekPlanner(model, node_url, drive_path)
        self.executor = JSONQueryExecutor(self.data_loader)
        self.analyzer = DeepSeekAnalyzer(model, node_url, drive_path)

        # Загружаем данные при инициализации
        print("📂 Загружаю данные из JSON файлов...")
        self.total_calls = len(self.data_loader.load_all_calls())

        if self.total_calls == 0:
            print("⚠️  Внимание: Нет данных для анализа")
            if self.drive_path:
                print(f"ℹ️  Проверьте наличие JSON файлов в Google Drive: {json_directory}")
        else:
            print(f"✅ Загружено {self.total_calls} звонков")
            if self.drive_path:
                print(f"🌐 Данные загружены из Google Drive")

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Обрабатывает запрос пользователя"""

        print(f"\n🔍 Анализирую запрос: '{user_query}'")

        if self.drive_path:
            print(f"🌐 Источник данных: Google Drive")

        # 1. Планирование (LLM)
        print("🤖 Создаю план анализа...")
        analysis_plan = self.planner.create_analysis_plan(user_query)

        print(f"   📅 Период: {analysis_plan.time_period['description']}, {analysis_plan.time_period['start']}, {analysis_plan.time_period['end']}")
        print(f"   🏷️  Теги: {', '.join(analysis_plan.target_tags)}")
        print(f"   📊 Метрики: {[m.value for m in analysis_plan.metrics]}")

        # 2. Выполнение анализа
        print("📊 Выполняю анализ...")
        analysis_results = self.executor.execute_plan(analysis_plan)

        # 3. Генерация ответа (LLM)
        print("💭 Формулирую ответ...")
        answer = self.analyzer.generate_answer(user_query, analysis_results, analysis_plan)

        # 4. Формируем полный ответ
        response = {
            'query': user_query,
            'analysis_plan': analysis_plan.to_dict(),
            'raw_results': analysis_results,
            'answer': answer,
            'total_calls_analyzed': analysis_results.get('summary_stats', {}).get('total_calls', 0),
            'processing_time': datetime.now().isoformat(),
            'model_used': self.planner.model_name,
            'data_source': 'Google Drive' if self.drive_path else 'Local'
        }

        # 5. Выводим краткую статистику
        self._print_analysis_summary(analysis_results)

        return response

    def _print_analysis_summary(self, results: Dict[str, Any]):
        """Выводит краткую статистику анализа"""
        print("📈 КРАТКАЯ СТАТИСТИКА:")
        print("-" * 40)

        if 'summary_stats' in results:
            stats = results['summary_stats']
            print(f"📅 Период: {stats.get('period', 'N/A')}")
            print(f"📞 Проанализировано звонков: {stats.get('total_calls', 0)}")
            print(f"📍 Источник данных: {stats.get('data_source', 'Local')}")

        if 'count_by_tag' in results:
            counts = results['count_by_tag']
            if counts:
                print("\n📊 Количество по тегам:")
                for tag, count in counts.items():
                    print(f"  • {tag}: {count}")
            else:
                print("\n⚠️  Нет совпадений по указанным тегам")

        if 'top_n_tags' in results and results['top_n_tags']:
            print("\n🏆 Топ теги:")
            for i, item in enumerate(results['top_n_tags'][:3], 1):
                print(f"  {i}. {item['tag']}: {item['count']}")

        if 'tag_trends' in results:
            for tag, trends in results['tag_trends'].items():
                if trends and len(trends) >= 2:
                    first = trends[0]['count']
                    last = trends[-1]['count']
                    change = ((last - first) / first * 100) if first > 0 else 0
                    trend_icon = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                    print(f"\n📅 Динамика '{tag}': {trend_icon} {abs(change):.1f}%")

        print("-" * 40)

    def get_system_info(self) -> Dict[str, Any]:
        """Возвращает информацию о системе"""
        calls = self.data_loader.load_all_calls()

        # Собираем все теги
        all_tags = []
        for call in calls:
            all_tags.extend(call['tags'])

        unique_tags = set(all_tags)

        # Даты
        dates = [call['call_date'] for call in calls]

        return {
            'total_calls': len(calls),
            'unique_tags_count': len(unique_tags),
            'date_range': {
                'start': min(dates).isoformat() if dates else None,
                'end': max(dates).isoformat() if dates else None
            },
            'average_text_length': sum(len(c['full_text']) for c in calls) // len(calls) if calls else 0,
            'model': self.planner.model_name,
            'data_source': 'Google Drive' if self.drive_path else 'Local Files',
            'drive_path': self.drive_path if self.drive_path else None
        }

    def test_system(self) -> bool:
        """Тестирует работоспособность системы"""
        print("🧪 Тестирую систему...")

        try:
            # Тест 1: Загрузка данных
            calls = self.data_loader.load_all_calls(limit=10)
            if not calls:
                print("❌ Нет данных для анализа")
                if self.drive_path:
                    print(f"ℹ️  Проверьте Google Drive: {self.data_loader.csv_dir}")
                return False

            print(f"✅ Загружено {len(calls)} тестовых записей")

            # Тест 2: Планирование
            test_query = "Тестовый запрос: жалобы на качество"
            plan = self.planner.create_analysis_plan(test_query)
            if not plan.target_tags:
                print("❌ Планировщик не вернул теги")
                return False

            print(f"✅ Планировщик работает, выбраны теги: {plan.target_tags}")

            # Тест 3: Выполнение
            results = self.executor.execute_plan(plan)
            if 'summary_stats' not in results:
                print("❌ Исполнитель не вернул результаты")
                return False

            print(f"✅ Исполнитель проанализировал {results['summary_stats'].get('total_calls', 0)} звонков")

            # Тест 4: Анализ
            answer = self.analyzer.generate_answer(test_query, results, plan)
            if not answer or len(answer) < 10:
                print("❌ Анализатор не сгенерировал ответ")
                return False

            print(f"✅ Анализатор сгенерировал ответ длиной {len(answer)} символов")

            print("\n🎉 Все тесты пройдены успешно!")
            return True

        except Exception as e:
            print(f"❌ Ошибка при тестировании: {e}")
            import traceback
            traceback.print_exc()
            return False