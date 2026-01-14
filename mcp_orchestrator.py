import json
import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import ollama
from collections import defaultdict, Counter
import sqlite3
from contextlib import contextmanager


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


# ==================== JSON Data Loader ====================

class JSONDataLoader:
    """Загружает и управляет данными из JSON файлов"""

    def __init__(self, json_directory: str):
        self.json_dir = json_directory
        self.calls_cache = None
        self.conn = None  # In-memory SQLite соединение

    def load_all_calls(self, limit: int = None) -> List[Dict]:
        """Загружает все звонки из JSON файлов"""
        if self.calls_cache is not None:
            return self.calls_cache[:limit] if limit else self.calls_cache

        all_calls = []
        files_processed = 0

        for filename in sorted(os.listdir(self.json_dir)):
            if not filename.endswith('.json'):
                continue

            filepath = os.path.join(self.json_dir, filename)

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Извлекаем дату из имени файла
                call_date = self._extract_date_from_filename(filename)

                # Формируем структурированную запись
                call_record = {
                    'id': f"call_{files_processed}",
                    'file_name': filename,
                    'call_date': call_date,
                    'year': call_date.year,
                    'month': call_date.month,
                    'day': call_date.day,
                    'full_text': data.get('text', ''),
                    'summary': data.get('reason', ''),
                    'tags': data.get('tags').get('fixed_tags', []),
                    'text_length': len(data.get('text', '')),
                    'source_file': filepath
                }

                all_calls.append(call_record)
                files_processed += 1

                if limit and files_processed >= limit:
                    break

            except Exception as e:
                print(f"⚠️  Ошибка загрузки {filename}: {e}")

        self.calls_cache = all_calls
        print(f"✅ Загружено {len(all_calls)} звонков из JSON файлов")
        return all_calls

    def _extract_date_from_filename(self, filename: str) -> datetime:
        """Извлекает дату из имени файла"""
        # Паттерны для поиска даты в имени файла
        patterns = [
            r'(\d{4})-(\d{2})-(\d{2})',  # YYYY-MM-DD
            # r'(\d{2})\.(\d{2})\.(\d{4})',  # DD.MM.YYYY
            # r'(\d{4})(\d{2})(\d{2})',  # YYYYMMDD
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
        filepath = os.path.join(self.json_dir, filename)
        if os.path.exists(filepath):
            return datetime.fromtimestamp(os.path.getmtime(filepath))

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
            text_length INTEGER
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
                              full_text, summary, tags_json, text_length)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                call['text_length']
            ))

            # Вставляем теги
            for tag in call['tags']:
                cursor.execute(
                    "INSERT INTO call_tags (call_id, tag) VALUES (?, ?)",
                    (call['id'], tag)
                )

        self.conn.commit()
        print(f"✅ Данные загружены в in-memory SQLite ({len(calls)} записей)")
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

    def __init__(self, model_name):
        self.client = ollama.Client()
        self.model_name = model_name
        self.available_tags = self._load_available_tags()

    def _load_available_tags(self) -> List[str]:
        """Загружает все уникальные теги из JSON файлов (для примера)"""
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
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                format="json",
                options={'temperature': 0.1, 'num_predict': 500}
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
        print('Current date:', current_date)

        return f"""Ты — аналитик базы телефонных звонков компании по аренде штор.

ЗАПРОС: "{user_query}"

ТВОЯ ЗАДАЧА: Создать план анализа.
Система будет обращаться по твоему плану к текстам с записями телефонных звонков клиентов за несколько последних лет, содержащими описательные теги каждого звонка.

ДОСТУПНЫЕ ТЕГИ:
{', '.join(self.available_tags)}

МЕТРИКИ, которые система может посчитать для ответа на запрос, если это необходимо:
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

Ответ:"""

    def _parse_time_period(self, period_data: Dict) -> Dict[str, Any]:
        """Парсит временной период"""
        today = datetime.now()
        description = period_data.get('description', '')

        # Определяем период на основе описания
        # if 'последние 6 месяцев' in description.lower():
        #     start = today - timedelta(days=30 * 6)
        #     end = today
        # elif 'этот месяц' in description.lower():
        #     start = datetime(today.year, today.month, 1)
        #     end = today
        # elif 'этот год' in description.lower():
        #     start = datetime(today.year, 1, 1)
        #     end = today
        # elif 'первый квартал 2024' in description.lower():
        #     start = datetime(2024, 1, 1)
        #     end = datetime(2024, 3, 31)
        # else:
        #     # По умолчанию - последний месяц
        #     start = today - timedelta(days=30)
        #     end = today

        # Переопределяем если указаны точные даты
        if period_data.get('start'):
            start = datetime.fromisoformat(period_data['start'])
        if period_data.get('end'):
            end = datetime.fromisoformat(period_data['end'])

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

        return valid_tags or ['жалоба_качество_стирки']  # Fallback

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
        if 'качеств' in user_query.lower():
            target_tags.append('жалоба_качество_стирки')
        if 'доставк' in user_query.lower():
            target_tags.append('жалоба_долгая_доставка')

        return AnalysisPlan(
            time_period={
                'start': today - timedelta(days=30 * 6),
                'end': today,
                'description': 'последние 6 месяцев'
            },
            target_tags=target_tags or ['жалоба_качество_стирки'],
            metrics=[MetricType.COUNT_BY_TAG, MetricType.TAG_TRENDS],
            grouping='month'
        )


# ==================== Query Executor ====================

class JSONQueryExecutor:
    """Выполняет аналитические запросы к JSON данным"""

    def __init__(self, data_loader: JSONDataLoader):
        self.data_loader = data_loader

    def execute_plan(self, plan: AnalysisPlan) -> Dict[str, Any]:
        """Выполняет план анализа"""

        results = {}

        # Получаем данные за период
        all_calls = self.data_loader.load_all_calls()
        print(f'{len(all_calls)} calls in total')
        print(all_calls[:2])
        filtered_calls = self._filter_calls_by_period(all_calls, plan.time_period)
        print(f'{len(filtered_calls)} calls after filtering')

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

            print(f'executions result: {results}')

        # Добавляем общую статистику
        results['summary_stats'] = {
            'total_calls': len(filtered_calls),
            'period': plan.time_period['description'],
            'date_range': f"{plan.time_period['start'].strftime('%Y-%m-%d')} - {plan.time_period['end'].strftime('%Y-%m-%d')}"
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
        if not target_tags:
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

    def __init__(self, model_name: str):
        self.client = ollama.Client()
        self.model_name = model_name

    def generate_answer(self, user_query: str, results: Dict, plan: AnalysisPlan) -> str:
        """Генерирует итоговый ответ на основе результатов"""

        prompt = self._build_analyzer_prompt(user_query, results, plan)

        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={'temperature': 0.3, 'num_predict': 800}
            )

            return response['response'].strip()

        except Exception as e:
            print(f"❌ Ошибка анализатора: {e}")
            return self._generate_fallback_answer(results, plan)

    def _build_analyzer_prompt(self, user_query: str, results: Dict, plan: AnalysisPlan) -> str:
        """Строит промпт для анализатора"""

        # Форматируем результаты для промпта
        results_str = json.dumps(results, ensure_ascii=False, indent=2, default=str)
        print(f'Generating answer using plan results: {results} for plan: {plan}')
        print(f'User query: {user_query}')

        return f"""Ты — старший аналитик компании по аренде ковров.

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
        if 'count_by_tag' in results:
            answer_parts.append("\n📈 Количество звонков по тегам:")
            for tag, count in results['count_by_tag'].items():
                answer_parts.append(f"  • {tag}: {count}")

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
        if 'count_by_tag' in results:
            max_tag = max(results['count_by_tag'].items(), key=lambda x: x[1])[0] if results['count_by_tag'] else None
            if max_tag and 'жалоба' in max_tag:
                answer_parts.append(
                    f"\n💡 Рекомендация: Обратите внимание на тег '{max_tag}' - это самая частая категория обращений")

        return "\n".join(answer_parts)


# ==================== Главная MCP система ====================

class JSONCallAnalyticsMCP:
    """Главная MCP система для работы с JSON файлами"""

    def __init__(self, json_directory: str, model_name: str):
        self.data_loader = JSONDataLoader(json_directory)
        self.planner = DeepSeekPlanner(model_name)
        self.executor = JSONQueryExecutor(self.data_loader)
        self.analyzer = DeepSeekAnalyzer(model_name)

        # Загружаем данные при инициализации
        print("📂 Загружаю данные из JSON файлов...")
        self.total_calls = len(self.data_loader.load_all_calls())
        print(f"✅ Загружено {self.total_calls} звонков")

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Обрабатывает запрос пользователя"""

        print(f"\n🔍 Анализирую запрос: '{user_query}'")

        # 1. Планирование (LLM)
        print("🤖 Создаю план анализа...")
        analysis_plan = self.planner.create_analysis_plan(user_query)

        print(f"   📅 Период: {analysis_plan.time_period['description'], analysis_plan.time_period['start'], analysis_plan.time_period['end']}")
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
            'model_used': self.planner.model_name
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

        if 'count_by_tag' in results:
            counts = results['count_by_tag']
            if counts:
                print("\n📊 Количество по тегам:")
                for tag, count in counts.items():
                    print(f"  • {tag}: {count}")

        if 'top_n_tags' in results:
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
            'data_source': 'JSON files'
        }

    def test_system(self) -> bool:
        """Тестирует работоспособность системы"""
        print("🧪 Тестирую систему...")

        try:
            # Тест 1: Загрузка данных
            calls = self.data_loader.load_all_calls(limit=10)
            if not calls:
                print("❌ Нет данных для анализа")
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