import ollama
import re
from dateutil import parser
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import json
from dataclasses import dataclass
from enum import Enum


# Структуры данных
@dataclass
class CallRecord:
    """Запись звонка в базе"""
    id: int
    call_date: datetime
    full_text: str
    summary: str  # Краткое описание
    tags: List[str]  # ["жалоба_качество", "доставка"]
    duration_sec: int
    customer_id: str


class MetricType(Enum):
    """Типы метрик для анализа"""
    COUNT_BY_TAG = "count_by_tag"  # Количество звонков по тегу
    TOP_N_TAGS = "top_n_tags"  # Топ-N тегов за период
    TAG_TRENDS = "tag_trends"  # Динамика тега по месяцам
    SENTIMENT_TREND = "sentiment"  # Динамика тональности
    COMPARISON = "comparison"  # Сравнение двух тегов


@dataclass
class AnalysisPlan:
    """План анализа от LLM"""
    time_period: Dict[str, datetime]  # start, end
    target_tags: List[str]  # Какие теги анализировать
    metrics: List[MetricType]  # Какие метрики считать
    grouping: str = "month"  # Группировка: day/week/month
    comparison_tags: List[str] = None  # Для сравнения
    additional_filters: Dict = None  # Доп. фильтры

class DeepSeekPlanner:
    """LLM, которая преобразует запрос пользователя в план анализа"""

    def __init__(self, model_name):
        self.client = ollama.Client()
        self.model_name = model_name

        # Доступные теги из вашей базы
        self.available_tags = [
            "жалоба_качество_стирки",
            "жалоба_долгая_доставка",
            "жалоба_повреждение_изделия",
            "благодарность",
            "вопрос_оплата",
            "расторжение_договора",
            "консультация",
            "жалоба_менеджер",
            "срочный_вопрос",
            "перенос_доставки",
            "отмена_заказа"
        ]

    def create_analysis_plan(self, user_query: str) -> AnalysisPlan:
        """Преобразует запрос пользователя в план анализа"""

        prompt = self._build_planner_prompt(user_query)

        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            format="json",
            options={'temperature': 0.1, 'num_predict': 500}
        )

        plan_data = json.loads(response['response'])

        # Преобразуем в AnalysisPlan
        return AnalysisPlan(
            time_period=self._parse_time_period(plan_data.get('time_period', {})),
            target_tags=self._validate_tags(plan_data.get('target_tags', [])),
            metrics=self._parse_metrics(plan_data.get('metrics', [])),
            grouping=plan_data.get('grouping', 'month'),
            comparison_tags=plan_data.get('comparison_tags', []),
            additional_filters=plan_data.get('filters', {})
        )

    def _build_planner_prompt(self, user_query: str) -> str:
        """Строит промпт для планировщика"""

        current_date = datetime.now().strftime("%Y-%m-%d")

        return f"""Ты — SQL-аналитик базы телефонных звонков компании по аренде штор и ковров.

ЗАПРОС ПОЛЬЗОВАТЕЛЯ: "{user_query}"

ТВОЯ ЗАДАЧА:
1. Определить временной период для анализа
2. Выбрать релевантные теги из списка
3. Определить какие метрики посчитать
4. Создать план анализа

ДОСТУПНЫЕ ТЕГИ в базе:
{', '.join(self.available_tags)}

ВОЗМОЖНЫЕ МЕТРИКИ:
- count_by_tag: количество звонков с определенным тегом
- top_n_tags: топ-N самых частых тегов за период  
- tag_trends: динамика тега по месяцам/неделям
- comparison: сравнение двух тегов (например, жалоб на качество vs доставку)

ПРИМЕРЫ АНАЛИЗА:
1. "жалобы на качество за последний месяц" → период: последние 30 дней, тег: "жалоба_качество_стирки", метрика: count_by_tag
2. "какие основные жалобы в этом году?" → период: с начала года, метрика: top_n_tags (топ-5)
3. "динамика жалоб на доставку по месяцам" → период: последние 6 месяцев, тег: "жалоба_долгая_доставка", метрика: tag_trends
4. "сравнить жалобы на качество и доставку" → теги: ["жалоба_качество_стирки", "жалоба_долгая_доставка"], метрика: comparison

ТЕКУЩАЯ ДАТА: {current_date}

ВЕРНИ ОТВЕТ В ФОРМАТЕ JSON:
{{
  "time_period": {{
    "type": "relative/absolute",
    "start": "YYYY-MM-DD или null", 
    "end": "YYYY-MM-DD или null",
    "description": "например, 'последние 6 месяцев'"
  }},
  "target_tags": ["тег1", "тег2"],
  "metrics": ["count_by_tag", "tag_trends"],
  "grouping": "month/week/day",
  "comparison_tags": ["тег1", "тег2"],
  "filters": {{}}
}}

ОТВЕТ JSON:"""

    def _parse_time_period(self, period_data: Dict) -> Dict[str, datetime]:
        """Парсит временной период из ответа LLM"""
        today = datetime.now()

        if period_data.get('type') == 'relative':
            if 'последние 6 месяцев' in period_data.get('description', ''):
                start = today - timedelta(days=30 * 6)
                end = today
            elif 'этот месяц' in period_data.get('description', ''):
                start = datetime(today.year, today.month, 1)
                end = today
            elif 'этот год' in period_data.get('description', ''):
                start = datetime(today.year, 1, 1)
                end = today
            else:
                # По умолчанию последний месяц
                start = today - timedelta(days=30)
                end = today
        else:
            # Абсолютные даты
            start_str = period_data.get('start')
            end_str = period_data.get('end')

            start = parser.parse(start_str) if start_str else today - timedelta(days=30)
            end = parser.parse(end_str) if end_str else today

        return {'start': start, 'end': end}

    def _validate_tags(self, tags: List[str]) -> List[str]:
        """Проверяет, что теги есть в доступных"""
        valid_tags = []
        for tag in tags:
            # Проверяем точное совпадение или частичное
            for available_tag in self.available_tags:
                if tag.lower() in available_tag.lower() or available_tag.lower() in tag.lower():
                    valid_tags.append(available_tag)
                    break

        return valid_tags or self.available_tags[:1]  # Fallback

    def _parse_metrics(self, metrics: List[str]) -> List[MetricType]:
        """Парсит метрики из строк в enum"""
        metric_map = {
            'count_by_tag': MetricType.COUNT_BY_TAG,
            'top_n_tags': MetricType.TOP_N_TAGS,
            'tag_trends': MetricType.TAG_TRENDS,
            'comparison': MetricType.COMPARISON,
            'sentiment': MetricType.SENTIMENT_TREND
        }

        result = []
        for metric in metrics:
            if metric in metric_map:
                result.append(metric_map[metric])

        return result or [MetricType.COUNT_BY_TAG]