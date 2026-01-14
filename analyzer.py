from llm_query_planner import  AnalysisPlan, Dict
import ollama
import json

class DeepSeekAnalyzer:
    """LLM, которая анализирует результаты и формулирует ответ"""

    def __init__(self, model_name: str):
        self.client = ollama.Client()
        self.model_name = model_name

    def generate_answer(self, user_query: str, analysis_results: Dict[str, Any],
                        analysis_plan: AnalysisPlan) -> str:
        """Генерирует итоговый ответ на основе результатов анализа"""

        prompt = self._build_analyzer_prompt(user_query, analysis_results, analysis_plan)

        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            options={'temperature': 0.3, 'num_predict': 800}
        )

        return response['response']

    def _build_analyzer_prompt(self, user_query: str, results: Dict,
                               plan: AnalysisPlan) -> str:
        """Строит промпт для аналитика"""

        # Форматируем результаты для LLM
        results_str = json.dumps(results, ensure_ascii=False, indent=2, default=str)

        # Форматируем план
        plan_str = f"""
        Период анализа: {plan.time_period['start'].strftime('%Y-%m-%d')} - {plan.time_period['end'].strftime('%Y-%m-%d')}
        Анализируемые теги: {', '.join(plan.target_tags)}
        Метрики: {[m.value for m in plan.metrics]}
        Группировка: {plan.grouping}
        """

        return f"""Ты — бизнес-аналитик компании по аренде штор и ковров.

ИСХОДНЫЙ ЗАПРОС ПОЛЬЗОВАТЕЛЯ: "{user_query}"

КАК МЫ АНАЛИЗИРОВАЛИ:
{plan_str}

РЕЗУЛЬТАТЫ АНАЛИЗА (сырые данные):
{results_str}

ТВОЯ ЗАДАЧА:
1. Проанализировать результаты
2. Сформулировать понятный ответ на русском языке
3. Выделить ключевые инсайты и тренды
4. Если есть данные по месяцам — показать динамику
5. Дать рекомендации если уместно
6. Будь конкретен, используй числа из данных

ФОРМАТ ОТВЕТА:
1. Краткий вывод (1-2 предложения)
2. Детальный анализ с цифрами
3. Визуальное описание тренда (если есть данные по времени)
4. Рекомендации (если уместно)

ПРИМЕР ХОРОШЕГО ОТВЕТА:
"За последние 6 месяцев жалоб на доставку стало на 40% меньше. Пик был в январе (60 жалоб), но к июню снизился до 15 в месяц. Рекомендуем продолжить текущую стратегию улучшения логистики."

ТВОЙ ОТВЕТ:"""