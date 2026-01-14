import sqlite3
from datetime import datetime, timedelta
from llm_query_planner import AnalysisPlan, MetricType, List, Dict

class DatabaseExecutor:
    """Выполняет запросы к базе данных по плану от LLM"""

    def __init__(self, db_path: str = "calls_database.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_database()

    def _init_database(self):
        """Инициализация таблиц (если не существует)"""
        cursor = self.conn.cursor()

        # Таблица звонков
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            call_date TIMESTAMP NOT NULL,
            full_text TEXT NOT NULL,
            summary TEXT NOT NULL,
            tags_json TEXT NOT NULL,  -- JSON массив тегов
            duration_sec INTEGER,
            customer_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Индекс для быстрого поиска по дате
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_call_date ON calls(call_date)")

        # Виртуальная таблица для полнотекстового поиска по тегам
        cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS calls_tags_fts 
        USING fts5(tags_json, content='calls', content_rowid='id')
        """)

        self.conn.commit()

    def execute_analysis_plan(self, plan: AnalysisPlan) -> Dict[str, Any]:
        """Выполняет план анализа и возвращает данные"""

        results = {}

        # Для каждой метрики в плане
        for metric in plan.metrics:
            if metric == MetricType.COUNT_BY_TAG:
                results['count_by_tag'] = self._get_counts_by_tag(
                    plan.time_period['start'],
                    plan.time_period['end'],
                    plan.target_tags
                )

            elif metric == MetricType.TAG_TRENDS:
                results['tag_trends'] = self._get_tag_trends(
                    plan.time_period['start'],
                    plan.time_period['end'],
                    plan.target_tags[0] if plan.target_tags else None,
                    plan.grouping
                )

            elif metric == MetricType.TOP_N_TAGS:
                results['top_n_tags'] = self._get_top_n_tags(
                    plan.time_period['start'],
                    plan.time_period['end'],
                    n=5
                )

            elif metric == MetricType.COMPARISON:
                results['comparison'] = self._compare_tags(
                    plan.time_period['start'],
                    plan.time_period['end'],
                    plan.comparison_tags or plan.target_tags[:2]
                )

        return results

    def _get_counts_by_tag(self, start_date: datetime, end_date: datetime,
                           tags: List[str]) -> Dict[str, int]:
        """Количество звонков по тегам за период"""
        cursor = self.conn.cursor()

        # Для каждого тега делаем запрос
        results = {}
        for tag in tags:
            query = """
            SELECT COUNT(*) 
            FROM calls 
            WHERE call_date BETWEEN ? AND ?
            AND tags_json LIKE ?
            """

            # Ищем тег в JSON массиве
            tag_pattern = f'%"{tag}"%'

            cursor.execute(query, (start_date, end_date, tag_pattern))
            count = cursor.fetchone()[0]
            results[tag] = count

        return results

    def _get_tag_trends(self, start_date: datetime, end_date: datetime,
                        tag: str, grouping: str = "month") -> List[Dict]:
        """Динамика тега по периодам (месяцам/неделям)"""
        cursor = self.conn.cursor()

        if grouping == "month":
            period_format = "strftime('%Y-%m', call_date)"
            period_name = "month"
        else:  # week
            period_format = "strftime('%Y-%W', call_date)"
            period_name = "week"

        query = f"""
        SELECT 
            {period_format} as period,
            COUNT(*) as count
        FROM calls 
        WHERE call_date BETWEEN ? AND ?
        AND tags_json LIKE ?
        GROUP BY period
        ORDER BY period
        """

        tag_pattern = f'%"{tag}"%' if tag else '%'
        cursor.execute(query, (start_date, end_date, tag_pattern))

        rows = cursor.fetchall()

        return [
            {period_name: row[0], 'count': row[1]}
            for row in rows
        ]

    def _get_top_n_tags(self, start_date: datetime, end_date: datetime,
                        n: int = 5) -> List[Dict]:
        """Топ-N самых частых тегов за период"""
        cursor = self.conn.cursor()

        # Сложный запрос для агрегации тегов из JSON
        query = """
        WITH tag_counts AS (
            SELECT 
                json_each.value as tag,
                COUNT(*) as count
            FROM calls, json_each(json(calls.tags_json))
            WHERE call_date BETWEEN ? AND ?
            GROUP BY json_each.value
            ORDER BY count DESC
            LIMIT ?
        )
        SELECT tag, count FROM tag_counts
        """

        cursor.execute(query, (start_date, end_date, n))
        rows = cursor.fetchall()

        return [{'tag': row[0], 'count': row[1]} for row in rows]

    def _compare_tags(self, start_date: datetime, end_date: datetime,
                      tags: List[str]) -> Dict[str, Any]:
        """Сравнение двух тегов"""
        if len(tags) < 2:
            tags = tags + [None] * (2 - len(tags))

        counts = self._get_counts_by_tag(start_date, end_date, tags[:2])

        # Дополнительная статистика
        total_calls = self._get_total_calls_count(start_date, end_date)

        return {
            'tag1': {'name': tags[0], 'count': counts.get(tags[0], 0)},
            'tag2': {'name': tags[1], 'count': counts.get(tags[1], 0)},
            'total_calls': total_calls,
            'percentage1': counts.get(tags[0], 0) / total_calls * 100 if total_calls > 0 else 0,
            'percentage2': counts.get(tags[1], 0) / total_calls * 100 if total_calls > 0 else 0
        }

    def _get_total_calls_count(self, start_date: datetime, end_date: datetime) -> int:
        """Общее количество звонков за период"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM calls WHERE call_date BETWEEN ? AND ?",
                       (start_date, end_date))
        return cursor.fetchone()[0]

