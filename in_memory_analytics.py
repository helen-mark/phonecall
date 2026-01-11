import sqlite3
from contextlib import contextmanager
import os
import json

class InMemoryJSONAnalytics:
    """Загружает JSON в оперативную SQLite для сложных запросов"""

    def __init__(self, json_dir: str):
        self.json_dir = json_dir
        self.conn = sqlite3.connect(':memory:')  # База в оперативке
        self._create_schema()
        self._load_json_files()

    def _create_schema(self):
        """Создает схему таблиц в памяти"""
        cursor = self.conn.cursor()

        # Основная таблица звонков
        cursor.execute("""
        CREATE TABLE calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            call_date TEXT,
            year INTEGER,
            month INTEGER,
            day INTEGER,
            full_text TEXT,
            summary TEXT,
            tags_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Таблица тегов (развернутая для быстрого поиска)
        cursor.execute("""
        CREATE TABLE call_tags (
            call_id INTEGER,
            tag TEXT,
            FOREIGN KEY (call_id) REFERENCES calls(id)
        )
        """)

        # Индексы для быстрого поиска
        cursor.execute("CREATE INDEX idx_tags ON call_tags(tag)")
        cursor.execute("CREATE INDEX idx_date ON calls(call_date)")

        self.conn.commit()

    def _load_json_files(self, limit: int = None):
        """Загружает JSON файлы в SQLite"""
        cursor = self.conn.cursor()
        files_processed = 0

        for filename in sorted(os.listdir(self.json_dir)):
            if not filename.endswith('.json'):
                continue

            filepath = os.path.join(self.json_dir, filename)

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Извлекаем дату
                import re
                date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', filename)

                if date_match:
                    call_date = date_match.group(0)
                    year, month, day = map(int, date_match.groups())
                else:
                    call_date = 'unknown'
                    year = month = day = 0

                # Вставляем в таблицу calls
                cursor.execute("""
                INSERT INTO calls (file_name, call_date, year, month, day, 
                                  full_text, summary, tags_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    filename,
                    call_date,
                    year,
                    month,
                    day,
                    data.get('text', ''),
                    data.get('reason', ''),
                    json.dumps(data.get('tags', []), ensure_ascii=False)
                ))

                call_id = cursor.lastrowid

                # Вставляем теги в отдельную таблицу
                tags = data.get('tags', [])
                for tag in tags:
                    cursor.execute(
                        "INSERT INTO call_tags (call_id, tag) VALUES (?, ?)",
                        (call_id, tag)
                    )

                files_processed += 1

                if limit and files_processed >= limit:
                    break

            except Exception as e:
                print(f"Ошибка загрузки {filename}: {e}")

        self.conn.commit()
        print(f"✅ Загружено {files_processed} звонков в оперативную БД")

    @contextmanager
    def get_cursor(self):
        """Контекстный менеджер для курсора"""
        cursor = self.conn.cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    def execute_analysis(self, sql_query: str, params: tuple = ()):
        """Выполняет SQL запрос к данным"""
        with self.get_cursor() as cursor:
            cursor.execute(sql_query, params)
            return cursor.fetchall()

    def analyze_complaints(self, tag_keyword: str, months: int = 6):
        """Пример анализа жалоб по тегу"""

        # SQL запрос для анализа
        query = """
        SELECT 
            strftime('%Y-%m', call_date) as month,
            COUNT(DISTINCT c.id) as complaint_count
        FROM calls c
        JOIN call_tags ct ON c.id = ct.call_id
        WHERE ct.tag LIKE ?
          AND c.call_date >= date('now', ?)
        GROUP BY month
        ORDER BY month DESC
        LIMIT ?
        """

        params = (
            f'%{tag_keyword}%',  # Ищем тег с ключевым словом
            f'-{months} months',  # Последние N месяцев
            months + 1  # Лимит
        )

        results = self.execute_analysis(query, params)

        # Форматируем результат
        return [
            {'month': row[0], 'count': row[1]}
            for row in results
        ]

    def get_top_tags(self, limit: int = 10, period_months: int = None):
        """Топ тегов за период"""

        if period_months:
            date_filter = "WHERE c.call_date >= date('now', ?)"
            params = (f'-{period_months} months', limit)
        else:
            date_filter = ""
            params = (limit,)

        query = f"""
        SELECT 
            ct.tag,
            COUNT(DISTINCT c.id) as tag_count
        FROM calls c
        JOIN call_tags ct ON c.id = ct.call_id
        {date_filter}
        GROUP BY ct.tag
        ORDER BY tag_count DESC
        LIMIT ?
        """

        results = self.execute_analysis(query, params)

        return [
            {'tag': row[0], 'count': row[1]}
            for row in results
        ]


# Использование
print("🚀 Загружаю JSON файлы в оперативную SQLite...")
analytics = InMemoryJSONAnalytics('/путь/к/json/файлам')

# Примеры аналитических запросов
print("\n📊 Анализ жалоб на качество за последние 6 месяцев:")
quality_complaints = analytics.analyze_complaints('качеств', months=6)
for item in quality_complaints:
    print(f"  {item['month']}: {item['count']} жалоб")

print("\n🏆 Топ-10 тегов за все время:")
top_tags = analytics.get_top_tags(limit=10)
for i, item in enumerate(top_tags, 1):
    print(f"  {i}. {item['tag']}: {item['count']}")