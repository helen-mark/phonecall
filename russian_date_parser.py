import re
from datetime import datetime, timedelta
from dateutil import parser
from typing import Dict, List, Optional, Tuple
import json
from collections import defaultdict


class RussianDateParser:
    """Парсер русских временных выражений с регулярками"""

    def __init__(self, reference_date: datetime = None):
        self.now = reference_date or datetime.now()
        self.today = self.now.date()

        # ОСНОВНЫЕ РЕГУЛЯРКИ (95% покрытие)
        self.patterns = {
            # 1. Абсолютные даты (01.01.2024, 1 января 2024)
            'date_dmy': r'(\d{1,2})[\.\/\-](\d{1,2})[\.\/\-](\d{4})',
            'date_ymd': r'(\d{4})[\.\/\-](\d{1,2})[\.\/\-](\d{1,2})',
            'date_words': r'(\d{1,2})\s+(январ[ья]|феврал[ья]|март[а]?|апрел[ья]|ма[йя]|июн[ья]|июл[ья]|август[а]?|сентябр[ья]|октябр[ья]|ноябр[ья]|декабр[ья])\s+(\d{4})',

            # 2. Относительные периоды (последние N единиц)
            'last_n_days': r'последни[ех]?\s*(\d+)\s*дн(?:ей|я|ю)?\b',
            'last_n_weeks': r'последни[ех]?\s*(\d+)\s*недел[ьию]?\b',
            'last_n_months': r'последни[ех]?\s*(\d+)\s*месяц(?:ев|а|е)?\b',
            'last_n_years': r'последни[ех]?\s*(\d+)\s*год(?:ов|а|у)?\b',
            'last_n_hours': r'последни[ех]?\s*(\d+)\s*час(?:ов|а)?\b',

            # 3. Специальные периоды
            'today': r'\bсегодн[яя]\b',
            'yesterday': r'\bвчера\b',
            'tomorrow': r'\bзавтра\b',
            'this_week': r'\bна\s*этой\s*недел[еи]\b',
            'last_week': r'\bна\s*прошлой\s*недел[еи]\b',
            'next_week': r'\bна\s*следующей\s*недел[еи]\b',
            'this_month': r'\bв\s*этом\s*месяц[ее]\b',
            'last_month': r'\bв\s*прошлом\s*месяц[ее]\b',
            'next_month': r'\bв\s*следующем\s*месяц[ее]\b',
            'this_year': r'\bв\s*этом\s*год[уу]\b',
            'last_year': r'\bв?\s*прошлом\s*год[уу]\b',
            'next_year': r'\bв?\s*следующем\s*год[уу]\b',

            # 4. Кварталы
            'quarter': r'(\d+)-?[ыи]?\s*квартал\s*(\d{4})?',
            'last_quarter': r'последни[ей]?\s*квартал',

            # 5. Сложные комбинации (ваш случай!)
            'last_n_of_year': r'последни[ех]?\s*(\d+)\s*месяц(?:ев|а|е)?\s*(прошлого|текущего|этого)\s+год[ау]?\b',
            'first_n_of_year': r'первы[ех]?\s*(\d+)\s*месяц(?:ев|а|е)?\s*(прошлого|текущего|этого)\s+год[ау]?\b',

            # 6. Периоды "с ... по ..."
            'from_to': r'с\s+(.+?)\s+по\s+(.+)',
            'between': r'между\s+(.+?)\s+и\s+(.+)',

            # 7. Без точной даты
            'recently': r'\bнедавно\b|\bна\s*днях\b',
            'lately': r'\bв\s*последнее\s*время\b',
            'in_past': r'\bв\s*прошлом\b',
        }

        # Месяцы для конвертации
        self.months = {
            'январ': 1, 'феврал': 2, 'март': 3,
            'апрел': 4, 'май': 5, 'мая': 5, 'июн': 6,
            'июл': 7, 'август': 8, 'сентябр': 9,
            'октябр': 10, 'ноябр': 11, 'декабр': 12
        }

    def parse_query(self, query: str) -> Dict:
        """Основной метод парсинга запроса"""
        query_lower = query.lower().strip()
        original_query = query

        print(f"🔍 Парсим запрос: '{query}'")

        # Сначала пробуем сложные комбинации
        result = self._parse_complex_combinations(query_lower)
        if result:
            print(f"   ✅ Распознано как сложный паттерн: {result['description']}")
            return result

        # Затем стандартные периоды
        result = self._parse_standard_periods(query_lower)
        if result:
            print(f"   ✅ Распознано как стандартный период: {result['description']}")
            return result

        # Абсолютные даты
        result = self._parse_absolute_dates(query_lower)
        if result:
            print(f"   ✅ Распознаны абсолютные даты: {result['description']}")
            return result

        # Fallback: dateutil для всего остального
        result = self._try_dateutil(query)
        if result:
            print(f"   ⚠️  Распознано dateutil: {result['description']}")
            return result

        print(f"   ❌ Не удалось распознать период")
        return {
            'type': 'unclear',
            'start': None,
            'end': None,
            'description': 'Не удалось распознать период',
            'original_query': original_query,
            'confidence': 0
        }

    def _parse_complex_combinations(self, query: str) -> Optional[Dict]:
        """Парсинг сложных комбинаций типа 'последние 6 месяцев прошлого года'"""

        # 1. Последние N месяцев прошлого/этого года
        match = re.search(self.patterns['last_n_of_year'], query)
        if match:
            n_months = int(match.group(1))
            year_type = match.group(2)  # 'прошлого', 'текущего', 'этого'

            if year_type == 'прошлого':
                year = self.now.year - 1
                end_date = datetime(year, 12, 31)
                start_month = 13 - n_months
                start_date = datetime(year, start_month, 1)

                return {
                    'type': 'last_n_months_of_year',
                    'start': start_date,
                    'end': end_date,
                    'description': f'Последние {n_months} месяцев {year} года',
                    'confidence': 0.95
                }

        # 2. Первые N месяцев года
        match = re.search(self.patterns['first_n_of_year'], query)
        if match:
            n_months = int(match.group(1))
            year_type = match.group(2)

            if year_type == 'прошлого':
                year = self.now.year - 1
                start_date = datetime(year, 1, 1)
                end_date = datetime(year, n_months, 1) + timedelta(days=32)
                end_date = end_date.replace(day=1) - timedelta(days=1)

                return {
                    'type': 'first_n_months_of_year',
                    'start': start_date,
                    'end': end_date,
                    'description': f'Первые {n_months} месяцев {year} года',
                    'confidence': 0.95
                }

        # 3. Период "с ... по ..."
        match = re.search(self.patterns['from_to'], query)
        if match:
            date1_str, date2_str = match.groups()
            date1 = self._parse_single_date(date1_str.strip())
            date2 = self._parse_single_date(date2_str.strip())

            if date1 and date2:
                # Убедимся, что date1 <= date2
                start_date, end_date = sorted([date1, date2])

                return {
                    'type': 'from_to',
                    'start': start_date,
                    'end': end_date,
                    'description': f'С {date1.strftime("%d.%m.%Y")} по {date2.strftime("%d.%m.%Y")}',
                    'confidence': 0.9
                }

        return None

    def _parse_standard_periods(self, query: str) -> Optional[Dict]:
        """Парсинг стандартных относительных периодов"""

        # Последние N дней/недель/месяцев/лет
        patterns = [
            ('last_n_days', 'days'),
            ('last_n_weeks', 'weeks'),
            ('last_n_months', 'months'),
            ('last_n_years', 'years'),
        ]

        for pattern_key, unit in patterns:
            match = re.search(self.patterns[pattern_key], query)
            if match:
                n = int(match.group(1))
                end_date = self.now

                if unit == 'days':
                    start_date = self.now - timedelta(days=n)
                elif unit == 'weeks':
                    start_date = self.now - timedelta(weeks=n)
                elif unit == 'months':
                    start_date = self.now - timedelta(days=30 * n)  # Приблизительно
                elif unit == 'years':
                    start_date = self.now - timedelta(days=365 * n)

                return {
                    'type': f'last_{n}_{unit}',
                    'start': start_date,
                    'end': end_date,
                    'description': f'Последние {n} {self._get_unit_name(n, unit)}',
                    'confidence': 0.9
                }

        # Специальные периоды
        special_cases = {
            'today': (self.today, self.today, 'Сегодня'),
            'yesterday': (self.today - timedelta(days=1), self.today - timedelta(days=1), 'Вчера'),
            'this_week': (self.today - timedelta(days=self.today.weekday()), self.today, 'На этой неделе'),
            'last_week': (self.today - timedelta(days=self.today.weekday() + 7),
                          self.today - timedelta(days=self.today.weekday() + 1), 'На прошлой неделе'),
            'this_month': (datetime(self.today.year, self.today.month, 1),
                           self.today, 'В этом месяце'),
            'last_month': (self._first_day_of_month(self.today - timedelta(days=31)),
                           self._last_day_of_month(self.today - timedelta(days=31)), 'В прошлом месяце'),
            'this_year': (datetime(self.today.year, 1, 1), self.today, 'В этом году'),
            'last_year': (datetime(self.today.year - 1, 1, 1),
                          datetime(self.today.year - 1, 12, 31), 'В прошлом году'),
        }

        for pattern_key, (start, end, desc) in special_cases.items():
            if re.search(self.patterns[pattern_key], query):
                return {
                    'type': pattern_key,
                    'start': start,
                    'end': end,
                    'description': desc,
                    'confidence': 0.95
                }

        return None

    def _parse_absolute_dates(self, query: str) -> Optional[Dict]:
        """Парсинг абсолютных дат"""

        # Формат ДД.ММ.ГГГГ
        match = re.search(self.patterns['date_dmy'], query)
        if match:
            day, month, year = map(int, match.groups())
            date = datetime(year, month, day)

            # Если это одна дата, ищем звонки за этот день
            return {
                'type': 'single_date',
                'start': date.replace(hour=0, minute=0, second=0),
                'end': date.replace(hour=23, minute=59, second=59),
                'description': f'За {day:02d}.{month:02d}.{year}',
                'confidence': 0.99
            }

        # Словарный формат "1 января 2024"
        match = re.search(self.patterns['date_words'], query)
        if match:
            day = int(match.group(1))
            month_word = match.group(2)
            year = int(match.group(3))

            # Находим номер месяца
            month_num = None
            for month_prefix, num in self.months.items():
                if month_word.startswith(month_prefix):
                    month_num = num
                    break

            if month_num:
                date = datetime(year, month_num, day)
                return {
                    'type': 'single_date_words',
                    'start': date.replace(hour=0, minute=0, second=0),
                    'end': date.replace(hour=23, minute=59, second=59),
                    'description': f'За {day} {month_word} {year} года',
                    'confidence': 0.98
                }

        return None

    def _parse_single_date(self, date_str: str) -> Optional[datetime]:
        """Парсинг одиночной даты из строки"""
        # Пробуем разные форматы
        for pattern in [self.patterns['date_dmy'], self.patterns['date_words']]:
            match = re.search(pattern, date_str)
            if match:
                if pattern == self.patterns['date_dmy']:
                    day, month, year = map(int, match.groups())
                    return datetime(year, month, day)
                else:
                    day = int(match.group(1))
                    month_word = match.group(2)
                    year = int(match.group(3))
                    for prefix, num in self.months.items():
                        if month_word.startswith(prefix):
                            return datetime(year, num, day)
        return None

    def _try_dateutil(self, query: str) -> Optional[Dict]:
        """Fallback через dateutil"""
        try:
            # Пробуем распознать как период
            if 'по' in query or 'с' in query:
                return None  # Пропускаем, т.к. это уже обработано

            result = parser.parse(query, fuzzy=True)
            if result:
                return {
                    'type': 'dateutil',
                    'start': result.replace(hour=0, minute=0, second=0),
                    'end': result.replace(hour=23, minute=59, second=59),
                    'description': f'За {result.strftime("%d.%m.%Y")}',
                    'confidence': 0.7
                }
        except:
            pass
        return None

    def _get_unit_name(self, n: int, unit: str) -> str:
        """Склонение единиц измерения"""
        if unit == 'days':
            if n % 10 == 1 and n % 100 != 11:
                return 'день'
            elif 2 <= n % 10 <= 4 and (n % 100 < 10 or n % 100 >= 20):
                return 'дня'
            else:
                return 'дней'
        elif unit == 'months':
            if n % 10 == 1 and n % 100 != 11:
                return 'месяц'
            elif 2 <= n % 10 <= 4 and (n % 100 < 10 or n % 100 >= 20):
                return 'месяца'
            else:
                return 'месяцев'
        # ... аналогично для других единиц
        return unit

    def _first_day_of_month(self, date):
        return date.replace(day=1)

    def _last_day_of_month(self, date):
        if date.month == 12:
            return date.replace(day=31)
        return date.replace(month=date.month + 1, day=1) - timedelta(days=1)