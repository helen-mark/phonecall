from datetime import datetime
import re
import pandas as pd

def _extract_date_from_filename(filename: str) -> datetime:
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


    # Fallback: текущая дата
    return datetime.now()

def convert_json_to_csv(json_dir: str, output_csv: str):
    """Конвертирует все JSON файлы в единый CSV"""
    import pandas as pd
    from datetime import datetime
    import json
    import os

    all_data = []

    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(json_dir, filename)

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Извлекаем дату из имени файла
                call_date = _extract_date_from_filename(filename)

                all_data.append({
                    'date': call_date,
                    'text': data.get('transcription').get('text', ''),
                    'tags': data.get('tags', {}).get('fixed_tags', []),
                    'summary': data.get('reason', ''),
                    'source_file': filename
                })

            except Exception as e:
                print(f"⚠️  Ошибка обработки {filename}: {e}")

    if not all_data:
        print("❌ Нет данных для конвертации")
        return None

    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"✅ Конвертировано {len(all_data)} записей в {output_csv}")
    return output_csv


if __name__ == '__main__':
    convert_json_to_csv('json_calls', 'calls.csv')