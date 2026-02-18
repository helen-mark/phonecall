import json
import os
from datetime import datetime
import ollama
from pathlib import Path
from phonecall.colab.reload_recursive import reload_recursive
import phonecall.colab.mcp_orchestrator

reload_recursive(phonecall.colab.mcp_orchestrator)
from typing import Union
#from llama_cpp import Llama


def enhanced_interactive_mode(_model, node_url = None, csv_dir: str = None, results_dir: str = None, drive_path: str = None):
    """Расширенный интерактивный режим с поддержкой Google Drive"""

    def show_help(in_drive_mode: bool):
        """Показывает справку по командам"""
        help_text = """
     КОМАНДЫ:

    АНАЛИТИЧЕСКИЕ ЗАПРОСЫ:
      Просто введите ваш вопрос, например:
      • "Сколько жалоб на качество в этом месяце?"
      • "Какие самые частые темы обращений?"
      • "Сравни жалобы на доставку и качество"

    СИСТЕМНЫЕ КОМАНДЫ:
      /? или /помощь      - эта справка
      /выход             - завершить работу
      /статистика        - статистика данных
      /история           - история запросов
      /очистить          - очистить экран
      /сохранить         - сохранить последний результат
      /тест              - протестировать систему
      /директории        - показать структуру данных
      /режим             - показать текущий режим работы
    """

        if in_drive_mode:
            help_text += """
    GOOGLE DRIVE КОМАНДЫ:
      • Все данные автоматически сохраняются в Drive
      • Результаты доступны в папке saved_results/
      • Модели кэшируются для ускорения работы

    ДОСТУП К ДАННЫМ:
      • Данные читаются из папки csv_calls/ в вашем Drive
      • Чтобы обновить данные, просто загрузите новые файлы в эту папку
    """

        print(help_text)

    def show_system_stats(system, in_drive_mode: bool):
        info = system.get_system_info()

        print("\n СТАТИСТИКА СИСТЕМЫ:")
        print("-" * 40)

        if in_drive_mode:
            print(" Режим: Google Drive")
            print("-" * 40)

        print(f" Всего звонков: {info['total_calls']}")
        print(f"  Уникальных тегов: {info['unique_tags_count']}")

        if info['date_range']['start']:
            start_date = datetime.fromisoformat(info['date_range']['start']).strftime('%d.%m.%Y')
            end_date = datetime.fromisoformat(info['date_range']['end']).strftime('%d.%m.%Y')
            print(f" Период данных: {start_date} - {end_date}")

        print(f" Средняя длина текста: {info['average_text_length']} симв.")
        print(f" Модель: {info['model']}")
        print(f" Источник: {info['data_source']}")

        if 'drive_path' in info and info['drive_path']:
            print(f" Google Drive путь: {info['drive_path']}")

    def show_query_history(history):
        if not history:
            print(" История запросов пуста")
            return

        print("\n ИСТОРИЯ ЗАПРОСОВ:")
        print("-" * 60)

        # Показываем последние 10 запросов
        for i, item in enumerate(reversed(history[-10:]), 1):
            time_str = item['timestamp'].strftime('%H:%M')
            status_icon = "✅" if item.get('status') == 'completed' else "❌" if item.get('status') == 'error' else "⏳"
            mode_icon = "🌐" if item.get('mode') == 'drive' else "💻"

            # Обрезаем длинные запросы
            query_preview = item['query']
            if len(query_preview) > 50:
                query_preview = query_preview[:47] + "..."

            print(f"{i}. [{time_str}] {mode_icon} {status_icon} {query_preview}")

            # Показываем время обработки если есть
            if item.get('processing_time'):
                print(f"   ⏱️  {item['processing_time']:.1f} сек")

        print("-" * 60)
        print(f"Всего запросов: {len(history)}")

    def save_last_result(history, results_dir):
        """Сохраняет последний результат"""
        if not history:
            print("❌ Нет результатов для сохранения")
            return

        # Ищем последний завершенный запрос
        completed_queries = [h for h in history if h.get('status') == 'completed' and 'result' in h]

        if not completed_queries:
            print("❌ Нет завершенных запросов для сохранения")
            return

        last_result = completed_queries[-1]['result']

        # Создаем имя файла
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = last_result['query'][:30].replace(' ', '_').replace('?', '').replace('/', '_')
        filename = f"result_{timestamp}_{safe_query}.json"
        filepath = os.path.join(results_dir, filename)

        # Создаем директорию если нет
        os.makedirs(results_dir, exist_ok=True)

        # Сохраняем
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(last_result, f, ensure_ascii=False, indent=2)

        print(f"✅ Результат сохранен в: {filepath}")

        # Если в Drive, показываем полный путь
        if '/drive/' in results_dir:
            print(f"   📍 Доступен в Google Drive")

    def test_system(system):
        """Тестирует систему"""
        print(" Тестирую систему...")

        test_queries = [
            "Сколько всего звонков в базе?",
            "Какие теги есть в данных?",
            "Тестовый запрос: жалобы"
        ]

        for query in test_queries:
            print(f"\n Тест: '{query}'")
            try:
                result = system.process_query(query)
                print(f"   ✅ Успешно, ответ: {len(result['answer'])} симв.")
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")

    def show_directories(csv_dir, results_dir, drive_path=None):
        """Показывает структуру директорий"""
        print("\n СТРУКТУРА ДИРЕКТОРИЙ:")
        print("-" * 50)

        if drive_path:
            print(f" Google Drive корень: {drive_path}")
            print("-" * 50)

        # Показываем JSON директорию
        print(f"Данные звонков ({csv_dir}):")
        if os.path.exists(csv_dir):
            print(f"   файлов: {len(csv_dir)}")
        else:
            print("   Директория не существует")

        # Показываем результаты
        print(f"\n Результаты ({results_dir}):")
        if os.path.exists(results_dir):
            result_files = os.listdir(results_dir)
            print(f"   Файлов результатов: {len(result_files)}")
        else:
            print("   Директория будет создана при сохранении")

        print("-" * 50)

    # Определяем режим работы
    IN_DRIVE_MODE = drive_path is not None

    print("""
╔══════════════════════════════════════════╗
║      АНАЛИТИК ЗВОНКОВ v3.1               ║
║      Google Drive Edition                ║
╚══════════════════════════════════════════╝
    """)

    if IN_DRIVE_MODE:
        print(f"Режим: Google Drive")
        print(f"Основной путь: {drive_path}")
        print(f"Данные: {csv_dir}")
        print(f"Сохранение: {results_dir}")
        print("-" * 50)

    # Инициализация системы
    JSON_DIRECTORY = csv_dir if csv_dir else "csv_calls"
    RESULTS_DIRECTORY = results_dir if results_dir else "saved_results"

    # Проверяем директории
    if not os.path.exists(JSON_DIRECTORY):
        print(f"❌ Директория {JSON_DIRECTORY} не найдена!")

        if IN_DRIVE_MODE:
            print(f"\n Для загрузки файлов в Google Drive:")
            print(f"1. Откройте {drive_path} в браузере")
            print(f"2. Создайте папку 'csv_calls'")
            print(f"3. Загрузите туда файл")
            print(f"4. Перезапустите систему")
        else:
            print("Сначала добавьте csv файл в директорию csv_calls/")

        return

    # Создаем директорию для результатов
    os.makedirs(RESULTS_DIRECTORY, exist_ok=True)

    system = phonecall.colab.mcp_orchestrator.JSONCallAnalyticsMCP(JSON_DIRECTORY, _model, node_url, drive_path)

    # История запросов
    query_history = []

    # Основной цикл
    while True:
        try:
            # Отображаем приглашение с информацией о режиме
            mode_indicator = "[Drive] " if IN_DRIVE_MODE else "[Local] "
            prompt = f"\n{mode_indicator}📝 Вопрос (/? для помощи): "
            user_input = input(prompt).strip()

            # Обработка команд
            if user_input.lower() in ['/выход', '/exit', 'выход', 'exit', '/q', 'q']:
                print(" До свидания!")
                if IN_DRIVE_MODE:
                    print(" Все данные сохранены.")
                break

            elif user_input.lower() in ['/?', '/помощь', '/help']:
                show_help(IN_DRIVE_MODE)
                continue

            elif user_input.lower() == '/статистика':
                show_system_stats(system, IN_DRIVE_MODE)
                continue

            elif user_input.lower() == '/история':
                show_query_history(query_history)
                continue

            elif user_input.lower() == '/очистить':
                os.system('cls' if os.name == 'nt' else 'clear')
                print("🧹 Экран очищен")
                continue

            elif user_input.lower().startswith('/сохранить'):
                save_last_result(query_history, RESULTS_DIRECTORY)
                continue

            elif user_input.lower() == '/тест':
                test_system(system)
                continue

            elif user_input.lower() == '/директории':
                show_directories(JSON_DIRECTORY, RESULTS_DIRECTORY, drive_path)
                continue

            elif user_input.lower() == '/режим':
                print(f"Текущий режим: {'Google Drive' if IN_DRIVE_MODE else 'Локальный'}")
                continue

            elif not user_input:
                continue

            # Обработка аналитического запроса
            print(f"Анализирую: '{user_input}'")

            if IN_DRIVE_MODE:
                print("Чтение данных из Google Drive...")

            # Сохраняем в историю перед обработкой
            query_history.append({
                'query': user_input,
                'timestamp': datetime.now(),
                'status': 'processing',
                'mode': 'drive' if IN_DRIVE_MODE else 'local'
            })

            # Ограничиваем историю
            if len(query_history) > 20:
                query_history = query_history[-20:]

            # Обрабатываем запрос
            start_time = datetime.now()
            result = system.process_query(user_input, query_history)
            processing_time = (datetime.now() - start_time).total_seconds()

            # Обновляем историю
            query_history[-1]['status'] = 'completed'
            query_history[-1]['result'] = result
            query_history[-1]['processing_time'] = processing_time

            # Показываем результат
            print("\n" + "=" * 60)
            print(f"ОТВЕТ ({processing_time:.1f} сек):")
            print("-" * 40)
            print(result['answer'])
            print("-" * 40)

            # Показываем информацию о данных
            print(f"Проанализировано звонков: {result.get('total_calls_analyzed', 0)}")

            if IN_DRIVE_MODE:
                print(f"Результаты сохраняются в Google Drive")

            # Быстрые действия
            print("\n⚡ Быстрые действия:")
            print("  • Задать уточняющий вопрос")
            print("  • /сохранить - сохранить этот результат")
            print("  • /история - показать предыдущие запросы")
            print("  • /директории - показать структуру данных")
            print("  • /выход - завершить работу")

        except KeyboardInterrupt:
            print("\n\n👋 Завершаю работу...")
            if IN_DRIVE_MODE:
                print("Данные сохранены в Google Drive")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
            if query_history:
                query_history[-1]['status'] = 'error'
                query_history[-1]['error'] = str(e)

