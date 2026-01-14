import json
import os
from datetime import datetime

from mcp_orchestrator import JSONCallAnalyticsMCP


def enhanced_interactive_mode(_model: str):
    """Расширенный интерактивный режим с командами и историей"""


    print("""
╔═══════════════════════════════════════╗
║    🤖 АНАЛИТИК ЗВОНКОВ v2.0            ║
╚═══════════════════════════════════════╝
    """)

    # Инициализация
    JSON_DIRECTORY = "json_calls"
    import os
    if not os.path.exists(JSON_DIRECTORY):
        print(f"❌ Директория {JSON_DIRECTORY} не найдена!")
        print("Сначала добавьте JSON файлы в директорию")
        return

    system = JSONCallAnalyticsMCP(JSON_DIRECTORY, _model)

    # История запросов
    query_history = []

    # Основной цикл
    while True:
        try:
            # Отображаем приглашение
            prompt = "\n📝 Вопрос (/? для помощи): "
            user_input = input(prompt).strip()

            # Обработка команд
            if user_input.lower() in ['/выход', '/exit', 'выход', 'exit']:
                print("👋 До свидания!")
                break

            elif user_input.lower() in ['/?', '/помощь', '/help']:
                show_help()
                continue

            elif user_input.lower() == '/статистика':
                show_system_stats(system)
                continue

            elif user_input.lower() == '/история':
                show_query_history(query_history)
                continue

            elif user_input.lower() == '/очистить':
                import os
                os.system('cls' if os.name == 'nt' else 'clear')
                print("🧹 Экран очищен")
                continue

            elif user_input.lower().startswith('/сохранить'):
                save_last_result(query_history)
                continue

            elif user_input.lower() == '/тест':
                test_system(system)
                continue

            elif not user_input:
                continue

            # Обработка аналитического запроса
            print(f"🔍 Анализирую: '{user_input}'")

            # Сохраняем в историю перед обработкой
            query_history.append({
                'query': user_input,
                'timestamp': datetime.now(),
                'status': 'processing'
            })

            # Ограничиваем историю
            if len(query_history) > 20:
                query_history = query_history[-20:]

            # Обрабатываем запрос
            start_time = datetime.now()
            result = system.process_query(user_input)
            processing_time = (datetime.now() - start_time).total_seconds()

            # Обновляем историю
            query_history[-1]['status'] = 'completed'
            query_history[-1]['result'] = result
            query_history[-1]['processing_time'] = processing_time

            # Показываем результат
            print("\n" + "=" * 60)
            print(f"💡 ОТВЕТ ({processing_time:.1f} сек):")
            print("-" * 40)
            print(result['answer'])
            print("-" * 40)

            # Быстрые действия
            print("\n⚡ Быстрые действия:")
            print("  • Задать уточняющий вопрос")
            print("  • /сохранить - сохранить этот результат")
            print("  • /история - показать предыдущие запросы")
            print("  • /выход - завершить работу")

        except KeyboardInterrupt:
            print("\n\n👋 Завершаю работу...")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
            if query_history:
                query_history[-1]['status'] = 'error'
                query_history[-1]['error'] = str(e)


def show_help():
    """Показывает справку по командам"""
    help_text = """
📖 КОМАНДЫ:

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

ПРИМЕРЫ ЗАПРОСОВ:
  • "Динамика жалоб за последние 3 месяца"
  • "Топ-5 проблем клиентов в ноябре"
  • "Сравни январь и февраль по всем метрикам"
  • "Какие рекомендации по улучшению сервиса?"
    """
    print(help_text)


def show_system_stats(system):
    """Показывает статистику системы"""
    info = system.get_system_info()

    print("\n📊 СТАТИСТИКА СИСТЕМЫ:")
    print("-" * 40)
    print(f"📞 Всего звонков: {info['total_calls']}")
    print(f"🏷️  Уникальных тегов: {info['unique_tags_count']}")

    if info['date_range']['start']:
        start_date = datetime.fromisoformat(info['date_range']['start']).strftime('%d.%m.%Y')
        end_date = datetime.fromisoformat(info['date_range']['end']).strftime('%d.%m.%Y')
        print(f"📅 Период данных: {start_date} - {end_date}")

    print(f"📝 Средняя длина текста: {info['average_text_length']} симв.")
    print(f"🤖 Модель: {info['model']}")
    print(f"📁 Источник: {info['data_source']}")


def show_query_history(history):
    """Показывает историю запросов"""
    if not history:
        print("📭 История запросов пуста")
        return

    print("\n🕐 ИСТОРИЯ ЗАПРОСОВ:")
    print("-" * 60)

    # Показываем последние 10 запросов
    for i, item in enumerate(reversed(history[-10:]), 1):
        time_str = item['timestamp'].strftime('%H:%M')
        status_icon = "✅" if item.get('status') == 'completed' else "❌" if item.get('status') == 'error' else "⏳"

        # Обрезаем длинные запросы
        query_preview = item['query']
        if len(query_preview) > 50:
            query_preview = query_preview[:47] + "..."

        print(f"{i}. [{time_str}] {status_icon} {query_preview}")

        # Показываем время обработки если есть
        if item.get('processing_time'):
            print(f"   ⏱️  {item['processing_time']:.1f} сек")

    print("-" * 60)
    print(f"Всего запросов: {len(history)}")


def save_last_result(history):
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
    safe_query = last_result['query'][:30].replace(' ', '_').replace('?', '')
    filename = f"saved_results/result_{timestamp}_{safe_query}.json"

    # Создаем директорию если нет
    os.makedirs('saved_results', exist_ok=True)

    # Сохраняем
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(last_result, f, ensure_ascii=False, indent=2)

    print(f"✅ Результат сохранен в: {filename}")


def test_system(system):
    """Тестирует систему"""
    print("🧪 Тестирую систему...")

    test_queries = [
        "Сколько всего звонков в базе?",
        "Какие теги есть в данных?",
        "Тестовый запрос: жалобы"
    ]

    for query in test_queries:
        print(f"\n📋 Тест: '{query}'")
        try:
            result = system.process_query(query)
            print(f"   ✅ Успешно, ответ: {len(result['answer'])} симв.")
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")