import argparse
import os
import sys
from pathlib import Path


def setup_google_drive():
    """Настройка Google Drive для Colab"""
    try:
        from google.colab import drive
        # Монтируем Google Drive
        drive.mount('/content/drive')
        print("✅ Google Drive смонтирован")
        return True
    except ImportError:
        print("⚠️  Не в Google Colab, пропускаю монтирование Drive")
        return False


def get_drive_path(base_path="call_analytics"):
    """Получает путь в Google Drive"""
    # Стандартный путь в Drive для Colab
    drive_root = "/content/drive/MyDrive"

    # Создаем полный путь
    full_path = os.path.join(drive_root, base_path)

    # Создаем директорию если нет
    os.makedirs(full_path, exist_ok=True)

    # Создаем поддиректории
    directories = ['json_calls', 'saved_results', 'logs', 'models_cache']
    for dir_name in directories:
        os.makedirs(os.path.join(full_path, dir_name), exist_ok=True)

    return full_path


def check_drive_contents(drive_path):
    """Проверяет содержимое Google Drive"""
    print(f"\n📁 Содержимое Google Drive ({drive_path}):")

    # Проверяем основную директорию
    if os.path.exists(drive_path):
        for item in os.listdir(drive_path):
            item_path = os.path.join(drive_path, item)
            if os.path.isdir(item_path):
                file_count = len([f for f in os.listdir(item_path) if f.endswith('.json')])
                print(f"  📂 {item}/ - {file_count} JSON файлов")
            else:
                print(f"  📄 {item}")
    else:
        print("  ℹ️  Директория не существует, будет создана")

    print("-" * 50)


if __name__ == "__main__":
    # Настраиваем Google Drive если в Colab
    IN_COLAB = setup_google_drive()

    if IN_COLAB:
        # Используем Google Drive как основное хранилище
        DRIVE_BASE = "MCP_Call_Analytics"  # Название папки в вашем Drive
        drive_path = get_drive_path(DRIVE_BASE)

        # Показываем структуру
        check_drive_contents(drive_path)

        # Пути к данным в Drive
        json_dir = os.path.join(drive_path, "json_calls")
        results_dir = os.path.join(drive_path, "saved_results")

        print(f"\n📍 Использую пути:")
        print(f"   📁 JSON данные: {json_dir}")
        print(f"   💾 Результаты: {results_dir}")
        print(f"   🔧 Логи: {os.path.join(drive_path, 'logs')}")

        # Автоматический режим для Colab
        args = type('Args', (), {
            'mode': 'interactive',
            'json_dir': json_dir,
            'results_dir': results_dir,
            'model': 'mistral-nemo:12b',
            'telegram_token': None,
            'drive_mode': True,
            'drive_path': drive_path
        })()

        # Импортируем после настройки путей
        from interactive import enhanced_interactive_mode

        enhanced_interactive_mode(args.model, args.json_dir, args.results_dir, args.drive_path)

    else:
        # Локальный режим (без Colab)
        parser = argparse.ArgumentParser(description='MCP система анализа телефонных звонков')
        parser.add_argument('--mode', default='interactive',
                            choices=['interactive', 'web', 'telegram', 'api', 'test'],
                            help='Режим работы')
        parser.add_argument('--json-dir', default='./json_calls',
                            help='Путь к JSON файлам')
        parser.add_argument('--results-dir', default='./saved_results',
                            help='Путь для сохранения результатов')
        parser.add_argument('--model', default='mistral-nemo:12b',
                            help='Модель Ollama')
        parser.add_argument('--telegram-token',
                            help='Токен Telegram бота (для режима telegram)')
        parser.add_argument('--drive-path', default=None,
                            help='Путь к Google Drive (только для Colab)')

        args = parser.parse_args()

        if args.mode == 'interactive':
            from interactive import enhanced_interactive_mode

            enhanced_interactive_mode(args.model, args.json_dir, args.results_dir, args.drive_path)
        elif args.mode == 'test':
            from mcp_orchestrator import JSONCallAnalyticsMCP

            system = JSONCallAnalyticsMCP(args.json_dir, args.model, args.drive_path)
            system.test_system()