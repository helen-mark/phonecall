"""
Настройка системы для Google Colab с Google Drive
"""

import os
import subprocess
import time
import requests
from pathlib import Path


class ColabDriveSetup:
    """Класс для настройки Colab с Google Drive"""

    @staticmethod
    def setup_environment():
        """Полная настройка окружения Colab"""

        print("=" * 60)
        print("🚀 НАСТРОЙКА MCP АНАЛИТИКИ ЗВОНКОВ ДЛЯ COLAB")
        print("🌐 Google Drive Integration")
        print("=" * 60)

        # Шаг 0: Проверяем, что мы в Colab
        IN_COLAB = ColabDriveSetup._check_colab()
        if not IN_COLAB:
            print("⚠️  Не в Google Colab, пропускаю настройку")
            return None, None

        # Шаг 1: Монтируем Google Drive
        drive_path = ColabDriveSetup._mount_google_drive()
        if not drive_path:
            return None, None

        # Шаг 2: Устанавливаем Ollama
        ColabDriveSetup._install_ollama()

        # Шаг 3: Настраиваем модель
        model_name = ColabDriveSetup._setup_model()

        # Шаг 4: Создаем структуру в Drive
        base_path = ColabDriveSetup._create_drive_structure(drive_path)

        # Шаг 5: Устанавливаем Python зависимости
        ColabDriveSetup._install_dependencies()

        print("\n" + "=" * 60)
        print("🎉 НАСТРОЙКА ЗАВЕРШЕНА!")
        print("=" * 60)

        return base_path, model_name

    @staticmethod
    def _check_colab():
        """Проверяем, запущено ли в Colab"""
        try:
            import google.colab
            print("✅ Обнаружен Google Colab")
            return True
        except:
            return False

    @staticmethod
    def _mount_google_drive():
        """Монтирует Google Drive"""
        try:
            from google.colab import drive
            print("\n📁 Шаг 1: Монтирую Google Drive...")
            drive.mount('/content/drive')

            drive_root = "/content/drive/MyDrive"
            print(f"✅ Google Drive смонтирован: {drive_root}")
            return drive_root
        except Exception as e:
            print(f"❌ Ошибка при монтировании Google Drive: {e}")
            return None

    @staticmethod
    def _install_ollama():
        """Устанавливает Ollama"""
        print("\n📦 Шаг 2: Устанавливаю Ollama...")

        # Устанавливаем Ollama
        !curl - fsSL
        https: // ollama.com / install.sh | sh

        # Запускаем в фоне
        print("⚙️  Запускаю Ollama сервер...")
        !ollama
        serve > / dev / null
        2 > & 1 &

        # Ждем запуска
        time.sleep(8)

        # Проверяем запуск
        for i in range(5):
            try:
                response = requests.get('http://localhost:11434/api/tags', timeout=5)
                if response.status_code == 200:
                    print("✅ Ollama сервер запущен успешно")
                    return True
            except:
                print(f"⏳ Попытка {i + 1}/5...")
                time.sleep(3)

        print("⚠️  Ollama сервер может быть не готов, продолжаем...")
        return False

    @staticmethod
    def _setup_model():
        """Настраивает модель"""
        print("\n🤖 Шаг 3: Настраиваю модель...")

        # Выбор модели в зависимости от доступной памяти
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory

                if gpu_memory >= 40e9:  # 40GB+ (A100)
                    model_name = "qwen2.5:14b"
                    print(f"🎮 Обнаружен мощный GPU (A100), использую модель: {model_name}")
                elif gpu_memory >= 16e9:  # 16GB+ (V100/T4)
                    model_name = "mistral-nemo:12b"
                    print(f"🎮 Обнаружен хороший GPU (V100/T4), использую модель: {model_name}")
                else:
                    model_name = "mistral:7b"
                    print(f"🎮 Обнаружен GPU с ограниченной памятью, использую модель: {model_name}")
            else:
                model_name = "mistral:7b"
                print("⚠️  GPU не обнаружен, использую легкую модель: mistral:7b")
        except:
            model_name = "mistral:7b"
            print("ℹ️  Использую модель по умолчанию: mistral:7b")

        # Скачиваем модель
        print(f"📥 Скачиваю модель {model_name}...")
        !ollama
        pull
        {model_name}

        return model_name

    @staticmethod
    def _create_drive_structure(drive_root):
        """Создает структуру директорий в Google Drive"""
        print("\n📁 Шаг 4: Создаю структуру в Google Drive...")

        base_folder = "MCP_Call_Analytics"
        base_path = os.path.join(drive_root, base_folder)

        # Создаем основную директорию
        os.makedirs(base_path, exist_ok=True)

        # Создаем поддиректории
        directories = {
            'json_calls': '📊 JSON файлы телефонных звонков',
            'saved_results': '💾 Сохраненные результаты анализа',
            'logs': '📝 Логи работы системы',
            'models_cache': '🤖 Кэш моделей Ollama'
        }

        for dir_name, description in directories.items():
            dir_path = os.path.join(base_path, dir_name)
            os.makedirs(dir_path, exist_ok=True)

            # Создаем README файл
            readme_path = os.path.join(dir_path, "README.txt")
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(f"{description}\n")
                f.write(f"Создано: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

            print(f"  ✓ {dir_name}/ - {description}")

        print(f"✅ Структура создана: {base_path}")
        return base_path

    @staticmethod
    def _install_dependencies():
        """Устанавливает Python зависимости"""
        print("\n🐍 Шаг 5: Устанавливаю зависимости...")

        dependencies = [
            'ollama-python',
            'python-dotenv',
            'requests',
            'ipywidgets',
            'ipython'
        ]

        for dep in dependencies:
            !pip
            install - q
            {dep}

        print("✅ Зависимости установлены")

    @staticmethod
    def quick_start():
        """Быстрый старт системы"""
        print("🚀 Быстрый старт MCP системы с Google Drive")
        print("-" * 50)

        # Настраиваем окружение
        base_path, model_name = ColabDriveSetup.setup_environment()

        if not base_path:
            print("❌ Настройка не удалась")
            return

        # Импортируем и запускаем систему
        print("\n▶️  Запускаю систему...")

        # Добавляем текущую директорию в путь
        import sys
        sys.path.append('.')

        # Импортируем
        from main import setup_google_drive, get_drive_path
        from interactive import enhanced_interactive_mode

        # Запускаем интерактивный режим
        json_dir = os.path.join(base_path, "json_calls")
        results_dir = os.path.join(base_path, "saved_results")

        print(f"\n📍 Пути:")
        print(f"   📊 Данные: {json_dir}")
        print(f"   💾 Результаты: {results_dir}")
        print(f"   🤖 Модель: {model_name}")
        print("-" * 50)

        # Проверяем наличие данных
        if not os.path.exists(json_dir) or len([f for f in os.listdir(json_dir) if f.endswith('.json')]) == 0:
            print("\n⚠️  Внимание: Нет JSON файлов в директории данных!")
            print(f"📤 Загрузите JSON файлы в: {json_dir}")
            print("\nМожно использовать команду:")
            print("from google.colab import files")
            print("uploaded = files.upload()")
            print("for filename in uploaded.keys():")
            print(f"    os.rename(filename, '{json_dir}/' + filename)")

        # Запускаем
        enhanced_interactive_mode(model_name, json_dir, results_dir, base_path)


if __name__ == "__main__":
    # Быстрый запуск
    ColabDriveSetup.quick_start()