"""
Минимальный скрипт для запуска в Colab
Сохраните как launch.py и запустите в Colab: !python launch.py
"""

import os
import subprocess
import time
import sys


def setup_colab():
    """Настройка Colab"""
    print("🚀 Настройка MCP системы в Colab")
    print("-" * 50)

    # 1. Устанавливаем Ollama
    print("1️⃣ Устанавливаю Ollama...")
    subprocess.run(["curl", "-fsSL", "https://ollama.com/install.sh"], shell=True)
    subprocess.run(["sh", "-c", "$(curl -fsSL https://ollama.com/install.sh)"], shell=True)

    # 2. Запускаем Ollama
    print("2️⃣ Запускаю Ollama сервер...")
    subprocess.Popen(["ollama", "serve"])
    time.sleep(10)

    # 3. Скачиваем модель
    print("3️⃣ Скачиваю модель mistral-nemo:12b...")
    subprocess.run(["ollama", "pull", "mistral-nemo:12b"], check=True)

    # 4. Монтируем Google Drive
    print("4️⃣ Монтирую Google Drive...")
    from google.colab import drive
    drive.mount('/content/drive')

    return True


def main():
    """Основная функция"""
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False
        print("⚠️  Не в Colab, запускаю локально...")

    if IN_COLAB:
        setup_colab()

        # 🚨 УКАЖИТЕ СВОЙ ПУТЬ К ДАННЫМ
        DRIVE_PATH = "/content/drive/MyDrive/ваша_папка"  # ИЗМЕНИТЕ!
        JSON_DIR = os.path.join(DRIVE_PATH, "json_calls")

        if not os.path.exists(JSON_DIR):
            print(f"❌ Папка не найдена: {JSON_DIR}")
            print(f"Создайте структуру в Google Drive или укажите правильный путь")
            return
    else:
        # Локальный режим
        DRIVE_PATH = None
        JSON_DIR = "./json_calls"  # Локальная папка

    # Проверяем файлы кода
    required_files = ["main.py", "interactive.py", "mcp_orchestrator.py"]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print(f"❌ Отсутствуют файлы: {missing_files}")
        print("Загрузите файлы в Colab через меню файлов или командой:")
        print("from google.colab import files")
        print("files.upload()")
        return

    # Запускаем систему
    print("\n" + "=" * 50)
    print("🤖 ЗАПУСКАЮ СИСТЕМУ АНАЛИЗА ЗВОНКОВ")
    print("=" * 50)

    from interactive import enhanced_interactive_mode

    # Запуск с вашими параметрами
    RESULTS_DIR = os.path.join(DRIVE_PATH, "saved_results") if DRIVE_PATH else "./saved_results"

    enhanced_interactive_mode(
        _model="mistral-nemo:12b",
        json_dir=JSON_DIR,
        results_dir=RESULTS_DIR,
        drive_path=DRIVE_PATH
    )


if __name__ == "__main__":
    main()


# 1. Установка Ollama
!sudo
apt - get
install
zstd
# 1. Установите ollama Python пакет
!pip
install
ollama

# 2. Проверьте установку
!pip
list | grep
ollama

# 3. Убедитесь, что сервер Ollama запущен
!ollama
list  # Проверка сервера

# 4. Если сервер не запущен, запустите его
import subprocess, time

print("Запускаю Ollama сервер...")
subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(10)

# 5. Проверьте доступность сервера
import requests

try:
    response = requests.get("http://localhost:11434/api/tags", timeout=5)
    if response.status_code == 200:
        print("✅ Ollama сервер работает")
    else:
        print(f"⚠️  Ollama сервер отвечает с кодом {response.status_code}")
except:
    print("❌ Ollama сервер не отвечает")

# 6. Скачайте модель если еще не скачана
print("Скачиваю модель mistral-nemo:12b...")
!ollama
pull
mistral - nemo: 12
b

# 4. Создаем структуру
import os
os.makedirs("/content/json_calls", exist_ok=True)
os.makedirs("/content/saved_results", exist_ok=True)

# 5. Загружаем ваши файлы системы
from google.colab import files

print("📤 ЗАГРУЗИТЕ ФАЙЛЫ СИСТЕМЫ:")
print("1. main.py")
print("2. interactive.py")
print("3. mcp_orchestrator.py")
uploaded = files.upload()

# 6. Загружаем JSON данные
print("📤 ЗАГРУЗИТЕ JSON ФАЙЛЫ ЗВОНКОВ:")
json_files = files.upload()

# Перемещаем JSON файлы
for filename in json_files.keys():
    if filename.endswith('.json'):
        os.rename(filename, f"/content/json_calls/{filename}")
        print(f"✅ JSON файл: {filename}")

import torch
print("🎮 Проверка GPU в Colam:")
print(f"CUDA доступен: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Устройство: {torch.cuda.get_device_name(0)}")
    print(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("❌ GPU не доступен в PyTorch")

# Проверка через Ollama
print("\n🔍 Проверка Ollama:")
!ollama list
!ollama ps

# 7. Запускаем систему
import sys
sys.path.append('.')

from interactive import enhanced_interactive_mode

enhanced_interactive_mode(
    _model="mistral-nemo:12b",
    json_dir="/content/json_calls",
    results_dir="/content/saved_results",
    drive_path=None
)