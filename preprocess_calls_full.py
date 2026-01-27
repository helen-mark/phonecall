import os
import json
import librosa
import numpy as np
import pandas as pd
import whisper
from datetime import datetime
from pydub import AudioSegment
import warnings

warnings.filterwarnings('ignore')


class AudioProcessor:
    def __init__(self, model_size):
        """
        Инициализация процессора аудио
        model_size: "tiny", "base", "small", "medium", "large"
        """
        print("Загрузка модели Whisper...")
        self.asr_model = whisper.load_model(model_size)
        print("Модель загружена!")

    def extract_date_from_filename(self, filename):
        """
        Извлечение даты из имени файла
        Предполагается формат: YYYY-MM-DD_* или подобный
        """
        try:
            # Пытаемся найти дату в формате YYYY-MM-DD
            for part in filename.split('_'):
                if len(part) == 10 and part[4] == '-' and part[7] == '-':
                    try:
                        return datetime.strptime(part, '%Y-%m-%d').strftime('%Y-%m-%d')
                    except:
                        continue
            # Если дата не найдена, используем дату изменения файла
            return datetime.now().strftime('%Y-%m-%d')
        except:
            return datetime.now().strftime('%Y-%m-%d')

    def convert_to_16k(self, audio_path, output_path=None):
        """
        Конвертация аудио в 16 кГц моно WAV
        """
        if output_path is None:
            base_name = os.path.splitext(audio_path)[0]
            output_path = f"{base_name}_16k.wav"

        # Проверяем текущую частоту
        y, sr = librosa.load(audio_path, sr=None)

        if sr == 16000:
            print(f"✅ Файл уже имеет частоту 16 кГц: {audio_path}")
            return audio_path
        else:
            print(f"🔄 Конвертируем {audio_path} из {sr} Hz в 16000 Hz...")

            # Конвертируем
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(output_path, format="wav")

            return output_path

    def assess_quality(self, audio_path):
        """
        Оценка качества аудио (упрощенная версия)
        Возвращает оценку от 1 до 10
        """
        try:
            y, sr = librosa.load(audio_path, sr=16000)

            # 1. Проверка громкости
            rms = librosa.feature.rms(y=y)
            rms_mean = np.mean(rms)

            # 2. Проверка на шум (через zero-crossing rate)
            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_mean = np.mean(zcr)

            # 3. Проверка на тишину (паузы)
            frame_length = 2048
            hop_length = 512
            rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            silence_threshold = np.mean(rms_frames) * 0.1
            silent_ratio = np.sum(rms_frames < silence_threshold) / len(rms_frames)

            # 4. Расчет оценки
            score = 5  # Базовая оценка

            # Корректировка на основе громкости
            if rms_mean > 0.05:
                score += 2
            elif rms_mean > 0.02:
                score += 1
            elif rms_mean < 0.005:
                score -= 2

            # Корректировка на основе шума
            if zcr_mean < 0.1:
                score += 1  # Низкий ZCR = меньше шума
            elif zcr_mean > 0.3:
                score -= 1  # Высокий ZCR = больше шума

            # Корректировка на основе пауз
            if 0.1 < silent_ratio < 0.4:
                score += 1  # Нормальное количество пауз
            elif silent_ratio > 0.7:
                score -= 2  # Слишком много тишины

            # Ограничение оценки от 1 до 10
            score = max(1, min(10, int(score)))

            return score

        except Exception as e:
            print(f"Ошибка при оценке качества: {e}")
            return 5  # Средняя оценка при ошибке

    def transcribe_audio(self, audio_path):
        """
        Транскрибация аудио в текст
        """
        try:
            result = self.asr_model.transcribe(audio_path)
            return result['text'].strip()
        except Exception as e:
            print(f"Ошибка при транскрибации: {e}")
            return ""

    def process_file(self, audio_path, quality_threshold, transcribe_all):
        """
        Полная обработка одного аудиофайла
        transcribe_all: если True - транскрибировать все файлы,
                       если False - транскрибировать только качественные
        """
        print(f"\n{'=' * 60}")
        print(f"Обработка файла: {audio_path}")
        print(f"{'=' * 60}")

        # Извлекаем дату из имени файла
        date = self.extract_date_from_filename(os.path.basename(audio_path))

        # Конвертируем в 16 кГц
        converted_path = self.convert_to_16k(audio_path)

        # Оцениваем качество
        quality_score = self.assess_quality(converted_path)
        print(f"Оценка качества: {quality_score}/10")

        # Определяем, нужно ли транскрибировать
        text = ""
        should_transcribe = transcribe_all or quality_score >= quality_threshold

        if should_transcribe:
            # Транскрибируем
            print("Транскрибация...")
            text = self.transcribe_audio(converted_path)
        else:
            print("⏭️  Пропускаем транскрибацию (низкое качество)")

        # Возвращаем результат для ВСЕХ файлов
        return {
            'date': date,
            'text': text,
            'source_file': os.path.basename(audio_path),
            'quality_score': quality_score,
            'tags': ''  # Пустая колонка для тегов
        }

    def process_directory(self, input_dir, output_csv, quality_threshold, transcribe_all):
        """
        Обработка всех аудиофайлов в директории
        """
        # Поддерживаемые форматы аудио
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']

        # Проверяем, существует ли уже CSV файл
        existing_files = set()
        if os.path.exists(output_csv):
            print(f"📁 Найден существующий файл: {output_csv}")
            try:
                existing_df = pd.read_csv(output_csv)
                existing_files = set(existing_df['source_file'].tolist())
                print(f"📊 В файле уже содержится {len(existing_files)} записей")
            except Exception as e:
                print(f"⚠️ Ошибка при чтении существующего файла: {e}")

        # Находим все аудиофайлы
        audio_files = []
        for file in os.listdir(input_dir):
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                file_path = os.path.join(input_dir, file)
                # Пропускаем уже обработанные файлы
                if file not in existing_files:
                    audio_files.append(file_path)
                else:
                    print(f"⏭️  Файл уже обработан: {file}")

        if not audio_files:
            print("ℹ️  Все файлы уже обработаны ранее")
            if os.path.exists(output_csv):
                return pd.read_csv(output_csv).to_dict('records')
            return []

        print(f"Найдено {len(audio_files)} новых аудиофайлов для обработки")

        # Обрабатываем файлы
        results = []
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n📁 Обработка файла {i}/{len(audio_files)}")
            try:
                result = self.process_file(audio_file, quality_threshold, transcribe_all)
                results.append(result)  # Теперь добавляем ВСЕ результаты
                print(f"✅ Добавлен в список: {os.path.basename(audio_file)}")
            except Exception as e:
                print(f"❌ Ошибка при обработке {audio_file}: {e}")

        # Объединяем старые и новые данные
        if results:
            new_df = pd.DataFrame(results)

            if os.path.exists(output_csv):
                # Загружаем существующие данные
                existing_df = pd.read_csv(output_csv)
                # Объединяем
                final_df = pd.concat([existing_df, new_df], ignore_index=True)
                # Убираем возможные дубликаты (на всякий случай)
                final_df = final_df.drop_duplicates(subset=['source_file'], keep='first')
            else:
                final_df = new_df

            # Сортируем по дате
            final_df = final_df.sort_values('date')

            # Сохраняем в CSV
            final_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            print(f"\n{'=' * 60}")
            print(f"✅ Обработка завершена!")
            print(f"📊 Всего записей в файле: {len(final_df)}")
            print(f"🆕 Добавлено новых записей: {len(results)}")
            print(f"💾 Результаты сохранены в: {output_csv}")
            print(f"{'=' * 60}")

            # Показываем краткую статистику
            print("\n📈 Статистика:")
            print(f"Средняя оценка качества: {final_df['quality_score'].mean():.1f}")
            print(f"Общее количество символов текста: {final_df['text'].str.len().sum()}")
            print(f"Количество файлов с оценкой >=8: {(final_df['quality_score'] >= 8).sum()}")

            return final_df.to_dict('records')
        else:
            print("❌ Не найдено новых файлов, соответствующих критериям качества")
            if os.path.exists(output_csv):
                return pd.read_csv(output_csv).to_dict('records')
            return []


def run():
    """
    Основная функция
    """
    # Настройки
    INPUT_DIR = "audio_pool"  # Директория с исходными аудиофайлами
    OUTPUT_CSV = "calls.csv"  # Выходной CSV файл
    QUALITY_THRESHOLD = 7  # Минимальная оценка качества
    MODEL_SIZE = "large"  # Модель Whisper: "tiny", "base", "small", "medium", "large"
    TRANSCRIBE_ALL = False  # True = транскрибировать все файлы, False = только качественные

    # Проверяем существование директории
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Директория '{INPUT_DIR}' не существует!")
        return

    if os.path.exists(OUTPUT_CSV):
        print(f"📁 Обнаружен существующий файл: {OUTPUT_CSV}")
        print(f"ℹ️  Будет выполнена дозапись новых файлов")

    # Создаем процессор
    processor = AudioProcessor(model_size=MODEL_SIZE)

    # Обрабатываем директорию
    processor.process_directory(
        input_dir=INPUT_DIR,
        output_csv=OUTPUT_CSV,
        quality_threshold=QUALITY_THRESHOLD,
        transcribe_all=TRANSCRIBE_ALL
    )


if __name__ == "__main__":
    run()