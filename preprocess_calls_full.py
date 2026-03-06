import os
import json
import librosa
import numpy as np
import pandas as pd
import whisper
import soundfile as sf
from datetime import datetime
from pydub import AudioSegment
import warnings
import ffmpeg

warnings.filterwarnings('ignore')


class AudioProcessor:
    _loaded_models = {}
    
    def __init__(self, model_size):
        self.model_size = model_size
        
        if model_size not in self._loaded_models:
            print(f"Загрузка модели Whisper {model_size}...")
            self._loaded_models[model_size] = whisper.load_model(model_size)
            print(f" Модель {model_size} загружена!")
        else:
            print(f" Используем уже загруженную модель {model_size}")
        
        self.asr_model = self._loaded_models[model_size]

    def extract_date_from_filename(self, filename):
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

        if output_path is None:
            base_name = os.path.splitext(audio_path)[0]
            output_path = f"{base_name}_16k.wav"

        # Загружаем аудио
        y, sr = librosa.load(audio_path, sr=None)

        if sr == 16000:
            print(f" Файл уже имеет частоту 16 кГц: {audio_path}")
            return audio_path
        else:
            print(f" Конвертируем из {sr} Hz в 16000 Hz...")

            # Ресемплируем до 16 кГц
            y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)

            # Сохраняем как WAV файл
            sf.write(output_path, y_16k, 16000, subtype='PCM_16')

            return output_path

    def convert_to_16k_ffprobe(self, audio_path, output_path=None):
        """
        Конвертация аудио в 16 кГц моно WAV
        """
        if output_path is None:
            base_name = os.path.splitext(audio_path)[0]
            output_path = f"{base_name}_16k.wav"

        # Проверяем текущую частоту
        y, sr = librosa.load(audio_path, sr=None)

        if sr == 16000:
            print(f" Файл уже имеет частоту 16 кГц: {audio_path}")
            return audio_path
        else:
            print(f" Конвертируем из {sr} Hz в 16000 Hz...")

            # Конвертируем
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(output_path, format="wav")

            return output_path

    def assess_quality(self, audio_path):
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
        print('Converted')

        print('Quality estimation...')
        quality_score = self.assess_quality(converted_path)
        print(f"Estimated quality: {quality_score}/10")

        # Определяем, нужно ли транскрибировать
        text = '-'
        should_transcribe = transcribe_all or quality_score >= quality_threshold

        if should_transcribe:
            # Транскрибируем
            print("Транскрибация...")
            text = self.transcribe_audio(converted_path)
            if text == '':
                text = '-'
        else:
            print(" Пропускаем транскрибацию (низкое качество)")
            

        # Возвращаем результат для ВСЕХ файлов
        return {
            'date': date,
            'text': text,
            'text_length': len(text),
            'source_audio': os.path.basename(audio_path),
            'quality_score': quality_score,
            'tags': '',
            'summary': '',
            'processing_date': datetime.now().isoformat(),
            'batch_number': 0,
            'whisper_model': self.model_size,
            'audio_duration': 0
        }

    def process_directory(self, input_dir, output_csv, quality_threshold, transcribe_all):
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
        batch_size = 1

        existing_files = set()
        if os.path.exists(output_csv):
            print(f"Found existing file: {output_csv}")
            try:
                existing_df = pd.read_csv(output_csv)
                existing_files = set(existing_df['source_audio'].tolist())
                print(f"File already contains {len(existing_files)} records")
            except Exception as e:
                print(f"Error reading existing file: {e}")

        audio_files = []
        for file in os.listdir(input_dir):
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                file_path = os.path.join(input_dir, file)
                if file not in existing_files:
                    audio_files.append(file_path)
                else:
                    print(f"File already processed: {file}")

        if not audio_files:
            print("All files have been processed previously")
            if os.path.exists(output_csv):
                return pd.read_csv(output_csv).to_dict('records')
            return []

        print(f"Found {len(audio_files)} new audio files to process")

        batches = [audio_files[i:i + batch_size] for i in range(0, len(audio_files), batch_size)]
        print(f"Files split into {len(batches)} batches of {batch_size} files")

        total_processed = 0
        all_results = []

        for batch_num, batch_files in enumerate(batches, 1):
            print(f"\n{'=' * 60}")
            print(f"Processing batch {batch_num}/{len(batches)}")
            print(f"{'=' * 60}")

            batch_results = []

            for i, audio_file in enumerate(batch_files, 1):
                global_idx = total_processed + i
                print(f"\nProcessing file {global_idx}/{len(audio_files)} (batch {batch_num}, file {i}/{len(batch_files)})")
                try:
                    result = self.process_file(audio_file, quality_threshold, transcribe_all)
                    batch_results.append(result)
                    print(f"Added to list: {os.path.basename(audio_file)}")
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")

            if batch_results:
                batch_df = pd.DataFrame(batch_results)

                write_mode = 'a' if os.path.exists(output_csv) and batch_num > 1 else 'w'
                write_header = not (os.path.exists(output_csv) and batch_num > 1)

                if write_mode == 'a' and batch_num > 1:
                    existing_df = pd.read_csv(output_csv)
                    final_df = pd.concat([existing_df, batch_df], ignore_index=True)
                    final_df = final_df.drop_duplicates(subset=['source_audio'], keep='first')
                    final_df = final_df.sort_values('date')
                    final_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
                else:
                    if os.path.exists(output_csv) and batch_num == 1:
                        existing_df = pd.read_csv(output_csv)
                        final_df = pd.concat([existing_df, batch_df], ignore_index=True)
                        final_df = final_df.drop_duplicates(subset=['source_audio'], keep='first')
                        final_df = final_df.sort_values('date')
                    else:
                        final_df = batch_df.sort_values('date')

                    final_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

                total_processed += len(batch_results)
                all_results.extend(batch_results)

                print(f"\nBatch {batch_num} results:")
                print(f"Files processed in batch: {len(batch_results)}")
                print(f"Total files processed: {total_processed}/{len(audio_files)}")
                print(f"Intermediate results saved to: {output_csv}")

        print(f"\n{'=' * 60}")
        print(f"All batches processing completed!")
        print(f"Total records in file: {len(pd.read_csv(output_csv))}")
        print(f"New records added: {total_processed}")
        print(f"Results saved to: {output_csv}")
        print(f"{'=' * 60}")

        final_df = pd.read_csv(output_csv)
        print("\nStatistics:")
        print(f"Average quality score: {final_df['quality_score'].mean():.1f}")
        print(f"Total text characters: {final_df['text'].str.len().sum()}")
        print(f"Files with score >=8: {(final_df['quality_score'] >= 8).sum()}")

        return final_df.to_dict('records')


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