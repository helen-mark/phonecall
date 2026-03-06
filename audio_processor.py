import os
import json
import time
import shutil
import yaml
import pandas as pd
from datetime import datetime
from pathlib import Path
import subprocess
from preprocess_calls_full import AudioProcessor
import assign_tags_from_fixed_list
from assign_tags_from_fixed_list import JsonFileTaggingAgent
from typing import Union
#from llama_cpp import Llama

import phonecall
import phonecall.preprocess_calls_full
from phonecall.colab.reload_recursive import reload_recursive

reload_recursive(phonecall.preprocess_calls_full)
reload_recursive(assign_tags_from_fixed_list)


class SmartAudioProcessor:

    def __init__(self, model, node_url, base_path, drive_audio_path, output_csv_path,
                 total_space_gb=80, batch_size_gb=2, config_path='phonecall/config.yml', do_audio=True, do_tags=True):
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        my_tags = config.get('tags_list', [])
        self.base_path = base_path
        self.drive_audio_path = drive_audio_path
        self.output_csv_path = output_csv_path
        self.total_space = total_space_gb
        self.batch_size = batch_size_gb

        if self.do_tags:
        self.tagger = JsonFileTaggingAgent(
            model=model,
            node_url=node_url,
            tags_list=my_tags,
            mail=False
        )
        print(f'base path: {self.base_path}')
        self.local_temp_dir = os.path.join(self.base_path, "temp_audio")
        self.local_whisper_dir = os.path.join(self.base_path, "whisper_output")
        self.local_batch_dir = os.path.join(self.base_path, "current_batch")

        # Create local dirs:
        for dir_path in [self.local_temp_dir, self.local_whisper_dir, self.local_batch_dir]:
            os.makedirs(dir_path, exist_ok=True)

        self.processed_files_log = os.path.join(self.base_path, "processed_files.json")
        
        if self.do_audio:
            self.ap = AudioProcessor(model_size='large')


        print(f" Initializing SmartAudioProcessor")
        print(f" Audiofiles: {drive_audio_path}")
        print(f" Batch limit! : {batch_size_gb} GB")
        print(f" Total space: {total_space_gb} GB")

    # ========== Main method ==========

    def process_large_dataset(self):

        print("=" * 60)
        print("Start processing large dataset")
        print("=" * 60)

        # 1. Only get filenames without loading the files
        all_files = self._list_files_without_download()
        if not all_files:
            print("No files for processing")
            return []

        total_files = len(all_files)
        total_gb = self._estimate_total_size_gb(all_files)

        print(f"Total files: {total_files:,}")
        print(f"Total volume: {total_gb:.1f} GB")
        print(f"Batch size: {self.batch_size} GB")
        print(f"N bathces: {int(total_gb / self.batch_size) + 1}")

        # 2. Which files are already processed?
        processed = self._load_processed_list()
        to_process = [f for f in all_files if f not in processed]

        print(f"\n Processing status:")
        print(f" Already processed: {len(processed):,} files")
        print(f" To process: {len(to_process):,} files")

        if not to_process:
            print("\n All files are processed!")
            return []

        # 3. Process bathces
        batches = self._create_batches(to_process)
        print(f"\n Создано {len(batches)} батчей для обработки")

        all_results = []
        start_time = time.time()

        for batch_num, batch_files in enumerate(batches, 1):
            batch_start = time.time()

            print(f"\n{'=' * 50}")
            print(f"⚡ BATCH {batch_num}/{len(batches)}")
            print(f"{'=' * 50}")

            batch_size_gb = sum(self._get_file_size_gb(f) for f in batch_files)
            print(f" Files per batch: {len(batch_files)}")
            print(f" Batch size: {batch_size_gb:.2f} GB")

            # Process batch
            try:
                batch_results = self._process_batch(batch_files, batch_num)

                if batch_results:
                    all_results.extend(batch_results)
                    self._append_to_main_csv(batch_results)
                    self._update_processed_list(batch_files)

                batch_time = time.time() - batch_start
                avg_time_per_file = batch_time / len(batch_files) if batch_files else 0

                print(f"\n Batch {batch_num} processed in {batch_time:.1f} sec")
                print(f" Avg time per batch: {avg_time_per_file:.1f} sec")

                remaining_batches = len(batches) - batch_num
                if remaining_batches > 0:
                    estimated_remaining = remaining_batches * (time.time() - start_time) / batch_num
                    print(f" Time left: ~{estimated_remaining / 60:.1f} min")

            except Exception as e:
                print(f" Batch processing error {batch_num}: {e}")
                import traceback
                traceback.print_exc()

                continue

        self._remove_duplicates_from_csv()

        total_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f" Processing finished!")
        print(f"{'=' * 60}")
        print(f" Processed: {len(all_results):,}")
        print(f" Total time: {total_time / 60:.1f} мин")
        print(f" Speed: {len(all_results) / total_time * 60:.1f} файлов/мин")
        print(f" Results saved to: {self.output_csv_path}")

        return all_results

    def _append_to_main_csv(self, batch_results):
        if not batch_results:
            print("No results to save")
            return

        df_batch = pd.DataFrame(batch_results)

        if os.path.exists(self.output_csv_path):
            try:
                df_existing = pd.read_csv(self.output_csv_path)
                df_combined = pd.concat([df_existing, df_batch], ignore_index=True)
                df_combined.to_csv(self.output_csv_path, index=False, encoding='utf-8')

                print(f" Uppended {len(batch_results)} entries to {self.output_csv_path}")
                print(f" Now have total: {len(df_combined)} entries")

            except Exception as e:
                print(f"Error uppending an entrty: {e}")
                df_batch.to_csv(self.output_csv_path, index=False, encoding='utf-8')
                print(f"Created new file with {len(batch_results)} entries")
        else:
            df_batch.to_csv(self.output_csv_path, index=False, encoding='utf-8')
            print(f"Created new file {self.output_csv_path} with {len(batch_results)} entries")

    def _remove_duplicates_from_csv(self):
        if not os.path.exists(self.output_csv_path):
            return

        try:
            df = pd.read_csv(self.output_csv_path)

            if 'source_audio' in df.columns:
                duplicates = df.duplicated(subset=['source_audio'], keep='last').sum()

                if duplicates > 0:
                    print(f"Найдено {duplicates} дубликатов, удаляю...")

                    df_clean = df.drop_duplicates(subset=['source_audio'], keep='last')
                    df_clean.to_csv(self.output_csv_path, index=False, encoding='utf-8')

                    print(f" Удалено {duplicates} дубликатов")
                    print(f" Осталось {len(df_clean)} уникальных записей")
            else:
                print("  Колонка 'source_audio' не найдена в CSV")

        except Exception as e:
            print(f"  Ошибка при удалении дубликатов: {e}")

    def _load_processed_list(self):
        processed_files = set()

        if os.path.exists(self.output_csv_path):
            try:
                df = pd.read_csv(self.output_csv_path)
                if 'source_audio' in df.columns:
                    csv_processed = set(df['source_audio'].dropna().unique())
                    processed_files.update(csv_processed)
                    print(f"Из основного CSV загружено: {len(csv_processed)} файлов")
            except Exception as e:
                print(f"  Ошибка чтения основного CSV: {e}")

        # if os.path.exists(self.processed_files_log):
        #     try:
        #         with open(self.processed_files_log, 'r') as f:
        #             data = json.load(f)
        #             if isinstance(data, list):
        #                 log_processed = set(data)
        #                 processed_files.update(log_processed)
        #                 print(f"📖 Из лога загружено: {len(log_processed)} файлов")
        #     except Exception as e:
        #         print(f"  Ошибка чтения лога: {e}")

        full_paths_processed = set()
        for filename in processed_files:
            full_path = os.path.join(self.drive_audio_path, filename)
            full_paths_processed.add(full_path)

        return full_paths_processed

    def _update_processed_list(self, processed_files):
        """Обновляет список обработанных файлов"""
        try:
            # Загружаем текущий список из лога
            current_processed = []
            if os.path.exists(self.processed_files_log):
                with open(self.processed_files_log, 'r') as f:
                    current_processed = json.load(f)

            # Добавляем новые файлы (только имена)
            new_processed = [os.path.basename(f) for f in processed_files]
            current_processed.extend(new_processed)

            # Удаляем дубликаты
            current_processed = list(set(current_processed))

            # Сохраняем
            with open(self.processed_files_log, 'w') as f:
                json.dump(current_processed, f)

            print(f"Лог обновлен: +{len(new_processed)} файлов, всего {len(current_processed)}")

        except Exception as e:
            print(f"  Ошибка обновления лога: {e}")

    def _process_batch(self, batch_files, batch_num):
        """Обрабатывает один батч файлов"""
        batch_results = []

        # 1. Очищаем локальные директории
        self._cleanup_local_dirs()

        # 2. Скачиваем файлы батча в локальную ФС
        local_files = []
        print(f"Скачиваю {len(batch_files)} файлов в локальную ФС...")

        for file_path in batch_files:
            try:
                filename = os.path.basename(file_path)
                local_path = os.path.join(self.local_batch_dir, filename)

                # Копируем из Drive в локальную ФС
                shutil.copy2(file_path, local_path)
                local_files.append(local_path)

                # Проверяем, что файл скопировался
                if os.path.exists(local_path):
                    size_mb = os.path.getsize(local_path) / (1024 * 1024)
                else: 
                    print(f"   {filename} - не скопировался")

            except Exception as e:
                print(f"   Ошибка копирования {os.path.basename(file_path)}: {e}")

        print(f" Скачано: {len(local_files)}/{len(batch_files)} файлов")

        if not local_files:
            print("  Нет файлов для обработки в этом батче")
            return []

        print("\n Начинаю обработку аудио...")

        for i, local_file in enumerate(local_files, 1):
            try:
                print(f"\n[{i}/{len(local_files)}] Обрабатываю: {os.path.basename(local_file)}")

                # 3.1 Whisper: Аудио → Текст
                print("   Шаг 1: Конвертация аудио в текст...")
                whisper_result = self._run_whisper_locally(local_file)

                if not whisper_result or 'text' not in whisper_result:
                    print("    Whisper не вернул текст")
                    continue

                text = whisper_result['text']
                print(f"  Текст извлечен ({len(text)} символов)")

                # 3.2 LLM: Текст → Теги
                print("    Шаг 2: Тегирование текста...")
                tagging_result = self._run_tagging_locally(text)

                tags = tagging_result.get('result', [])
                summary = tagging_result.get('summary', '')

                print(f"  Добавлено тегов: {len(tags)}")

                # 3.3 Создаем результат
                result = {
                    'source_audio': os.path.basename(local_file),
                    'text': text,
                    'tags': str(tags),  # Сохраняем как строку
                    'summary': summary,
                    'text_length': len(text),
                    'processing_date': datetime.now().isoformat(),
                    'batch_number': batch_num,
                    'whisper_model': whisper_result.get('model', 'unknown'),
                    'audio_duration': whisper_result.get('duration', 0)
                }

                # Добавляем дополнительные поля из whisper_result
                if 'date' in whisper_result:
                    result['date'] = whisper_result['date']
                if 'quality_score' in whisper_result:
                    result['quality_score'] = whisper_result['quality_score']

                batch_results.append(result)

                # 3.4 Удаляем обработанный локальный файл
                os.remove(local_file)
                print(f"   Локальный файл удален")

            except Exception as e:
                print(f"   Ошибка обработки файла: {e}")
                import traceback
                traceback.print_exc()

        # 4. Очищаем временные файлы
        self._cleanup_local_dirs()

        print(f"\n Батч {batch_num} обработан: {len(batch_results)}/{len(batch_files)} успешно")

        return batch_results

    def _run_whisper_locally(self, local_audio_path):
        """Запускает Whisper для локального аудиофайла"""
        try:
            return self.ap.process_file(local_audio_path, 7, False)
        except Exception as e:
            print(f" Ошибка Whisper: {e}")
            return {'text': '', 'error': str(e)}

    def _run_tagging_locally(self, text):
        """Запускает тегирование для текста"""
        try:
            return self.tagger.get_tags_from_llm(text)
        except Exception as e:
            print(f" Ошибка тегирования: {e}")
            return {'result': [], 'summary': f'Ошибка: {str(e)}'}

    # ========== Auxiliary methods ==========

    def _list_files_without_download(self):
        if not os.path.exists(self.drive_audio_path):
            print(f" Папка не найдена: {self.drive_audio_path}")
            return []

        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma']

        all_files = []
        for ext in audio_extensions:
            pattern = f"*{ext}"
            files = list(Path(self.drive_audio_path).glob(pattern.lower()))
            files.extend(Path(self.drive_audio_path).glob(pattern.upper()))

            for file_path in files:
                if file_path.is_file():
                    all_files.append(str(file_path))

        all_files.sort()

        print(f" Найдено {len(all_files)} аудиофайлов")
        return all_files

    def _estimate_total_size_gb(self, file_paths):
        """Оценивает общий размер файлов в GB"""
        total_bytes = 0

        for file_path in file_paths:
            try:
                total_bytes += os.path.getsize(file_path)
            except:
                total_bytes += 50 * 1024 * 1024  # 50 MB'

        return total_bytes / (1024 ** 3)

    def _get_file_size_gb(self, file_path):
        try:
            return os.path.getsize(file_path) / (1024 ** 3)
        except:
            return 0.05  

    def _create_batches(self, file_paths):
        batches = []
        current_batch = []
        current_batch_size = 0
        print('Creating batches...')

        for file_path in file_paths:
            file_size_gb = self._get_file_size_gb(file_path)

            if current_batch_size + file_size_gb > self.batch_size and current_batch:
                batches.append(current_batch.copy())
                current_batch = []
                current_batch_size = 0
                

            current_batch.append(file_path)
            current_batch_size += file_size_gb

        if current_batch:
            batches.append(current_batch)

        optimized_batches = self._optimize_batches(batches)

        return optimized_batches

    def _optimize_batches(self, batches):
        if len(batches) <= 1:
            return batches

        optimized = []
        current_batch = []
        current_size = 0

        for batch in batches:
            batch_size = sum(self._get_file_size_gb(f) for f in batch)

            if current_size + batch_size <= self.batch_size * 1.2:
                current_batch.extend(batch)
                current_size += batch_size
            else:
                if current_batch:
                    optimized.append(current_batch)
                current_batch = batch.copy()
                current_size = batch_size

        if current_batch:
            optimized.append(current_batch)

        print(f" Оптимизировано: {len(batches)} → {len(optimized)} батчей")

        return optimized

    def _cleanup_local_dirs(self):
        for dir_path in [self.local_temp_dir, self.local_whisper_dir, self.local_batch_dir]:
            if os.path.exists(dir_path):
                for file in os.listdir(dir_path):
                    try:
                        os.remove(os.path.join(dir_path, file))
                    except:
                        pass

        print(" Локальные директории очищены")

    def get_processing_stats(self):
        stats = {
            'total_files': 0,
            'processed_files': 0,
            'remaining_files': 0,
            'estimated_size_gb': 0,
            'last_processed': None
        }

        try:
            all_files = self._list_files_without_download()
            stats['total_files'] = len(all_files)
            stats['estimated_size_gb'] = self._estimate_total_size_gb(all_files)

            processed = self._load_processed_list()
            stats['processed_files'] = len(processed)
            stats['remaining_files'] = len([f for f in all_files if f not in processed])


            if os.path.exists(self.output_csv_path):
                mod_time = os.path.getmtime(self.output_csv_path)
                stats['last_processed'] = datetime.fromtimestamp(mod_time).strftime("%d.%m.%Y %H:%M")

        except Exception as e:
            print(f"  Ошибка получения статистики: {e}")

        return stats


def main():
    DRIVE_AUDIO_PATH = "/content/drive/MyDrive/MCP_Call_Analytics/audio_raw"
    OUTPUT_CSV_PATH = "/content/drive/MyDrive/MCP_Call_Analytics/csv_calls/calls.csv"

    print(" SMART AUDIO PROCESSOR - ЗАПУСК")
    print("=" * 50)

    # Инициализация процессора
    processor = SmartAudioProcessor(
        drive_audio_path=DRIVE_AUDIO_PATH,
        output_csv_path=OUTPUT_CSV_PATH,
        total_space_gb=50,
        batch_size_gb=2
    )

    # Показываем статистику
    stats = processor.get_processing_stats()
    print(f"\nСТАТИСТИКА ДО ОБРАБОТКИ:")
    print(f"Всего файлов: {stats['total_files']:,}")
    print(f"Обработано: {stats['processed_files']:,}")
    print(f"Осталось: {stats['remaining_files']:,}")
    print(f"Общий объем: {stats['estimated_size_gb']:.1f} GB")

    if stats['last_processed']:
        print(f" Последняя обработка: {stats['last_processed']}")

    if stats['remaining_files'] == 0:
        print("\n Все файлы уже обработаны!")
        return

    confirm = input(f"\n Начать обработку {stats['remaining_files']:,} файлов? (y/n): ")

    if confirm.lower() != 'y':
        print("Обработка отменена")
        return

    print("\n ЗАПУСК ОБРАБОТКИ...")
    results = processor.process_large_dataset()

    if results:
        print(f"\n ОБРАБОТКА ЗАВЕРШЕНА УСПЕШНО!")
        print(f" Обработано записей: {len(results):,}")
        print(f" Результаты: {OUTPUT_CSV_PATH}")

        if os.path.exists(OUTPUT_CSV_PATH):
            df = pd.read_csv(OUTPUT_CSV_PATH)
            print(f" Размер файла: {len(df)} строк, {len(df.columns)} колонок")
            print(" Колонки:", list(df.columns))
    else:
        print("\n Обработка завершена без результатов")


if __name__ == "__main__":
    main()