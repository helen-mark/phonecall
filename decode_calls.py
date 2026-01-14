import json
import librosa
import numpy as np
import whisper
import os

#warnings.filterwarnings('ignore')
import soundfile as sf
from pydub import AudioSegment

import noisereduce as nr


def get_audio_info(file_path):
    """Получить информацию об аудиофайле"""

    print(f"Анализ файла: {file_path}")
    print("=" * 40)

    # Способ 1: через librosa (лучший для анализа)
    try:
        y, sr = librosa.load(file_path, sr=None)  # sr=None - загружаем с исходной частотой
        print(f"📊 Librosa: {sr} Hz")
        print(f"📏 Длительность: {len(y) / sr:.2f} секунд")
        print(f"🎵 Количество каналов: {y.ndim}")
    except Exception as e:
        print(f"Ошибка librosa: {e}")

    print("-" * 20)

    # Способ 2: через soundfile
    try:
        info = sf.info(file_path)
        print(f"📊 SoundFile: {info.samplerate} Hz")
        print(f"📏 Кадров: {info.frames}")
        print(f"🎵 Каналы: {info.channels}")
    except Exception as e:
        print(f"Ошибка soundfile: {e}")

    print("-" * 20)

    # Способ 3: через pydub
    try:
        audio = AudioSegment.from_file(file_path)
        print(f"📊 Pydub: {audio.frame_rate} Hz")
        print(f"📏 Длительность: {len(audio) / 1000:.2f} секунд")
        print(f"🎵 Каналы: {'Моно' if audio.channels == 1 else 'Стерео'}")
        print(f"📝 Размер сэмпла: {audio.sample_width} байта")
    except Exception as e:
        print(f"Ошибка pydub: {e}")




class AudioAnalyzer:
    def __init__(self, model_size):
        """
        Инициализация анализатора аудио
        model_size: "tiny", "base", "small", "medium", "large"
        """
        print("Загрузка модели Whisper...")
        self.asr_model = whisper.load_model(model_size)
        print("Модель загружена!")

    def extract_audio_features(self, audio_path):
        """
        Извлечение интонационных признаков из аудио
        """
        print(f"Извлечение признаков из {audio_path}...")

        # Загрузка аудио
        y, sr = librosa.load(audio_path, sr=16000)

        features = {}

        # 1. Основные статистики аудио
        features['duration'] = len(y) / sr
        features['sample_rate'] = sr

        # 2. Громкость (Energy)
        rms = librosa.feature.rms(y=y)
        features['loudness'] = {
            'mean': float(np.mean(rms)),
            'std': float(np.std(rms)),
            'max': float(np.max(rms)),
            'min': float(np.min(rms))
        }

        # 3. Высота тона (Pitch)
        pitch, voiced_flag, voiced_probs = librosa.pyin(y=y, fmin=50, fmax=400, sr=sr)
        pitch_values = pitch[~np.isnan(pitch)]

        if len(pitch_values) > 0:
            features['pitch'] = {
                'mean': float(np.mean(pitch_values)),
                'std': float(np.std(pitch_values)),
                'max': float(np.max(pitch_values)),
                'min': float(np.min(pitch_values)),
                'range': float(np.max(pitch_values) - np.min(pitch_values))
            }
        else:
            features['pitch'] = {'mean': 0, 'std': 0, 'max': 0, 'min': 0, 'range': 0}

        # 4. Темп речи (через onset detection)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = float(tempo)
        features['beat_frames'] = len(beats)

        # 5. Спектральные характеристики (MFCC для тембра)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfcc_stats'] = {
            'mfcc_mean': [float(x) for x in np.mean(mfcc, axis=1)],
            'mfcc_std': [float(x) for x in np.std(mfcc, axis=1)]
        }

        # 6. Zero-crossing rate (показатель шума/резкости)
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zero_crossing_rate'] = {
            'mean': float(np.mean(zcr)),
            'std': float(np.std(zcr))
        }

        # 7. Паузы и сегментация
        # Определение тихих участков как потенциальных пауз
        frame_length = 2048
        hop_length = 512
        rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        silence_threshold = np.mean(rms_frames) * 0.3
        silent_frames = np.sum(rms_frames < silence_threshold)
        total_frames = len(rms_frames)

        features['pauses'] = {
            'silence_ratio': float(silent_frames / total_frames),
            'total_silent_frames': int(silent_frames),
            'silence_threshold': float(silence_threshold)
        }

        print("Признаки успешно извлечены!")
        return features

    def transcribe_audio(self, audio_path):
        """
        Транскрибация аудио в текст
        """
        print(f"Транскрибация {audio_path}...")

        # Транскрибация с помощью Whisper
        result = self.asr_model.transcribe(audio_path)

        transcription = {
            'text': result['text'],
            'language': result.get('language', 'ru'),
            'segments': []
        }

        # Сохраняем сегменты с временными метками
        for segment in result.get('segments', []):
            transcription['segments'].append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'],
                'confidence': segment.get('confidence', 0)
            })

        print("Транскрибация завершена!")
        return transcription

    def analyze_audio_file(self, audio_path, output_file=None):
        """
        Полный анализ аудиофайла
        """
        print(f"\n=== Начало анализа {audio_path} ===")

        # Транскрибация
        transcription = self.transcribe_audio(audio_path)

        # Извлечение признаков
        audio_features = self.extract_audio_features(audio_path)

        # Объединение результатов
        result = {
            'audio_file': audio_path,
            'transcription': transcription,
            'audio_features': audio_features,
            'summary': self._generate_summary(transcription, audio_features)
        }

        # Сохранение в файл
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\nРезультаты сохранены в: {output_file}")

        print("=== Анализ завершен ===")
        return result

    def _generate_summary(self, transcription, features):
        """
        Генерация текстового описания характеристик
        """
        text_length = len(transcription['text'])
        duration = features['duration']

        # Анализ темпа речи
        words_per_minute = (text_length / 6) / (duration / 60) if duration > 0 else 0

        # Анализ интонации
        pitch_variability = features['pitch']['std']
        loudness_variability = features['loudness']['std']

        summary = {
            'text_length': text_length,
            'audio_duration_seconds': round(duration, 2),
            'words_per_minute_approx': round(words_per_minute, 2),
            'speech_characteristics': []
        }

        # Характеристики речи
        if pitch_variability > 20:
            summary['speech_characteristics'].append("выраженная интонационная вариативность")
        else:
            summary['speech_characteristics'].append("ровная интонация")

        if loudness_variability > 0.01:
            summary['speech_characteristics'].append("переменная громкость")
        else:
            summary['speech_characteristics'].append("стабильная громкость")

        if features['pauses']['silence_ratio'] > 0.3:
            summary['speech_characteristics'].append("много пауз")
        else:
            summary['speech_characteristics'].append("непрерывная речь")

        return summary


def main():
    # 2025-10-09_08-52-53.022174_from_79851005767_to_79258972401_session_5396115979_talk_16k.wav  - плохое качество звука. Ковер забрали не с того юр лица, срочно перезвонить
    audio_dir = 'audio_pool/'
    out_dir = 'transcriptions/'
    with open('quality_assessment.json', 'r') as f:
        quality_data = json.load(f)

    # Фильтруем файлы с качеством >= 7
    high_quality_files = [
        filename for filename in os.listdir(audio_dir)
        if filename.endswith('.wav')
           and filename in quality_data
           and quality_data[filename].get('quality_score', 0) >= 7
    ]

    print(f"Найдено {len(high_quality_files)} файлов высокого качества")
    # Инициализация анализатора
    analyzer = AudioAnalyzer(model_size="large")  # Используйте "tiny" для быстрого тестирования

    for filename in high_quality_files:

        audio_file = os.path.join(audio_dir, filename)
        output_filename = filename[:-4] + '.json'
        output_file = os.path.join(out_dir, output_filename)

        n = 0
        if output_filename in os.listdir(out_dir):
            n += 1
            continue
    print(f"{n} files already processed")

    for filename in high_quality_files:

        audio_file = os.path.join(audio_dir, filename)
        output_filename = filename[:-4] + '.json'
        output_file = os.path.join(out_dir, output_filename)

        n = 0
        if output_filename in os.listdir(out_dir):
            n += 1
            continue
        print(f"{n} files already processed")

        get_audio_info(audio_file)


        try:
            result = analyzer.analyze_audio_file(audio_file, output_file)

            # Вывод кратких результатов в консоль
            print("\n" + "=" * 50)
            print("КРАТКИЕ РЕЗУЛЬТАТЫ:")
            print("=" * 50)
            print(f"Текст: {result['transcription']['text'][:200]}...")
            print(f"Длительность: {result['summary']['audio_duration_seconds']} сек.")
            print(f"Язык: {result['transcription']['language']}")
            print(f"Характеристики: {', '.join(result['summary']['speech_characteristics'])}")
            print(f"Средняя высота тона: {result['audio_features']['pitch']['mean']:.2f} Hz")
            print(f"Вариативность громкости: {result['audio_features']['loudness']['std']:.4f}")

        except Exception as e:
            print(f"Ошибка при анализе файла: {e}")


if __name__ == "__main__":
    main()