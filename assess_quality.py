import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import noisereduce as nr
import json


class AudioQualityAssessor:
    def __init__(self, sr=16000):
        self.sr = sr

    def load_audio(self, file_path):
        """Загрузка аудио файла"""
        audio, sr = librosa.load(file_path, sr=self.sr)
        return audio, sr

    def _ensure_scalar(self, value):
        """Преобразует значение в скаляр, если это массив"""
        if isinstance(value, (np.ndarray, list, tuple)):
            # Берем среднее значение массива
            return float(np.mean(value))
        return float(value)

    def calculate_snr(self, audio):
        """Расчет Signal-to-Noise Ratio (SNR)"""
        # Разделение на сегменты для оценки шума
        segment_length = int(0.1 * self.sr)  # 100ms сегменты

        # Убедимся, что сегменты имеют одинаковую длину
        num_segments = len(audio) // segment_length
        if num_segments == 0:
            return 0.0  # Если аудио слишком короткое

        segments = []
        energies = []

        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = start_idx + segment_length
            segment = audio[start_idx:end_idx]

            # Убедимся, что сегмент не пустой
            if len(segment) > 0:
                segments.append(segment)
                # Вычисляем энергию как скалярное значение
                energy = np.sum(segment ** 2)
                energies.append(float(energy))  # Явное преобразование в float

        if len(energies) == 0:
            return 0.0

        # Находим сегменты с наименьшей энергией (предположительно шум)
        # Теперь energies - список чисел, а не массивов
        sorted_data = sorted(zip(energies, segments), key=lambda x: x[0])
        noise_segments_count = max(1, len(sorted_data) // 4)  # Не менее 1 сегмента
        noise_segments = sorted_data[:noise_segments_count]

        noise_energy = np.mean([energy for energy, _ in noise_segments])

        # Полная энергия сигнала
        signal_energy = np.mean(energies)

        if noise_energy == 0:
            return float('inf')

        snr = 10 * np.log10(signal_energy / noise_energy)
        return float(snr)

    def calculate_spectral_centroid(self, audio):
        """Спектральный центроид - мера яркости звука"""
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sr)
        return np.mean(spectral_centroids)

    def calculate_spectral_rolloff(self, audio):
        """Спектральный rolloff - мера высокочастотного содержания"""
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)
        return np.mean(rolloff)

    def calculate_zero_crossing_rate(self, audio):
        """Rate zero-crossing - мера шума"""
        zcr = librosa.feature.zero_crossing_rate(audio)
        return np.mean(zcr)

    def calculate_energy(self, audio):
        """Энергия сигнала"""
        return np.mean(audio ** 2)

    def assess_quality(self, file_path):
        """Комплексная оценка качества аудио"""
        audio, sr = self.load_audio(file_path)

        metrics = {
            'snr': self.calculate_snr(audio),
            'spectral_centroid': self.calculate_spectral_centroid(audio),  # high or low voice/sound; "brightness"
            'spectral_rolloff': self.calculate_spectral_rolloff(audio),  # whether the sound is detailed or smooth; higher better
            'zero_crossing_rate': self.calculate_zero_crossing_rate(audio),  # lower better
            'energy': self.calculate_energy(audio) # mean amplitude
        }

        # Оценка качества на основе метрик
        quality_score = self._calculate_quality_score(metrics)

        return quality_score, metrics

    def _calculate_quality_score(self, metrics):
        """Вычисление общего скора качества с защитой от массивов"""
        score = 0

        # Преобразуем все метрики в числа
        snr = self._ensure_scalar(metrics['snr'])
        spectral_centroid = self._ensure_scalar(metrics['spectral_centroid'])
        zero_crossing_rate = self._ensure_scalar(metrics['zero_crossing_rate'])
        energy = self._ensure_scalar(metrics['energy'])

        # Теперь безопасные сравнения
        if snr > 20:
            score += 3
        elif snr > 10:
            score += 2
        elif snr > 5:
            score += 1

        # Спектральный центроид
        if 1000 < spectral_centroid < 3000:
            score += 2
        elif 500 < spectral_centroid < 4000:
            score += 1

        # Zero-crossing rate
        if zero_crossing_rate < 0.1:
            score += 2
        elif zero_crossing_rate < 0.2:
            score += 1

        # Энергия
        if 0.001 < energy < 0.1:
            score += 2
        elif 0.0001 < energy < 0.5:
            score += 1

        return min(10, score)


def process_audio_files(audio_files, quality_threshold=7):
    """Обработка аудио файлов с фильтрацией по качеству"""
    assessor = AudioQualityAssessor()

    result = {}
    good_files = []
    bad_files = []

    for filename in os.listdir(audio_files):
        if not filename.lower().endswith('.wav'):
            continue

        file_path = os.path.join(audio_files, filename)
        #try:
        quality_score, metrics = assessor.assess_quality(file_path)

        print(f"Файл: {filename}")
        print(f"Качество: {quality_score}/10")
        print(f"SNR: {metrics['snr']:.2f} dB")
        print(f"Zero-crossing rate: {metrics['zero_crossing_rate']:.3f}")
        print("-" * 50)

        # Сохраняем детальную информацию
        result[filename] = {
            'quality_score': float(quality_score),
            'snr': float(metrics['snr']),
            'zero_crossing_rate': float(metrics['zero_crossing_rate']),
            'spectral_centroid': float(metrics['spectral_centroid']),
            'energy': float(metrics['energy']),
            'status': 'good' if quality_score >= quality_threshold else 'bad'
        }

        # Разделяем файлы по качеству
        if quality_score >= quality_threshold:
            good_files.append(filename)
        else:
            bad_files.append(filename)

        # except Exception as e:
        #     print(f"Ошибка при обработке {file_path}: {e}")
        #     result[filename] = {'error': str(e)}

    # Добавляем summary в результат
    result['_summary'] = {
        'total_files': len([f for f in os.listdir(audio_files) if f.lower().endswith('.wav')]),
        'processed_files': len(result) - (1 if '_summary' in result else 0),
        'good_quality_files': len(good_files),
        'bad_quality_files': len(bad_files),
        'quality_threshold': quality_threshold
    }

    return result, good_files, bad_files

# Пример использования
if __name__ == "__main__":
    # Список ваших аудио файлов
    audio_files = 'audio_pool/'

    # Оценка качества
    result, good_files, bad_files = process_audio_files(audio_files)

    # Сохраняем основной результат как JSON
    with open('quality_assessment.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Сохраняем списки хороших и плохих файлов отдельно
    with open('good_files.json', 'w', encoding='utf-8') as f:
        json.dump(good_files, f, ensure_ascii=False, indent=2)

    with open('bad_files.json', 'w', encoding='utf-8') as f:
        json.dump(bad_files, f, ensure_ascii=False, indent=2)

    print(f"Обработано файлов: {result['_summary']['processed_files']}")
    print(f"Хорошее качество: {len(good_files)}")
    print(f"Плохое качество: {len(bad_files)}")