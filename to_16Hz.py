import librosa

import os
import soundfile as sf
from pydub import AudioSegment


def quick_check(file_path):
    """Быстрая проверка частоты файла"""
    y, sr = librosa.load(file_path, sr=None)
    return sr


def ensure_16k(audio_path, output_path=None):
    """Гарантировать, что аудио имеет частоту 16 кГц"""

    if output_path is None:
        output_path = audio_path.replace('.mp3', '_16k.wav')

    # Проверяем текущую частоту
    sr = quick_check(audio_path)

    if sr == 16000:
        print(f"✅ Файл уже имеет частоту 16 кГц")
        return audio_path
    else:
        print(f"🔄 Конвертируем из {sr} Hz в 16000 Hz...")

        # Конвертируем
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format="wav")

        # Проверяем результат
        new_sr = quick_check(output_path)
        print(f"✅ Сконвертировано: {new_sr} Hz")
        return output_path


if __name__=='__main__':
    for filename in os.listdir('audio_pool'):
        converted_file = ensure_16k(os.path.join('audio_pool', filename))