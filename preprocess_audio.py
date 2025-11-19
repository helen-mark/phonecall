from pydub import AudioSegment
import librosa
import noisereduce as nr
import soundfile as sf
import os

def preprocess_audio(input_path, output_path):
    # Загрузка
    audio = AudioSegment.from_wav(input_path)

    # Увеличиваем громкость
    audio = audio + 10  # dB

    # Обрезаем тихие участки
    audio = audio.strip_silence(silence_len=500, silence_thresh=-40)

    # Экспортируем для обработки
    audio.export("temp.wav", format="wav")

    # Шумоподавление
    y, sr = librosa.load("temp.wav", sr=16000)
    y_denoised = nr.reduce_noise(y=y, sr=sr)

    # Сохраняем результат
    sf.write(output_path, y_denoised, sr)

    return output_path

if __name__=='__main__':
    for filename in os.listdir('audio_pool'):
        src = os.path.join('audio_pool', filename)
        dst = src[:-4] + '_preprocessed.wav'
        converted_file = preprocess_audio(src, dst)