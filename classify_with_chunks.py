import re
import numpy as np
from collections import defaultdict, Counter
from transformers import pipeline
import torch
import json


class LongCallEmotionAnalyzer:
    def __init__(self, model_name="blanchefort/rubert-base-cased-sentiment"):
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            return_all_scores=True,
            max_length=512,
            truncation=True
        )

    def smart_chunking(self, text, chunk_size=2000, overlap=200):
        """Split long text into overlapping chunks at natural boundaries"""
        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            print('next paragraph...')
            # If adding this paragraph doesn't exceed chunk size
            if len(current_chunk) + len(paragraph) <= chunk_size:
                current_chunk += " " + paragraph if current_chunk else paragraph
            else:
                # If current chunk has content, save it
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # If paragraph itself is larger than chunk_size, split it
                if len(paragraph) > chunk_size:
                    sentences = re.split(r'[.!?]+', paragraph)
                    current_sentences = []
                    current_length = 0

                    for sentence in sentences:
                        if current_length + len(sentence) <= chunk_size:
                            current_sentences.append(sentence)
                            current_length += len(sentence)
                        else:
                            if current_sentences:
                                chunks.append('. '.join(current_sentences).strip() + '.')
                            current_sentences = [sentence]
                            current_length = len(sentence)

                    if current_sentences:
                        current_chunk = '. '.join(current_sentences).strip() + '.'
                else:
                    current_chunk = paragraph

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def analyze_long_call(self, call_text):
        """Analyze very long phone calls"""
        chunks = self.smart_chunking(call_text)
        print(f"Analyzing {len(chunks)} chunks...")

        chunk_results = []
        emotion_evolution = []

        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:  # Skip very short chunks
                continue

            try:
                result = self.classifier(chunk[:1000])[0]  # Further limit for safety

                # Get top emotion for this chunk
                top_emotion = max(result, key=lambda x: x['score'])

                chunk_results.append({
                    'chunk_id': i,
                    'text_preview': chunk[:100] + "...",
                    'emotions': {item['label']: item['score'] for item in result},
                    'top_emotion': top_emotion['label'],
                    'top_confidence': top_emotion['score']
                })

                emotion_evolution.append({
                    'position': i,
                    'emotion': top_emotion['label'],
                    'confidence': top_emotion['score']
                })

            except Exception as e:
                print(f"Error in chunk {i}: {e}")
                continue

        # Aggregate results
        overall_emotions = self.aggregate_emotions(chunk_results)
        emotion_trend = self.analyze_emotion_trends(emotion_evolution)

        return {
            'total_chunks': len(chunk_results),
            'overall_emotions': overall_emotions,
            'dominant_emotion': max(overall_emotions.items(), key=lambda x: x[1])[0],
            'emotion_evolution': emotion_evolution,
            'emotion_trends': emotion_trend,
            'chunk_analysis': chunk_results,
            'emotional_volatility': self.calculate_volatility(emotion_evolution)
        }

    def aggregate_emotions(self, chunk_results):
        """Weighted aggregation of emotions across chunks"""
        emotion_weights = defaultdict(list)

        for chunk in chunk_results:
            for emotion, score in chunk['emotions'].items():
                # Weight by chunk confidence and length
                weight = score * chunk['top_confidence']
                emotion_weights[emotion].append(weight)

        return {
            emotion: np.mean(weights)
            for emotion, weights in emotion_weights.items()
        }

    def analyze_emotion_trends(self, emotion_evolution):
        """Analyze how emotions change over time"""
        if len(emotion_evolution) < 2:
            return {"trend": "insufficient_data"}

        emotions = [item['emotion'] for item in emotion_evolution]
        emotion_changes = []

        for i in range(1, len(emotions)):
            if emotions[i] != emotions[i - 1]:
                emotion_changes.append(f"{emotions[i - 1]}→{emotions[i]}")

        return {
            'unique_emotions': list(set(emotions)),
            'emotion_transitions': Counter(emotion_changes),
            'stability': len(set(emotions)) / len(emotions),  # Lower = more stable
            'most_frequent_emotion': Counter(emotions).most_common(1)[0][0]
        }

    def calculate_volatility(self, emotion_evolution):
        """Calculate how much emotions fluctuate"""
        if len(emotion_evolution) < 2:
            return 0

        confidences = [item['confidence'] for item in emotion_evolution]
        return np.std(confidences)


# Usage
analyzer = LongCallEmotionAnalyzer()

file_path = 'результат_анализа_large.json'
text = json.load(open(file_path, 'r', encoding='utf-8'))
text = text['transcription']['text']
print(text[:300])

result = analyzer.analyze_long_call(text)
print(result)