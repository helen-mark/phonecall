from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import json

def setup_advanced_russian_sentiment():
    """Use specialized Russian sentiment models"""

    # Model optimized for Russian business context
    sentiment_analyzer = pipeline(
        "text-classification",
        model="blanchefort/rubert-base-cased-sentiment",
        return_all_scores=True,
        max_length=512,
        truncation=True
    )

    return sentiment_analyzer


class RussianBusinessSentiment:
    def __init__(self):
        self.sentiment_analyzer = setup_advanced_russian_sentiment()
        self.tokenizer = AutoTokenizer.from_pretrained("blanchefort/rubert-base-cased-sentiment")

    def analyze_complaint_pattern(self, long_text):
        """Analyze complaint patterns in long Russian text"""
        # Split into sentences for more granular analysis
        import re
        sentences = re.split(r'[.!?]+', long_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        sentence_sentiments = []

        for sentence in sentences:
            try:
                result = self.sentiment_analyzer(sentence[:400])[0]  # Limit sentence length
                top_sentiment = max(result, key=lambda x: x['score'])

                sentence_sentiments.append({
                    'sentence': sentence,
                    'sentiment': top_sentiment['label'],
                    'confidence': top_sentiment['score'],
                    'all_scores': {item['label']: item['score'] for item in result}
                })
            except Exception as e:
                continue

        # Analyze sentiment progression
        sentiment_trend = self.analyze_sentiment_trend(sentence_sentiments)
        overall_complaint_score = self.calculate_overall_complaint_score(sentence_sentiments)

        return {
            'overall_complaint_score': overall_complaint_score,
            'sentiment_trend': sentiment_trend,
            'sentence_analysis': sentence_sentiments,
            'total_sentences': len(sentence_sentiments)
        }

    def analyze_sentiment_trend(self, sentence_sentiments):
        """Check if sentiment becomes more negative (escalating complaint)"""
        if len(sentence_sentiments) < 2:
            return "insufficient_data"

        negative_count = sum(1 for s in sentence_sentiments if s['sentiment'] == 'negative')
        negative_ratio = negative_count / len(sentence_sentiments)

        # Check if negative sentiment increases
        negative_intensity = []
        for i, sent in enumerate(sentence_sentiments):
            if sent['sentiment'] == 'negative':
                negative_intensity.append((i, sent['confidence']))

        if len(negative_intensity) > 1:
            # Simple trend analysis
            first_half_neg = sum(1 for s in sentence_sentiments[:len(sentence_sentiments) // 2]
                                 if s['sentiment'] == 'negative')
            second_half_neg = sum(1 for s in sentence_sentiments[len(sentence_sentiments) // 2:]
                                  if s['sentiment'] == 'negative')

            if second_half_neg > first_half_neg:
                trend = "escalating"
            else:
                trend = "stable"
        else:
            trend = "single_incident"

        return {
            'negative_ratio': negative_ratio,
            'trend': trend,
            'negative_sentences': negative_count
        }

    def calculate_overall_complaint_score(self, sentence_sentiments):
        """Calculate overall complaint probability"""
        negative_scores = []
        neutral_scores = []

        for sent in sentence_sentiments:
            for label, score in sent['all_scores'].items():
                if 'negative' in label.lower():
                    negative_scores.append(score)
                elif 'neutral' in label.lower():
                    neutral_scores.append(score)

        if not negative_scores:
            return 0.0

        avg_negative = sum(negative_scores) / len(negative_scores)
        avg_neutral = sum(neutral_scores) / len(neutral_scores) if neutral_scores else 0

        # Complaint score based on negative sentiment strength and consistency
        complaint_score = avg_negative * (len(negative_scores) / len(sentence_sentiments))

        return min(1.0, complaint_score * 1.2)  # Scale slightly

# Usage
analyzer = RussianBusinessSentiment()

file_path = 'результат_анализа_medium_long_talk.json'
text = json.load(open(file_path, 'r', encoding='utf-8'))
text = text['transcription']['text']
print(text[:300])

result = analyzer.analyze_complaint_pattern(text)
for i in result['sentence_analysis']:
    print(i['sentence'])
    print(i['sentiment'], i['confidence'])
