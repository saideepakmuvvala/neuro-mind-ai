# predict_sentiment.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

# Load pretrained BERT model + tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # Small, fast version fine-tuned on SST2
classifier = pipeline("sentiment-analysis", model=model_name)

# Input text
text = "I really love this AI project!"

# Run prediction
result = classifier(text)

# Show result
print(f"Text: {text}")
print(f"Prediction: {result[0]['label']} (Confidence: {round(result[0]['score'], 3)})")
