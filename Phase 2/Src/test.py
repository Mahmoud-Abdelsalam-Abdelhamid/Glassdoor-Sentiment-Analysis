import sys
sys.path.append("F:\Programming\Projects\GlassDoor sentiment analysis\Phase 2\Src")

import data_preprocessing
from model import build_sentiment_model
from train import train_epoch, eval_model

import os
import torch
import argparse
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Import the custom model class and config
from model import build_sentiment_model

MODEL_DIR = r"F:\Programming\Projects\GlassDoor sentiment analysis\Phase 2\Models\distilbert BERT model\50K sample 90acc"
PRE_TRAINED_MODEL_NAME = 'distilbert-base-uncased'
NUM_CLASSES = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def predict_sentiment(text):
    """Predicts sentiment using the fine-tuned custom PyTorch model."""
    if not text.strip():
        return {"error": "Input text is empty."}
        
    try:
        # --- MODIFICATION: Load the custom model architecture first ---
        # 1. Instantiate the model structure (with random weights)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        
        model.to(DEVICE)
        model.eval()

    except Exception as e:
        return {"error": f"Failed to load model. Have you trained it yet? Details: {e}"}

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    
    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    prediction = np.argmax(probabilities)
    confidence = probabilities[prediction]
    
    # Assuming 2 classes: 0 for Negative, 1 for Positive
    # If you are using 3 classes, update this map accordingly.
    sentiment_map = {0: "Negative", 1: "Positive"}
    if NUM_CLASSES == 3:
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        
    sentiment = sentiment_map.get(prediction, "Unknown")

    # Prepare probabilities for JSON output
    probs_output = {label: float(prob) for label, prob in zip(sentiment_map.values(), probabilities)}

    return {
        "text": text, "sentiment": sentiment, "confidence": float(confidence),
        "probabilities": probs_output
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict sentiment of text with a custom PyTorch model.")
    # parser.add_argument('--text', type=str, required=True, help='The text to analyze.')
    parser.add_argument('--text', required=False, default="There’s no work-life balance and the salary is disappointing.")

    
    args = parser.parse_args()
    result = predict_sentiment(args.text)
    
    if "error" in result:
        print(result["error"])
    else:
        print("\n--- Custom PyTorch Model Sentiment Analysis ---")
        print(f"Input Text: {result['text']}")
        print(f"Predicted Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("--- Class Probabilities ---")
        for sentiment, prob in result['probabilities'].items():
            print(f"  - {sentiment}: {prob:.2%}")
        print("------------------------------------------")

