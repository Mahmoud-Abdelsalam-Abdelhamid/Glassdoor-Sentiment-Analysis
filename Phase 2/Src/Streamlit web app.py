import sys
sys.path.append("F:\Programming\Projects\GlassDoor sentiment analysis\Phase 2\Src")
from model import build_sentiment_model

import os
import torch
import numpy as np
import streamlit as st
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = r"F:\Programming\Projects\GlassDoor sentiment analysis\Phase 2\Models\distilbert BERT model\50K sample 90acc"
PRE_TRAINED_MODEL_NAME = 'distilbert-base-uncased'
NUM_CLASSES = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

st.set_page_config(page_title="Custom Sentiment Analysis (PyTorch)", page_icon="🧠", layout="wide")

@st.cache_resource
def load_custom_pytorch_model():
    """Load and cache the custom PyTorch model and tokenizer."""
    if not os.path.exists(MODEL_DIR):
        return None, None
    try:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model.to(DEVICE)
        model.eval()
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}.")
        return None, None

model, tokenizer = load_custom_pytorch_model()

st.title("Sentiment Analysis with a Custom PyTorch Model 🧠")
st.markdown("This app uses a fine-tuned **BERT** model with added **Dropout** for better generalization.")

if model is None or tokenizer is None:
    st.error("Model artifacts not found. Please run the training script: `python src/train.py`")
else:
    col1, col2 = st.columns([2, 1])
    with col1:
        review_text = st.text_area(
            "Enter an employee review:", height=200,
            placeholder="e.g., 'The team is wonderful, but management is disorganized.'"
        )

        if st.button("Analyze Sentiment", type="primary"):
            if not review_text.strip():
                st.warning("Please enter text to analyze.")
            else:
                with st.spinner("Analyzing..."):
                    encoding = tokenizer.encode_plus(
                        review_text, add_special_tokens=True, max_length=512,
                        return_token_type_ids=False, padding='max_length',
                        truncation=True, return_attention_mask=True, return_tensors='pt'
                    )
                    
                    input_ids = encoding['input_ids'].to(DEVICE)
                    attention_mask = encoding['attention_mask'].to(DEVICE)

                    with torch.no_grad():
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = outputs.logits  # ✅ extract logits tensor
                    
                    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                    prediction = np.argmax(probabilities)
                    confidence = probabilities[prediction]
                    
                    # Define sentiment map based on NUM_CLASSES
                    sentiment_map = {0: "Negative 😟", 1: "Positive 😊"}
                    prob_map = {"Positive": probabilities[1], "Negative": probabilities[0]}

                    sentiment = sentiment_map.get(prediction, "Unknown")
                    
                    with col2:
                        st.subheader("Analysis Result")
                        st.metric("Predicted Sentiment", sentiment)
                        st.progress(float(confidence))
                        st.write(f"**Confidence:** {confidence:.2%}")
                        st.subheader("Probability Distribution")
                        st.bar_chart(prob_map)

st.sidebar.header("About This Model")
st.sidebar.info(
    "This demo uses a custom model that adds a Dropout layer to a pre-trained Transformer. The model's trained weights (`state_dict`) are loaded after the architecture is built."
)
st.sidebar.markdown(f"- **Base Model:** `{PRE_TRAINED_MODEL_NAME}`\n- **Framework:** PyTorch\n- **Regularization:** Dropout (p=0.3)")

