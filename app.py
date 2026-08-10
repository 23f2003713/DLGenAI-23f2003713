import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

st.set_page_config(page_title="MCQ Answer Predictor", page_icon="🎯")

REPO_ID = "a23f2003713a/mcq-roberta-classifier"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    model = AutoModelForSequenceClassification.from_pretrained(REPO_ID)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def predict(prompt, a, b, c, d):
    prompt, a, b, c, d = [clean_text(x) for x in (prompt, a, b, c, d)]
    text = f"""
Question:
{prompt}
Option A:
{a}
Option B:
{b}
Option C:
{c}
Option D:
{d}
"""
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = outputs.logits.argmax(-1).item()
        probs = torch.softmax(outputs.logits, dim=-1)[0]
    return model.config.id2label[pred_id], probs[pred_id].item()

st.title("🎯 MCQ Answer Predictor")
st.caption("RoBERTa-base fine-tuned model — hosted on Hugging Face")

prompt = st.text_area("Question / Prompt", height=100)
col1, col2 = st.columns(2)
with col1:
    a = st.text_input("Option A")
    c = st.text_input("Option C")
with col2:
    b = st.text_input("Option B")
    d = st.text_input("Option D")

if st.button("Predict Answer", type="primary"):
    if not prompt or not a or not b or not c or not d:
        st.warning("Please fill in the question and all four options.")
    else:
        answer, confidence = predict(prompt, a, b, c, d)
        st.success(f"**Predicted Answer: {answer}**")
        st.metric("Confidence", f"{confidence:.2%}")