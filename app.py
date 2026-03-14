import streamlit as st
import pickle
import pdfplumber

# Load model, vectorizer, and label encoder
model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))
le = pickle.load(open("label_encoder.pkl","rb"))

st.title("AI Resume Screening App")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf","txt"])

def extract_text(file):
    if file.type == "application/pdf":
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
        return text
    else:
        return file.read().decode("utf-8")

if uploaded_file is not None:

    resume_text = extract_text(uploaded_file)

    # Convert text to vector
    resume_vector = vectorizer.transform([resume_text])

    # Predict numeric label
    numeric_prediction = model.predict(resume_vector)

    # Convert number to original category name
    predicted_category = le.inverse_transform(numeric_prediction)[0]

    st.success(f"Predicted Category: {predicted_category}")