import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download required NLTK data (quietly)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Load the pre-trained model and vectorizer
model = joblib.load('xgb_emotion_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Define the correct emotion mapping based on the training notebook
emotion_map = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise"
}

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) 
    tokens = word_tokenize(text)  
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Streamlit UI
st.set_page_config(page_title="Text Emotion Detector", page_icon="😊", layout="centered")

st.title("🎭 Text Emotion Detection App")
st.write("Enter a sentence below, and the model will predict the **emotion** it expresses!")

# Text input from user
user_input = st.text_area("Enter your text:", placeholder="Type something like 'I am so happy today!'")

# Predict emotion
if st.button("🔍 Predict Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Preprocess the text to match training pipeline
        cleaned_text = preprocess_text(user_input)
        
        # Vectorize the cleaned input text
        text_vectorized = vectorizer.transform([cleaned_text])
        
        # Predict emotion
        prediction = model.predict(text_vectorized)
        predicted_emotion = emotion_map.get(prediction[0], "Unknown")

        # Display result
        st.success(f"**Predicted Emotion:** {predicted_emotion.capitalize()}")

        # Optionally show probabilities (if supported by model)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(text_vectorized)[0]
            emotion_probs = {emotion_map[i]: round(prob, 3) for i, prob in enumerate(proba)}
            st.subheader("🔢 Prediction Probabilities:")
            st.json(emotion_probs)

# Footer
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and XGBoost")
