import streamlit as st
import joblib

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Load model & vectorizer
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

# Title and description
st.title("📰 Fake News Detector")
st.markdown("""
This app uses a Machine Learning model to classify whether a news article is **Real** or **Fake**.
Paste any article below to test:
""")

# Text input
news_input = st.text_area("📝 Enter News Article:", "")

# Check button only
check_btn = st.button("Check News ✅")

# Logic
if check_btn:
    if news_input.strip():
        with st.spinner("Analyzing..."):
            transformed_input = vectorizer.transform([news_input])
            prediction = model.predict(transformed_input)
        if prediction[0] == 1:
            st.success("✅ The news is Real!")
        else:
            st.error("⚠️ The news is Fake!")
    else:
        st.warning("⚠️ Please enter text to analyze")

# Sample expandable section
with st.expander("📌 Try Sample Articles"):
    st.write("""
    **Example Real News:**  
    The government has introduced a new policy to improve healthcare access...

    **Example Fake News:**  
    Aliens landed in New York, confirmed by anonymous officials...
    """)
