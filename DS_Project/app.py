import streamlit as st
import joblib

model = joblib.load("model/sentiment_model.pkl")
tfidf = joblib.load("model/tfidf.pkl")

st.title("Sentiment Analysis for User Feedback")

text = st.text_area("Enter Feedback")

if st.button("Predict"):

    if text.strip() != "":

        vec = tfidf.transform([text])

        prediction = model.predict(vec)[0]

        probs = model.predict_proba(vec)[0]

        if prediction == "Positive":
            st.success(f"Sentiment: {prediction}")

        elif prediction == "Negative":
            st.error(f"Sentiment: {prediction}")

        else:
            st.info(f"Sentiment: {prediction}")

        st.subheader("Confidence Scores")

        labels = model.classes_

        for label, prob in zip(labels, probs):
            st.write(f"{label}: {prob:.2%}")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        ax.bar(labels, probs)

        ax.set_ylabel("Confidence")
        ax.set_title("Sentiment Confidence Scores")

        st.pyplot(fig)