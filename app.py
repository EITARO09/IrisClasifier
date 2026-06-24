import streamlit as st
import pandas as pd
import pickle
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training model jika belum ada
if not os.path.exists("model.pkl"):

    data = pd.read_csv(
        "SMSSpamCollection",
        sep="\t",
        names=["label", "message"]
    )

    vectorizer = CountVectorizer()

    X = vectorizer.fit_transform(data["message"])
    y = data["label"]

    model = MultinomialNB()
    model.fit(X, y)

    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Spam Message Detection")

msg = st.text_area("Masukkan Pesan")

if st.button("Prediksi"):

    data = vectorizer.transform([msg])

    hasil = model.predict(data)

    if hasil[0] == "spam":
        st.error("SPAM")
    else:
        st.success("HAM (Bukan Spam)")