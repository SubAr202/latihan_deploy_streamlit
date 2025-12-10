import joblib
import streamlit as st
model = joblib.load("model/model_logistic_regression.pkl")
tfidf = joblib.load("model/tfidf_vectorizer.pkl")

st.title("aplikasi klasifikasi komentar publik")
st.write ("aplikasi ini dibuat menggunakakn teknologi nlp dengan memanfaatkan model machine learning logistic regresion")
input = st.text_input("masukan komentar anda : ")
if st.button("submit"):
    if input.strip() == "":
        st.warning("komentar tidak boleh kosong")
    else:
        vector = tfidf.transform([input])
        prediksi = model.predict(vector)[0]

        label_map = {
            0: "negatif",
            1: "positif"
        }

        st.subheader("hasil analisis")
        st.write("**komentar : **",label_map.get(prediksi, prediksi))