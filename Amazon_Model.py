import pandas as pd
import nltk
nltk.download('punkt')
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import streamlit as st
from PIL import Image
df=pd.read_csv("realistic_amazon_products.csv")
df.drop("id",axis=1,inplace=True)
from nltk.stem.snowball import SnowballStemmer
stemmer=SnowballStemmer("english")
def tokenize_stem(text):
    tokens=nltk.word_tokenize(text.lower())
    stemmed=[stemmer.stem(w) for w in tokens]
    return" ".join(stemmed)

df["stemmed_tokens"]=df.apply(lambda row: tokenize_stem(row["title"] + " " + row["description"]), axis=1)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import nltk
from sklearn.feature_extraction.text import CountVectorizer
tfidf_vectorizer=TfidfVectorizer(tokenizer=tokenize_stem)
def cosine_sim(text1, text2):
    text1_concatenated=''.join(text1)
    text2_concatenated=''.join(text2)
    tfidf_matrix=tfidf_vectorizer.fit_transform([text1_concatenated,text2_concatenated])
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return nltk([vectors[0]], [vectors[1]])[0][0]
def search_product(query):
    stemmed_query = tokenize_stem(query)
    df['similarity'] = df['stemmed_tokens'].apply(lambda x: cosine_sim(stemmed_query, x))
    res = df.sort_values(by=['similarity'], ascending=False).head(10)[['title', 'description', 'category']]
    return res

# Web App
try:
    img = Image.open('amazon_logo.png')
    st.image(img, width=600)
except FileNotFoundError:
    st.warning("Logo image not found.")
st.title("Search Engine and Product Recommendation")
query = st.text_input("Enter product name")
submit = st.button("Search")

if submit:
    res = search_product(query)
    st.write(res)





