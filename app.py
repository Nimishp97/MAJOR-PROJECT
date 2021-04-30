import streamlit as st
import pandas as pd
import numpy as np
import re
import string


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

df = pd.read_csv("/content/gdrive/MyDrive/Kaggle/data.csv",usecols=['review','rating'])
df.columns = ['reviews','ratings']

def preprocessing(x):
  x = x.lower()
  x = re.sub(r"http\S+|www\S+", "",x,flags=re.MULTILINE)
  x = x.translate(str.maketrans("", "", string.punctuation))

  x_token = word_tokenize(x)
  filtered_words = [word for word in x_token if word not in stop_words]
  return " ".join(filtered_words)

def remove_numbers(text):
    result = re.sub(r'\d+', ' ', text)
    return result

def vectorize(train_fit):
  vector = TfidfVectorizer(sublinear_tf=True)
  vector.fit_transform(train_fit.split('\n'))
  vector.fit([train_fit])
  return vector

df.reviews = df.reviews.apply(preprocessing)
df.reviews = df.reviews.apply(remove_numbers)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,5), analyzer='char',lowercase=True)
X = tfidf.fit_transform(df.reviews)
y = df['ratings']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.35, random_state = 0)

clf = LinearSVC(C=40)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


st.title("Rating Device \n",)
st.image('5-Star-Rating-PNG-Free-Download.png', caption=None, width=200, use_column_width=None, clamp=False, channels='RGB', output_format='auto')


user_input = st.text_input("Input Review : ")
k = preprocessing(user_input)
l = tfidf.transform([k])
t = clf.predict(l)
t = int(t)
st.write(f'Rating : ',t,'⭐')

st.text('input text eg. your product is very good,worst product ever etc.')