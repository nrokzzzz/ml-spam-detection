import pandas as pd
from joblib import dump,load
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv('C:/Users/nagur/OneDrive/Desktop/projects/email_spams/spamdata.csv')

X = data['text']
y = data['label']

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)


modelNaive = MultinomialNB()

modelNaive.fit(X,y)

dump(vectorizer,"vectorizer.joblib")
dump(modelNaive,'modelNaive.joblib')