from flask import Flask, render_template, url_for, request
from flask_bootstrap import Bootstrap
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.externals import joblib
import joblib
import keras
#from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import pickle
import re
import nltk
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))
app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def home():
	return render_template("home.html")

@app.route('/predict', methods = ['POST'])
def predict():
	#return render_template("result.html")
	

	df= pd.read_csv("dataset\data.csv")

	df_data = df[["class", "comments"]]
	df_x = df_data["comments"]
	df_y = df_data["class"]

	corpus = df_x.values.astype('U')
	cv = CountVectorizer()
	X = cv.fit_transform(corpus)

	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.3, random_state=42)

	from sklearn.linear_model import LogisticRegression
	lclf = LogisticRegression()
	lclf.fit(X_train, y_train)
	lscore=lclf.score(X_test, y_test)
	from sklearn.ensemble import RandomForestClassifier
	Rclf = RandomForestClassifier(n_estimators = 100)
	Rclf.fit(X_train, y_train)
	rscore=Rclf.score(X_test, y_test)
	if request.method == 'POST':
		comment = request.form['comment']
		data = [comment]
		nlp_prediction=deeplearning(comment)
		vect = cv.transform(data).toarray()
		l_prediction = lclf.predict(vect)
		print(l_prediction)
		R_prediction=Rclf.predict(vect)
		print(R_prediction)
	return render_template('home.html', name = data, l_prediction = l_prediction,R_prediction = R_prediction,nlp_prediction=nlp_prediction, user_comment = comment)
def clean_text(text):
    print(text)
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    print(text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
def deeplearning(test):
	load_model=keras.models.load_model("./hate&abusive_model.h5")
	with open('tokenizer.pickle', 'rb') as handle:
		load_tokenizer = pickle.load(handle)
	test=[clean_text(test)]
	print(test)
	seq = load_tokenizer.texts_to_sequences(test)
	padded = sequence.pad_sequences(seq, maxlen=300)
	print(seq)
	pred = load_model.predict(padded)
	print("pred", pred)
	if pred<0.5:
		return 0
	else:
		return 1
if __name__ == '__main__':
	app.run(debug = True)