import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# read the data from a word file
data = pd.read_csv("texts.csv")

# create a bag of words representation of the text
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(data['text'])

# convert labels to binary form
y = np.where(data['label'] == 'positive', 1, 0)

# train a logistic regression model
model = LogisticRegression()
model.fit(x, y)

# predict the class of a new text
new_text = ["this is a positive review"]
new_text_vectorized = vectorizer.transform(new_text)
prediction = model.predict(new_text_vectorized)
print("Class predicted:", prediction)
