import numpy as np
import pandas as pd
import nltk, stop_words, re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('news.csv')

df.loc[df.label == 'FAKE', 'label'] = 0
df.loc[df.label == 'REAL', 'label'] = 1

def transformation(text):
    text = re.sub('[^A-Za-z]', ' ', text)
    return ' '.join([nltk.stem.PorterStemmer().stem(word) for word in text.lower().split() if word not in stop_words.get_stop_words('english')])

text = df['text'].apply(transformation)
labels = df['label']

vectorize = CountVectorizer(max_features=200, ngram_range=(1,3))
x = vectorize.fit_transform(text).toarray()
y = np.array(labels)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=34)
print('\nTotal number of samples in the Train Dataset :',xtrain.shape[0])
print('Total number of samples in the Test Dataset :',xtest.shape[0])

bnb = BernoulliNB().fit(xtrain,ytrain)
print('\nAccuracy score :',bnb.score(xtest,ytest))

ypred = bnb.predict(xtest)
print('Confusion Metrics : \n\n',confusion_matrix(ytest,ypred),'\n\n')
print('Classification Report :\n\n',classification_report(ytest,ypred))

def predict(text):
    text = transformation(text)
    text = vectorize.transform([text])
    if bnb.predict(text)[0] == 1:
        print('This is fake news.')
    else:
        print('This is real news.')