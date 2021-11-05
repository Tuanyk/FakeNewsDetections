import pickle
import pandas as pd


model = pickle.load(open('model', 'rb'))
vector = pickle.load(open('vector', 'rb'))

df = pd.read_csv('news.csv')

test = df.loc[40:41, 'text']
label = df.loc[40:41, 'label']
tfidf_test = vector.transform(test)
print(model.predict(tfidf_test))
print(label)