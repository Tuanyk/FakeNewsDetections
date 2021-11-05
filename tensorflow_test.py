import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

df = pd.read_csv('news.csv')

df.loc[df.label == 'FAKE', 'label'] = 0
df.loc[df.label == 'REAL', 'label'] = 1

labels = df.label

xtrain, xtest, ytrain, ytest = train_test_split(df['text'], labels, train_size=0.2, random_state=7)

train_sentences = xtrain.tolist()
test_sentences = xtest.tolist()

vocab_size = 5000
embedding_dim = 16
max_length = 500
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<oov>'

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(train_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type)

testing_sequences = tokenizer.texts_to_sequences(test_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, dropout=0.3, recurrent_dropout=0.3)))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

data_model = model.fit(training_padded, ytrain, epochs=50, validation_data = (testing_padded, ytest), callbacks=[cb])
