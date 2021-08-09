from keras.datasets import imdb
from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np

VOCAB_SIZE = 88584
MAXLEN = 250
BATCH_SIZE = 644

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE)

#la longitud siempre varia, por lo que tenemos que fijarla para eso utilizamos el MAXLEN
train_data = sequence.pad_sequences(train_data, MAXLEN)
test_data = sequence.pad_sequences(test_data, MAXLEN)

#

model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, 32),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

#
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['acc'])
history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)

results = model.evaluate(test_data, test_labels)
print(results)

word_index = imdb.get_word_index()

def encode_text(text):
    tokens = keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    return sequence.pad_sequences([tokens], MAXLEN)[0]

text = "that movie was just amazing, so amazing"
encoded = encode_text(text)
print(encoded)

def predict(text):
    encode_text = encode_text(text)
    prediction = np.zeros((1,250))
    prediction[0] = encode_text
    result = model.predict(prediction)
    print(result[0])
