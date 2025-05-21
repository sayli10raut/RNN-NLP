#!/usr/bin/env python
# coding: utf-8

# 1. Classification Preprocessing

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

# Load dataset
df = pd.read_csv("classification_dataset.csv")

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])

# Padding
max_len = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_len)

# Labels
labels = df['label'].astype('category').cat.codes
y = to_categorical(labels)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# 2. Text Generation Preprocessing

# In[ ]:


import numpy as np

# Load corpus
with open("science_corpus.txt") as f:
    text = f.read().lower()

tokenizer_gen = Tokenizer()
tokenizer_gen.fit_on_texts([text])
total_words = len(tokenizer_gen.word_index) + 1

# Create input sequences
input_sequences = []
token_list = tokenizer_gen.texts_to_sequences([text])[0]

for i in range(10, len(token_list)):
    seq = token_list[i-10:i+1]
    input_sequences.append(seq)

input_sequences = np.array(input_sequences)
X_gen = input_sequences[:, :-1]
y_gen = to_categorical(input_sequences[:, -1], num_classes=total_words)


# Phase 3: Model Building

# 1. RNN Classification Model

# In[ ]:


from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense

model_clf = Sequential([
    Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=64, input_length=max_len),
    SimpleRNN(64),
    Dense(32, activation='relu'),
    Dense(y.shape[1], activation='softmax')
])

model_clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_clf.summary()


# 2. RNN Text Generation Model

# In[ ]:


model_gen = Sequential([
    Embedding(input_dim=total_words, output_dim=64, input_length=10),
    SimpleRNN(128),
    Dense(total_words, activation='softmax')
])

model_gen.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_gen.summary()


# Training and Evaluation

# 1. Train and Evaluate Classification Model

# In[ ]:


model_clf.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
loss, accuracy = model_clf.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")


# 2. Train and Demonstrate Text Generation

# In[ ]:


model_gen.fit(X_gen, y_gen, epochs=20, verbose=1)


# Text Generation Function

# In[ ]:


def generate_text(seed_text, next_words=20):
    for _ in range(next_words):
        token_list = tokenizer_gen.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=10, padding='pre')
        predicted = model_gen.predict(token_list, verbose=0)
        predicted_word = tokenizer_gen.index_word[np.argmax(predicted)]
        seed_text += ' ' + predicted_word
    return seed_text

print(generate_text("the solar system consists of the sun and"))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




