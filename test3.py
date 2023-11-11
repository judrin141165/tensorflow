
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

train_sentences = [
   'It is a sunny day',
   'weather is poor',
   'i live in reading',
   # add a new sentence here
]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

print(word_index)
