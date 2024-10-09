import random
import os
import numpy as np
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Dense
from tensorflow.keras.optimizers import SGD
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

lemmatizer = WordNetLemmatizer()

intents = json.load(open('/home/emmanuel/NLP/chatbot/intents.json'))

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Extract intents, tokenize them, and map them to corresponding tags
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words and remove duplicates
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('/home/emmanuel/NLP/chatbot/models/words/words.pkl', 'wb'))
pickle.dump(classes, open('/home/emmanuel/NLP/chatbot/models/classes/classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Ensure train_x and train_y have the correct shapes
train_x = np.array(train_x)

if train_x.shape[1] != len(words):
    train_x = train_x.reshape(-1, len(words))

train_y = np.array(train_y)

print("Shape of train_x:", train_x.shape)
print("Shape of train_y:", train_y.shape)

# Model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Training
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

model.save('/home/emmanuel/NLP/chatbot/models/primus_model-v1.h5')
print("Done Training")
