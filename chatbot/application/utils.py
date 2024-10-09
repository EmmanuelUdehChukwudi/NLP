import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import numpy as np
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

ignore_letters = ['?','!','.',',']

def clean_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]

def bag_of_words(sentence):
    vocab_words = pickle.load(open('/home/emmanuel/NLP/chatbot/models/words/words.pkl', 'rb'))
    sentence_words = clean_sentence(sentence)
    bag = [0] * len(vocab_words)
    for i, word in enumerate(vocab_words):
        if word in sentence_words:
            bag[i] = 1
    
    return np.array(bag)

def predict_class(sentence):
    classes = pickle.load(open('/home/emmanuel/NLP/chatbot/models/classes/classes.pkl','rb'))
    bag = bag_of_words(sentence)
    model = load_model('/home/emmanuel/NLP/chatbot/models/primus_model-v1.h5')
    result = model.predict(np.array([bag]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(result) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    result_list = []
    for r in results:
        result_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
        
    return result_list

def get_response(intents_list):
    intents_json = json.load(open('/home/emmanuel/NLP/chatbot/intents.json'))
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


    