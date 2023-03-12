import os
import pickle
import uuid
import tensorflow as tf
import math
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt

from .preprocessor import preprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#================
# Load Data
model_1 = tf.keras.models.load_model("ItepeeApp/model/model1_fold_batch32_epoch20_lr0.0001.h5", compile=False)
model_2 = tf.keras.models.load_model("ItepeeApp/model/model2_fold_batch32_epoch20_lr0.0001.h5", compile=False)
model_3 = tf.keras.models.load_model("ItepeeApp/model/model3_fold_batch64_epoch20_lr0.001.h5", compile=False)
model_4 = tf.keras.models.load_model("ItepeeApp/model/model4_fold_batch32_epoch10_lr0.001.h5", compile=False)

tokenizer_path = "ItepeeApp/model/tokenizer_1.pickle"
with open(tokenizer_path, 'rb') as handle:
    tokenizer: Tokenizer = pickle.load(handle)

# Helper Function
def predict_hate_speech(text):
    clean_text = preprocess(text)
    clean_texts = [clean_text]
    seq = tokenizer.texts_to_sequences(clean_texts)
    seq = pad_sequences(seq)
    logits = model_1.predict(seq)
    prediction = logits[0][0]
    prediction = round(prediction)
    return prediction
#================

def index(request):
    context = {
        'title':'Mainpage'
    }
    return render(request, 'index.html', context=context)

@csrf_exempt
def predict(request):
    if request.method == "POST":
        raw_tweet = request.POST['raw_tweet']
        first_predict = predict_hate_speech(raw_tweet)
        # print(f"result : {first_predict}")
        if first_predict == 0:
            return redirect('/non_hate_speech')
        else:
            return redirect('/hate_speech')
    return render(request, 'predict.html')

def h_speech(request):
    return render(request, 'hate_speech.html')

def non_hs(request):
    context = {
        'title':'Non Hate Speech'
    }
    return render(request, 'non_hate_speech.html', context=context)