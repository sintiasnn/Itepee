import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import sys
import pickle
import tensorflow as tf
import tf_keras
import numpy as np

# Make keras import path work for legacy pickled tokenizer
sys.modules['keras'] = tf_keras

from tf_keras.preprocessing.text import Tokenizer
from tf_keras.preprocessing.sequence import pad_sequences

import time

from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
from .models import TweetModel

from .preprocessor import preprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#================
# Load Model & Tokenizer
model_1 = tf.keras.models.load_model("ItepeeApp/model/model1_fold_batch32_epoch20_lr0.0001.h5", compile=False)
model_2 = tf.keras.models.load_model("ItepeeApp/model/model2_fold_batch32_epoch20_lr0.0001.h5", compile=False)
model_3 = tf.keras.models.load_model("ItepeeApp/model/model3_fold_batch64_epoch20_lr0.001.h5", compile=False)
model_4 = tf.keras.models.load_model("ItepeeApp/model/model4_fold_batch32_epoch10_lr0.001.h5", compile=False)

tokenizer_path = "ItepeeApp/model/tokenizer_1.pickle"
with open(tokenizer_path, 'rb') as handle:
    tokenizer: Tokenizer = pickle.load(handle)
#================

# Helper Function
def predict_hate_speech(text):
    clean_text = preprocess(text)
    clean_texts = [clean_text]
    seq = tokenizer.texts_to_sequences(clean_texts)
    seq = pad_sequences(seq)
    logits = model_1.predict(seq, verbose=0)
    prediction = logits[0][0]
    prediction = round(prediction)
    return prediction

def predict_full_hate_speech(txt):
    clean_text = preprocess(txt)
    clean_texts = [clean_text]
    seq = tokenizer.texts_to_sequences(clean_texts)
    seq = pad_sequences(seq)
    prediction2 = model_2.predict(seq, verbose=0)[0]
    prediction3 = model_3.predict(seq, verbose=0)[0]
    prediction4 = model_4.predict(seq, verbose=0)[0]
    return {
        "target_hs_individu": prediction2[0],
        "target_hs_group": prediction2[1],
        "type_hs_religion": prediction3[0],
        "type_hs_ras": prediction3[1],
        "type_hs_physical": prediction3[2],
        "type_hs_gender": prediction3[3],
        "type_hs_other": prediction3[4],
        "level_hs_weak": prediction4[0],
        "level_hs_moderate": prediction4[1],
        "level_hs_strong": prediction4[2],
    }

def get_prediction_summary(result):
    target = "Individu" if result["target_hs_individu"] > result["target_hs_group"] else "Kelompok"

    type_labels = ["Agama", "Ras", "Fisik", "Gender", "Lainnya"]
    type_values = [
        result["type_hs_religion"], result["type_hs_ras"],
        result["type_hs_physical"], result["type_hs_gender"],
        result["type_hs_other"]
    ]
    hs_type = type_labels[np.argmax(type_values)]

    level_labels = ["Lemah", "Sedang", "Kuat"]
    level_values = [
        result["level_hs_weak"], result["level_hs_moderate"],
        result["level_hs_strong"]
    ]
    hs_level = level_labels[np.argmax(level_values)]

    return target, hs_type, hs_level
#================

def index(request):
    context = {
        'title': 'Mainpage'
    }
    return render(request, 'index.html', context=context)

def predict(request):
    if request.method == "POST":
        raw_tweet = request.POST.get('raw_tweet', '').strip()
        if not raw_tweet:
            messages.error(request, "Tolong masukkan tweet terlebih dahulu.")
            return render(request, 'predict.html')

        # Rate limit: max 10 requests per minute per session
        now = time.time()
        last_time = request.session.get('last_predict_time', 0)
        predict_count = request.session.get('predict_count', 0)
        if now - last_time < 60:
            request.session['predict_count'] = predict_count + 1
        else:
            request.session['predict_count'] = 1
        request.session['last_predict_time'] = now

        if request.session.get('predict_count', 0) > 10:
            messages.error(request, "Terlalu banyak permintaan. Silakan coba lagi nanti.")
            return render(request, 'predict.html')

        try:
            first_predict = predict_hate_speech(raw_tweet)
            if first_predict == 0:
                return redirect('/non_hate_speech')
            else:
                prediction_result = predict_full_hate_speech(raw_tweet)
                tweetMod = TweetModel(
                    tweet=raw_tweet,
                    is_hate_speech=float(first_predict),
                    **prediction_result
                )
                tweetMod.save()
                return redirect('/hate_speech')
        except Exception as e:
            messages.error(request, f"Terjadi kesalahan saat prediksi. Coba lagi.")
            return render(request, 'predict.html')
    return render(request, 'predict.html')

def h_speech(request):
    result = TweetModel.objects.all().order_by('-id').first()
    if not result:
        return redirect('/predict')
    target, hs_type, hs_level = get_prediction_summary({
        "target_hs_individu": result.target_hs_individu,
        "target_hs_group": result.target_hs_group,
        "type_hs_religion": result.type_hs_religion,
        "type_hs_ras": result.type_hs_ras,
        "type_hs_physical": result.type_hs_physical,
        "type_hs_gender": result.type_hs_gender,
        "type_hs_other": result.type_hs_other,
        "level_hs_weak": result.level_hs_weak,
        "level_hs_moderate": result.level_hs_moderate,
        "level_hs_strong": result.level_hs_strong,
    })
    return render(request, 'hate_speech.html', context={
        'tweet': result.tweet,
        'target': target,
        'type': hs_type,
        'level': hs_level,
    })

def non_hs(request):
    return render(request, 'non_hate_speech.html', context={
        'title': 'Non Hate Speech'
    })

def detail(request):
    result_model = TweetModel.objects.all().order_by('-id').first()
    if not result_model:
        return redirect('/predict')

    target, hs_type, hs_level = get_prediction_summary({
        "target_hs_individu": result_model.target_hs_individu,
        "target_hs_group": result_model.target_hs_group,
        "type_hs_religion": result_model.type_hs_religion,
        "type_hs_ras": result_model.type_hs_ras,
        "type_hs_physical": result_model.type_hs_physical,
        "type_hs_gender": result_model.type_hs_gender,
        "type_hs_other": result_model.type_hs_other,
        "level_hs_weak": result_model.level_hs_weak,
        "level_hs_moderate": result_model.level_hs_moderate,
        "level_hs_strong": result_model.level_hs_strong,
    })

    return render(request, 'detail.html', context={
        "tweet": result_model.tweet,
        "target": target,
        "type": hs_type,
        "level": hs_level,
        "target_hs_individu": result_model.target_hs_individu,
        "target_hs_group": result_model.target_hs_group,
        "types": [
            ("Agama", result_model.type_hs_religion),
            ("Ras", result_model.type_hs_ras),
            ("Fisik", result_model.type_hs_physical),
            ("Gender", result_model.type_hs_gender),
            ("Lainnya", result_model.type_hs_other),
        ],
        "levels": [
            ("Lemah", result_model.level_hs_weak),
            ("Sedang", result_model.level_hs_moderate),
            ("Kuat", result_model.level_hs_strong),
        ],
    })