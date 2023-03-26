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
from .models import TweetModel

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

class FullHateSpeechPrediction:
    def __init__(self, pred2, pred3, pred4) -> None:
        self.prediction_model_2_individual = pred2[0]
        self.prediction_model_2_group = pred2[1]
        self.prediction_model_3_religion = pred3[0]
        self.prediction_model_3_ras = pred3[1]
        self.prediction_model_3_physical = pred3[2]
        self.prediction_model_3_gender = pred3[3]
        self.prediction_model_3_others = pred3[4]
        self.prediction_model_4_weak = pred4[0]
        self.prediction_model_4_moderate = pred4[1]
        self.prediction_model_4_strong = pred4[2]

def predict_full_hate_speech(txt: 'str') -> 'FullHateSpeechPrediction' :
    """
    Generating FullHateSpeechPrediction
    """
    clean_text = preprocess(txt)
    clean_texts = [clean_text]
    seq = tokenizer.texts_to_sequences(clean_texts)
    seq = pad_sequences(seq)
    prediction2 = model_2.predict(seq)[0]
    prediction3 = model_3.predict(seq)[0]
    prediction4 = model_4.predict(seq)[0]
    return FullHateSpeechPrediction(prediction2, prediction3, prediction4)
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
        if first_predict == 0:
            return redirect('/non_hate_speech')
        else:
            prediction_full = predict_full_hate_speech(raw_tweet)
            tweetMod = TweetModel(
                tweet = request.POST['raw_tweet'],
                is_hate_speech = first_predict,
                target_hs_individu=prediction_full.prediction_model_2_individual,
                target_hs_group=prediction_full.prediction_model_2_group,
                type_hs_religion=prediction_full.prediction_model_3_religion,
                type_hs_ras=prediction_full.prediction_model_3_ras,
                type_hs_physical=prediction_full.prediction_model_3_physical,
                type_hs_gender=prediction_full.prediction_model_3_gender,
                type_hs_other=prediction_full.prediction_model_3_others,
                level_hs_weak=prediction_full.prediction_model_4_weak,
                level_hs_moderate=prediction_full.prediction_model_4_moderate,
                level_hs_strong=prediction_full.prediction_model_4_strong
            )
            tweetMod.save()
            return redirect('/hate_speech')
    return render(request, 'predict.html')

def h_speech(request):
    return render(request, 'hate_speech.html')
    
def non_hs(request):
    context = {
        'title':'Non Hate Speech'
    }
    return render(request, 'non_hate_speech.html', context=context)

@csrf_exempt
def detail(request):
    result_model = TweetModel.objects.all().order_by('-id').first()
    context = {
        'Title':'Detail',
        'id':result_model.id,
        'tweet':result_model.tweet,
        'target_hs_individu':result_model.target_hs_individu,
        'target_hs_group':result_model.target_hs_group,
        'type_hs_religion':result_model.type_hs_religion,
        'type_hs_ras':result_model.type_hs_ras,
        'type_hs_physical':result_model.type_hs_physical,
        'type_hs_gender':result_model.type_hs_gender,
        'type_hs_other':result_model.type_hs_other,
        'level_hs_weak':result_model.level_hs_weak,
        'level_hs_moderate':result_model.level_hs_moderate,
        'level_hs_strong':result_model.level_hs_strong,
        
    }

    if context.get("target_hs_individu") > context.get("target_hs_group"):
        target_final = 'Individu'
    else:
        target_final = 'Kelompok'

    hs_types = [
        result_model.type_hs_religion,
        result_model.type_hs_ras,
        result_model.type_hs_physical,
        result_model.type_hs_gender,
        result_model.type_hs_other,
    ]
    max_val_hs_type = max(hs_types)
    hs_type = "Tidak diketahui"
    if max_val_hs_type == hs_types[0]:
        hs_type = "Agama"
    elif max_val_hs_type == hs_types[1]:
        hs_type = "Ras"
    elif max_val_hs_type == hs_types[2]:
        hs_type = "Fisik"
    elif max_val_hs_type == hs_types[3]:
        hs_type = "Gender"
    elif max_val_hs_type == hs_types[4]:
        hs_type = "Lainnya"
        
    hs_levels = [
        result_model.level_hs_weak,
        result_model.level_hs_moderate,
        result_model.level_hs_strong,
    ]
    max_val_hs_levels = max(hs_levels)
    hs_level = "Tidak diketahui"
    if max_val_hs_levels == hs_levels[0]:
        hs_level = "Lemah"
    elif max_val_hs_levels == hs_levels[1]:
        hs_level = "Sedang"
    elif max_val_hs_levels == hs_levels[2]:
        hs_level = "Kuat"
    
    return render(request, 'detail.html', context={
        "tweet": context['tweet'],
        "target" : target_final,
        "type": hs_type,
        "level": hs_level,
        "target_hs_individu":context['target_hs_individu'],
        "target_hs_group":context['target_hs_group'],
        "type_hs_religion":context['type_hs_religion'],
        "type_hs_ras":context['type_hs_ras'],
        "type_hs_physical":context['type_hs_physical'],
        "type_hs_gender":context['type_hs_gender'],
        "type_hs_other":context['type_hs_other'],
        "level_hs_weak":context['level_hs_weak'],
        "level_hs_moderate":context['level_hs_moderate'],
        "level_hs_strong":context['level_hs_strong'],
    })