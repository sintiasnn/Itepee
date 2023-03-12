import re
import string
import pandas as pd
from typing import Callable

df_special_char = pd.read_csv("ItepeeApp/data/spesial_characters_HTML.csv", encoding='ISO-8859-1')
df_alay_map = pd.read_csv("ItepeeApp/data/new_kamusalay.csv", encoding='ISO-8859-1')
df_id_stopword = pd.read_csv("ItepeeApp/data/stopwordbahasa.csv", encoding='ISO-8859-1')

def remove_emoji_manual(text):
    temp2=[]
    for words in text.split(" "):
        if "\\x" not in words:
            temp2.append(words)
    return " ".join([word for word in temp2])

def remove_unnecessary_char(text):
    text = re.sub('\n',' ',text) # Remove every '\n'
    text = re.sub('RT',' ',text) # Remove every retweet symbol
    text = re.sub('USER',' ',text) # Remove every username
    text = re.sub('URL',' ',text) # Remove every username
    text = re.sub('Retweeted',' ',text) # Remove every username
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text) # Remove every URL and HT
    text = re.sub('  +', ' ', text) # Remove extra spaces
    return text

def remove_special_char(text):
    text = ' '.join(['' if word in df_special_char else word for word in text.split(' ')])
    text = re.sub(' +', ' ', text)
    text = text.strip()
    return text

def remove_number(text):
    return str(text).replace(r'\d+','')

def remove_whitespace(text):
    return str(text).replace('\n','')

def lowercase(text):
    return str(text).lower()

def remove_punctuation(text):
    punctuationfree= [i for i in text if i not in string.punctuation]
    word_punct = ''.join(punctuationfree)
    return word_punct

def normalize_alay(text):
    return ' '.join([df_alay_map[word] if word in df_alay_map else word for word in text.split(' ')])

def remove_stopword(text):
    text = ' '.join(['' if word in df_id_stopword.stopword.values else word for word in text.split(' ')])
    text = re.sub('  +', ' ', text) # Remove extra spaces
    text = text.strip()
    return text

def preprocess(text: str) -> 'str':
    preprocessor: 'list[Callable[[str], str]]' = [
        remove_emoji_manual,
        remove_unnecessary_char,
        remove_special_char,
        remove_number,
        remove_whitespace,
        lowercase,
        remove_punctuation,
        normalize_alay,
        remove_stopword
    ]
    new_text = text
    for process in preprocessor:
        new_text = process(new_text)
    return new_text