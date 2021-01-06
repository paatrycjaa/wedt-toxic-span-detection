import joblib
from src.preprocessing import preprocess_bayes, getSpansByToxicWords, getToxicWordsBayes, clean_str
import nltk
from nltk import tokenize
from nltk.corpus import stopwords
import keras
from keras.models import load_model, Model
import numpy as np
import pickle
from src.attention_exp import getWordsByAttention, wordAttentionWeights
from src.Attention import Attention
import pickle
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer

MAX_WORD_NUM = 100 
MAX_FEATURES = 200000
TRESHOLD = 0.016
MAX_WORD_NUM_2 = 40

class Transform():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        print("")
    
    def fit(self, X, y = None):
        print("")
        return self
    
    def transform(self, X, y =None):
        X_ = vectorize(X, self.tokenizer)
        return X_

def getPredictedWordsFromSentence(sentence, threshold,c):
    #print(sentence)
    #y_pred = new_predict_working([sentence])
    #if np.argmax(y_pred) == 1 :
    explainer = LimeTextExplainer(class_names=["NoToxic","Toxic"])
    predicted_words = []
    exp = explainer.explain_instance(sentence,c.predict_proba, num_features=6, top_labels = 2)
    expWords = exp.as_list()
    maxScore = max(expWords, key = lambda i : i[1])[0]
    expWords = filter(lambda t: t[1] > threshold, expWords )
    wordsList = [i[0] for i in expWords]
    wordsList.append(maxScore) if maxScore not in wordsList else wordsList
    return wordsList

def test_sentence(text, classifierType):
    span = []
    if(classifierType =="bayes"):
        span = test_bayes(text)
    elif (classifierType == "lime"):
        span = []
    elif (classifierType == "attention"):
        span = test_attention(text)
    return span
    
def test_bayes(text):
    count_vect = joblib.load('vectorizer.jbl')
    transformer = joblib.load('transformer.jbl')
    model = joblib.load('model.jbl')
    lemmer = nltk.stem.WordNetLemmatizer()

    tokenized = tokenize.sent_tokenize(text)
    sentences = [preprocess_bayes(sentence) for sentence in tokenized]
    toxic_words = []
    for sentence in sentences:
        data = ' '.join([lemmer.lemmatize(word) for word in sentence.split()])
        x = count_vect.transform([data])
        y = model.predict(x)
        if y == 1.0:
            toxic = getToxicWordsBayes(count_vect, x, 0.5)
            toxic_words = [*toxic_words, *toxic]

    text_preprocessed = preprocess_bayes(text)        
    spans = getSpansByToxicWords(toxic_words,text_preprocessed)
    return spans

def neuralnetwork_preprocess(text):
    text_cleaned = clean_str(text)
    tokenized = tokenize.sent_tokenize(text_cleaned)
    print(tokenized)
    sentences =[]
    for i in tokenized:
        sentences.append(nltk.word_tokenize(i))
    for i, w in enumerate(sentences):
        sentences[i] = [word for word in sentences[i] if word.isalpha()]
    sentences = [x for x in sentences if x!=[]]
    data_index = np.zeros((len(sentences), MAX_WORD_NUM), dtype='int32')
    ##load tokenizer
    with open('tokenizer_nn.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    for i, sentence in enumerate(sentences):
        for k, word in enumerate(sentence):
            try:
                if k<MAX_WORD_NUM and tokenizer.word_index[word]<MAX_FEATURES:
                    data_index[i,k] = tokenizer.word_index[word]
            except:
                #print(word)
                pass
    return text_cleaned, tokenized, data_index

def vectorize(x, tokenizer):
    if not isinstance(x, list): 
        x = [x]
    data_temp = np.zeros((len(x), MAX_WORD_NUM_2), dtype='int32')
    for i, sentence in enumerate(x):
        #words = sentence.split(" ")
        for k, word in enumerate(sentence):
            try:
                if k<MAX_WORD_NUM and tokenizer.word_index[word]<MAX_FEATURES:
                    data_temp[i,k] = tokenizer.word_index[word]
            except:
                pass
    return data_temp

def preprocess_lstm(text):
        
    text = clean_str(text)
    tokenized = tokenize.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sentence) for sentence in tokenized]
    
    for i, w in enumerate(sentences):
        sentences[i] = [word for word in sentences[i] if word.isalpha()]
        
    #nltk.download('stopwords')
    #stop_words = stopwords.words('english')
    
    #for i, w in enumerate(sentences):
    #    sentences[i] = [w for w in sentences[i] if not w in stop_words]
        
    sentences = [x for x in sentences if x!=[]]
    return text, tokenized, sentences

## tring another - thae same resul as test_attention
def test_attention_2(text):
    
    with open('tokenizer_lstm.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    model = keras.models.load_model('attention_model')
    hidden_word_encoding_out = Model(inputs=model.input, outputs= model.get_layer('dense').output)
    word_context = model.get_layer('attention').get_weights()
    
    text_clean, tokenized, sentences =preprocess_lstm(text)
    vect = vectorize(sentences, tokenizer)
    toxic_words = []
    for i in range(len(vect)):
        in_data = vect[i].reshape(1,MAX_WORD_NUM)
        y = model.predict(in_data)
        Y = np.where(y > 0.5,1,0)
        #warunek na klase
        hidden_word_encodings = hidden_word_encoding_out.predict(in_data)
        ait = wordAttentionWeights(hidden_word_encodings,word_context)
        toxic = getWordsByAttention(ait,in_data, tokenized[i] ,TRESHOLD)
        toxic_words = toxic_words+toxic
    spans = getSpansByToxicWords(toxic_words, text)
    return spans

def test_lime(text):
    with open('tokenizer_nn.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    text_cleaned, tokenized, sentences = preprocess_lstm(text)
    vect = vectorize(sentences, tokenizer)
    #print(vect)
    model = keras.models.load_model("lstm_drop_jul_train.h5")
    c = make_pipeline(Transform(tokenizer),model)
    y_pred = model.predict(vect)
    toxic_words  = []
    for i in range(len(tokenized)):
        #print(vect[i])
        if np.argmax(y_pred[i]) == 1 :
            text_ = ' '.join(sentences[i])
            #print(text_)
            toxic = getPredictedWordsFromSentence(text_, TRESHOLD,c)
            toxic_words = toxic_words + toxic
    spans = getSpansByToxicWords(toxic_words, text_cleaned)
    return spans

def test_attention(text):
    text_cleaned, sentences, tokenized_data = neuralnetwork_preprocess(text)
    model = keras.models.load_model('attention_model')
    hidden_word_encoding_out = Model(inputs=model.input, outputs= model.get_layer('dense').output)
    word_context = model.get_layer('attention').get_weights()
    toxic_words = []
    for i in range(len(tokenized_data)):
        in_data = tokenized_data[i].reshape(1,MAX_WORD_NUM)
        y = model.predict(in_data)
        hidden_word_encodings = hidden_word_encoding_out.predict(in_data)
        # Compute context vector using output of dense layer
        ait = wordAttentionWeights(hidden_word_encodings,word_context)
        # print(ait)
        toxic = getWordsByAttention(ait,in_data, sentences[i],TRESHOLD)
        toxic_words = toxic_words+toxic
    spans = getSpansByToxicWords(toxic_words, text_cleaned)
    return spans

if __name__ == "__main__":
    nltk.download('wordnet')
    span = test_sentence("Another idiot!", "attention")
    span = test_sentence("To be or not be that is a question", "attention")
    print(span)
    #span = test_sentence("These freaking donkeys all need to be removed from office. I'm so sick and tired of these lifelong politicians who all seem clueless and could never run their own business.", "attention")