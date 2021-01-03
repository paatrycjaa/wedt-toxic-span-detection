import joblib
from src.preprocessing import preprocess_bayes, getSpansByToxicWords, getToxicWordsBayes
import nltk
from nltk import tokenize
from keras.models import load_model, Model




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
def test_lime(text):
    
    return

def test_attention(text):
    
    model = load_model('best_model_embeddings.h5')
    hidden_word_encoding_out = Model(inputs=model.input, outputs= model.get_layer('dense').output)
    word_context = model.get_layer('attention').get_weights()
    return

if __name__ == "__main__":
    nltk.download('wordnet')
    span = test_sentence("Another idiot!", "bayes")
    span = test_sentence("These freaking donkeys all need to be removed from office. I'm so sick and tired of these lifelong politicians who all seem clueless and could never run their own business.", "bayes")