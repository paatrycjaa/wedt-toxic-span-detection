import pandas as pd
from src.WordsExtraction import WordsExtraction
from ast import literal_eval
from nltk import tokenize
import nltk
from keras.preprocessing.text import Tokenizer,  text_to_word_sequence
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding, Input, Dense, LSTM, GRU, Bidirectional, TimeDistributed, Dropout
from keras import backend as K
from keras import optimizers
from keras.models import Model
from src.Attention import Attention
import re
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score


##set to .env
MAX_FEATURES = 300000 # maximum number of unique words that should be included in the tokenized word index
MAX_WORD_NUM = 50     # maximum number of words in each sentence
EMBED_SIZE = 50 
VAL_SPLIT = 0.2  
REG_PARAM = 1e-13
l2_reg = regularizers.l2(REG_PARAM)


nltk.download('punkt')
## load data
train = pd.read_csv("data/tsd_trial.csv")
train["spans"] = train.spans.apply(literal_eval)


## set category 1- toxic, 0- non-toxic
train["toxicity"] = ( train.spans.map(len) > 0 ).astype(int)
#train = train.iloc[0:60000]
## extract toxic words
words_extractor = WordsExtraction()
train['toxic_words'] = train.apply(lambda row: words_extractor.extractToxicWordIndexUsingSpans(row), axis=1)

def clean_str(string):
    """
    string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)    
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)
    string = re.sub(r"\\n","", string)    
    return string.strip().lower()

def extract_toxity(toxic_words, sentences):
    toxicity = []
    for sentence in sentences:
        if any(word in sentence for word in toxic_words):
            toxicity.append(np.full( MAX_WORD_NUM, 1.0, dtype='float32'))
        else:
            toxicity.append(np.full(MAX_WORD_NUM, 0.0, dtype='float32'))
    return toxicity

paras = []
labels = []
texts = []
sent_lens = []
sent_nums = []
## clean text
train['text'] = train.apply(lambda row: clean_str(row.text), axis=1)
## clean toxic words
train['toxic_words'] = train.apply(lambda row: [clean_str(word) for word in row.toxic_words], axis =1 )
## extract senteces
train['sentences'] = train.apply(lambda row: tokenize.sent_tokenize(row.text), axis=1)

## toxity per sentence
train['toxicity_sentence'] = train.apply(lambda row: extract_toxity(row.toxic_words, row.sentences), axis = 1)

first = train.iloc[1]
train_data = {'sentence':  train.sentences.sum(),
        'toxicity_sentence': train.toxicity_sentence.sum()
        }

train_df = pd.DataFrame (train_data, columns = ['sentence','toxicity_sentence'])


##tokenize words
tokenizer = Tokenizer(num_words=MAX_FEATURES, oov_token=True)
tokenizer.fit_on_texts(train.sentences.sum())
word_index = tokenizer.word_index
word_counts = tokenizer.word_counts
###
#print('word_index',word_index)

##get glove embeddings
embeddings_index = {}
f = open(os.path.join(os.getcwd(), 'data/glove.twitter.27B.50d.txt'), encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

min_wordCount = 2
absent_words = 0

embedding_matrix = np.zeros((len(word_index) + 1, EMBED_SIZE))

for word, i in word_index.items():
    
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
            # if word_counts[word] > min_wordCount:
            embedding_matrix[i] = embedding_vector
        else:
            absent_words += 1
print('Total absent words are', absent_words, 'which is', "%0.2f" % (absent_words * 100 / len(word_index)),
      '% of total words')


### tokenize data- to rewrite

data = np.zeros((len(train_df), MAX_WORD_NUM), dtype='int32')
for i, sentence in enumerate(train_df.sentence):
    for k, word in enumerate(sentence):
        try:
            if k<MAX_WORD_NUM and tokenizer.word_index[word]<MAX_FEATURES:
                data[i,k] = tokenizer.word_index[word]
        except:
            #print(word)
            pass
## split to test and train
#print(data)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
##IMPORTANT
data = data.astype(np.float32)
labels = train_df.toxicity_sentence.iloc[indices]
# labels = labels.astype(np.float32)
nb_validation_samples = int(VAL_SPLIT * data.shape[0])
x_train = data[:-nb_validation_samples]
y_train = np.vstack(labels[:-nb_validation_samples])
#y_train = np.array([classes(xi) for xi in y_train])

x_val = data[-nb_validation_samples:]
y_val = np.vstack(labels[-nb_validation_samples:])
#y_val = classes(y_val)
print(x_train, y_train.shape)

embedding_layer = Embedding(len(word_index)+1 ,EMBED_SIZE,weights=[embedding_matrix], input_length=MAX_WORD_NUM, trainable= True)
word_input = Input(shape=MAX_WORD_NUM, dtype='float32')
word_sequences = embedding_layer(word_input)
word_lstm = Bidirectional(LSTM(150, return_sequences=True, kernel_regularizer=l2_reg))(word_sequences)
word_dense = TimeDistributed(Dense(100, kernel_regularizer=l2_reg))(word_lstm)
word_att = Dropout(0.5)(Attention()(word_dense))#-to finish
preds = Dense(1, activation='relu')(word_att)
model = Model(word_input, preds)
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['acc'])
checkpoint = ModelCheckpoint('best_model.h5', verbose=-2, monitor='val_loss',save_best_only=True, mode='auto') 
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=200, batch_size=512,shuffle=True, callbacks=[checkpoint])

print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save('lstm.h5')


print('Found %s word vectors.' % len(embeddings_index))
