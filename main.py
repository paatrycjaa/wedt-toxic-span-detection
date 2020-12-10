import pandas as pd

from src.SemEvalData import SemEvalData
from src.JigsawData import JigsawData
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
from sklearn.metrics import roc_auc_score, accuracy_score



##set to .env
MAX_FEATURES = 300000 # maximum number of unique words that should be included in the tokenized word index
MAX_WORD_NUM = 50     # maximum number of words in each sentence
EMBED_SIZE = 50  ## same value as in dimension of glove
VAL_SPLIT = 0.2  
REG_PARAM = 1e-13
l2_reg = regularizers.l2(REG_PARAM)


nltk.download('punkt')
## load data
train_data_semeval = SemEvalData(MAX_WORD_NUM)
train_data_semeval.load_data("data/tsd_trial.csv")
train_df = train_data_semeval.preprocess()

extra_train = JigsawData(MAX_WORD_NUM)
extra_train.load_data("data/train.csv")
extra_train_df = extra_train.preprocess()



## set category 1- toxic, 0- non-toxic




paras = []
labels = []
texts = []
sent_lens = []
sent_nums = []

print( train_df)
##tokenize words
len_tr = len(train_df)
# result = train_df.append(extra_train_df, ignore_index=True, sort=False)
result = train_df
train_data = {'sentence':  result.sentences.sum(),
        'toxicity_sentence': result.toxicity_sentence.sum()
        }

train_df = pd.DataFrame (train_data, columns = ['sentence','toxicity_sentence'])

tokenizer = Tokenizer(num_words=MAX_FEATURES, oov_token=True)
tokenizer.fit_on_texts(train_df.sentence.sum())
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
word_lstm = Bidirectional(LSTM(40, return_sequences=True, kernel_regularizer=l2_reg))(word_sequences)
word_dense = TimeDistributed(Dense(70, kernel_regularizer=l2_reg))(word_lstm)
word_att = Dropout(0.2)(Attention()(word_dense))#
preds = Dense(1, activation='elu')(word_att) ##softmax, elu?
model = Model(word_input, preds)
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['acc']) ##adam
checkpoint = ModelCheckpoint('best_model.h5', verbose=-2, monitor='val_loss',save_best_only=True, mode='auto') 
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=200, batch_size=256,shuffle=True, callbacks=[checkpoint])
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
