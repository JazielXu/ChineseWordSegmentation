from keras.models import load_model
from keras.utils import plot_model
from keras.optimizers import SGD
from keras.preprocessing.text import Tokenizer
from keras.layers import LSTM, Dense, Dropout, Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import random
window_size = 5
testSeq, testLabel = pickle.load(open('test.pkl', 'rb'))
trainSeq, trainLabel = pickle.load(open('train.pkl', 'rb'))
fitSeq = trainSeq + testSeq
bigram_to_fit = []
trigram = []
for line in fitSeq:
    tem = ''
    temp = ''
    for i, charactor in enumerate(line):
        if i == 0 and i != len(line)-1:
            temp = temp + 's' + charactor + line[i+1] + ' '
        elif i != len(line)-1:
            temp = temp + line[i-1] + charactor + line[i+1] + ' '
        if i != len(line)-1:
            tem = tem + charactor + line[i+1] + ' '

    trigram.append(temp)
    bigram_to_fit.append(tem)

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(fitSeq)
bikenizer = Tokenizer(char_level=False)
bikenizer.fit_on_texts(bigram_to_fit)
trikenizer = Tokenizer(char_level=False)
trikenizer.fit_on_texts(trigram)
bigram_to_fit = []
trigram = []
for line in trainSeq:
    tem = ''
    temp = ''
    for i, charactor in enumerate(line):
        if i == 0 and i != len(line)-1:
            temp = temp + 's' + charactor + line[i+1] + ' '
        elif i != len(line)-1:
            temp = temp + line[i-1] + charactor + line[i+1] + ' '
        if i != len(line)-1:
            tem = tem + charactor + line[i+1] + ' '

    trigram.append(temp)
    bigram_to_fit.append(tem)

test_bi = []
test_tri = []
for line in testSeq:
    tem = ''
    temp = ''
    for i, charactor in enumerate(line):
        if i == 0 and i != len(line) - 1:
            temp = temp + 's' + charactor + line[i + 1] + ' '
        elif i != len(line) - 1:
            temp = temp + line[i - 1] + charactor + line[i + 1] + ' '
        if i != len(line) - 1:
            tem = tem + charactor + line[i + 1] + ' '

    test_tri.append(temp)
    test_bi.append(tem)


trainSeq = tokenizer.texts_to_sequences(trainSeq)
testSeq = tokenizer.texts_to_sequences(testSeq)



bigram_to_fit = bikenizer.texts_to_sequences(bigram_to_fit)
test_bi = bikenizer.texts_to_sequences(test_bi)



trigram=trikenizer.texts_to_sequences(trigram)
test_tri = trikenizer.texts_to_sequences(test_tri)

def genBatch(seqs, label, gram, tri):
    assert window_size % 2 == 1
    pad_len = window_size // 2
    retSeq, retLabel, retgram, rettri = [], [], [], []
    print('getting batch')
    for j,seq in enumerate(seqs):
        seq = [1] * pad_len + seq + [1] * pad_len
        gram[j] = [1] + gram[j] + [1]
        tri[j] = tri[j] + [1]
        for idx in range(pad_len, len(seq)-pad_len):
            retSeq.append(seq[idx-pad_len: idx+pad_len+1])
            retLabel.append(label[j][idx-pad_len])
            retgram.append(gram[j][idx-pad_len:idx+2-pad_len])
            rettri.append(tri[j][idx-pad_len])
    return retSeq, np.array(retLabel), retgram, rettri


trainSeq, trainLabel, trainGram, trainTri = genBatch(trainSeq, trainLabel, bigram_to_fit, trigram)
trainx = []
#train_x = []
train_y = keras.utils.to_categorical(trainLabel)
for i, thing in enumerate(trainSeq):
    temp = thing + trainGram[i]
    temp.append(trainTri[i])
    trainx.append(thing)

train_x = np.array(trainx)



eval_ratio = 0.15
x_train = train_x[int(len(train_x)*eval_ratio):]
y_train = train_y[int(len(train_x)*eval_ratio):]
x_test = train_x[:int(len(train_x)*eval_ratio)]
y_test = train_y[:int(len(train_x)*eval_ratio)]

model = Sequential()
#model.add(Embedding(5131, 36, input_length=5))
model.add(Dense(32, activation='relu',input_dim=len(train_x[0])))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

plot_model(model, to_file='/data/model.png',show_shapes=True)
print("ploted")

model.fit(x_train, y_train, epochs=3, batch_size=50)
score, acc = model.evaluate(x_test, y_test,
                            batch_size= 32)
model.save('/data/my_model.h5')
print("saved")