from keras.preprocessing.text import Tokenizer
import pickle
import numpy as np
max_features = 5130
bi_features = 150000 #399475
tri_features = 150000#1505313
#batch_size = 50
window_size = 5
rate = 0.35
testSeq, testLabel = pickle.load(open('test.pkl', 'rb'))
trainSeq, trainLabel = pickle.load(open('train.pkl', 'rb'))
bigram_to_fit = []
trigram = []
fitSeq = trainSeq + testSeq
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
print(len(test_tri[3]), len(testSeq[3]), len(test_bi[3]))


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
    return np.array(retSeq), np.array(retLabel), np.array(retgram), np.array(rettri)


def get_feature(seq, bi, tri):
    temp = np.zeros((max_features + bi_features + tri_features))
    for ch in seq:
        if ch < max_features:
            temp[ch] = 1
        else:
            temp[max_features-1] = 1
    for bf in bi:
        if bf < bi_features:
            temp[bf + max_features] = 1
        else:
            temp[max_features + bi_features -1] =1
    if tri < tri_features:
        temp[tri + max_features + bi_features] = 1
    else:
        temp[tri_features + max_features + bi_features -1] = 1
    return temp


def evaluate(testSeq, testLabel, test_bi, test_tri):
    test_right = 0
    test_wrong = 0
    testSeq, testLabel, test_bi, test_tri = genBatch(testSeq, testLabel, test_bi, test_tri)
    print("batched")
    theta = pickle.load(open('/data/new_0.data', 'rb'))
    print("opened")
    for i, thing in enumerate(testSeq):
        feature = get_feature(thing, test_bi[i], test_tri[i])
        answer = theta @ feature
        if i%1000 == 1:
            print(i)
            print('result: right=', test_right, 'wrong=', test_wrong)
        predict_label = 0
        for i in range(4):
            if answer[i] == max(answer):
                predict_label = i
        if predict_label == testLabel[i]:
            test_right += 1
        else:
            test_wrong += 1
    print('result: right=', test_right, 'wrong=', test_wrong)

trainSeq, trainLabel, trainGram, trainTri = genBatch(trainSeq, trainLabel, bigram_to_fit, trigram)
# testSeq, testLabel = genBatch(testSeq, testLabel)

#print(len(trainSeq), len(trainSeq[0]), len(trainTri), len(trainGram), len(trainGram[0]))


theta = pickle.load(open('/data/no_0.data', 'rb'))
v = np.zeros((max_features + bi_features + tri_features))
for wtf in range(3):
    for e in range(50):
        right = 0
        wrong = 0
        for j, thing in enumerate(trainSeq):
            feature = np.zeros((max_features + bi_features + tri_features))
            if j < e * 79800:
                continue
            if j > (e + 1) * 79800:
                break
            for ch in thing:
                feature[ch] = 1
            for bi in trainGram[j]:
                if bi < bi_features:
                    feature[bi + max_features] = 1
                else:
                    feature[max_features + bi_features - 1] = 1
            if trainTri[j] < tri_features:
                feature[trainTri[j] + max_features + bi_features] = 1
            else:
                feature[max_features + bi_features + tri_features - 1] = 1
            answer = theta @ feature
            predict_label = 0
            for i in range(4):
                if answer[i] == max(answer):
                    predict_label = i

            # print(predict_label)

            if predict_label == trainLabel[j]:
                right = right + 1
            else:
                # print('wrong',predict_label,trainLabel[j])
                wrong = wrong + 1
                # theta[trainLabel[j]] = theta[trainLabel[j]] + feature
                for ch in thing:
                    theta[trainLabel[j]][ch] += rate
                    theta[predict_label][ch] -= rate
                for bi in trainGram[j]:
                    if bi < bi_features:
                        theta[trainLabel[j]][bi + max_features] += rate
                        theta[predict_label][bi + max_features] -= rate
                    else:
                        theta[trainLabel[j]][bi_features + max_features - 1] += rate
                        theta[predict_label][bi_features + max_features - 1] -= rate

                if trainTri[j] < tri_features:
                    theta[trainLabel[j]][max_features + bi_features + trainTri[j]] += rate
                    theta[predict_label][max_features + bi_features + trainTri[j]] -= rate
                else:
                    theta[trainLabel[j]][max_features + bi_features + tri_features - 1] += rate
                    theta[predict_label][max_features + bi_features + tri_features - 1] -= rate


                    # v = v + theta
                    # theta = v / 7980

        print(right, wrong, right / 79800, e)
    file = '/data/new_' + str(wtf+1) + '.data'
    fs = open(file, 'wb')

    pickle.dump(theta, fs)
    fs.close()

evaluate(testSeq, testLabel, test_bi, test_tri)
#evaluate(trainSeq,trainLabel,bigram_to_fit,trigram)





