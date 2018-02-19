import re
import os
import keras
from keras.preprocessing.text import Tokenizer
import pickle


digit_pattern = re.compile(r'[１２３４５６７８９０.％∶]+')
letter_pattern = re.compile(r'[ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ]+')
mark_pattern = re.compile(r'[，。、（）：；＊！？－\-“”《》『』＇｀〈〉…／]+')


def replace(s):
    for digit in digit_pattern.findall(s):
        if len(digit) == 1:
            s = s.replace(digit, 'a')
        elif len(digit) == 2:
            s = s.replace(digit, 'b')
        else:
            s = s.replace(digit, 'c')

    for letter in letter_pattern.findall(s):
        if len(letter) == 1:
            s = s.replace(letter, 'z')
        else:
            s = s.replace(letter, 'y')

    for mark in mark_pattern.findall(s):
        s = s.replace(mark, 'm')

    return s


def longReplace(s):
    ss = s.split('\n')
    ss = [replace(s) for s in ss]
    return '\n'.join(ss)


tokenizer = Tokenizer(filters='\n', char_level=True)

trainFile = open('/hdd/data/Sighan-2004/train.txt')
testFile = open('/hdd/data/Sighan-2004/test.answer.txt')


def preprocess(input_file):
    lines = input_file.read()
    lines = longReplace(lines)
    lines = ['A' + _ + 'A' for _ in lines.split('\n') if len(_) > 0]
    retSeq = []
    retLabel = []
    for line in lines:
        curr_token = []
        curr_label = []
        line = line.replace('  ', 'A').replace(' ', 'A')
        line = line.replace('AAA', 'A').replace('AA', 'A').replace('AA', 'A')
        assert 'AA' not in line
        for idx, w in enumerate(line):
            if w == 'A':
                continue
            if line[idx - 1] == 'A' and line[idx + 1] == 'A':
                curr_label.append(3)
            elif line[idx - 1] == 'A' and not line[idx + 1] == 'A':
                curr_label.append(0)
            elif not line[idx - 1] == 'A' and line[idx + 1] == 'A':
                curr_label.append(2)
            else:
                curr_label.append(1)
            curr_token.append(w)

        retSeq.append(curr_token)
        retLabel.append(curr_label)
    return retSeq, retLabel


pickle.dump(preprocess(testFile), open('test.pkl', 'wb'))
pickle.dump(preprocess(trainFile), open('train.pkl', 'wb'))

