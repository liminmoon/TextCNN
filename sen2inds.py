# -*- coding: utf_8 -*-

import json
import sys, io
import jieba
import random

# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030') #改变标准输出的默认编码

# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf8")

trainFile = '/Users/limin/opt/anaconda3/envs/risk/limin/textCNN_demo/data/my_traindata.json'

stopwordFile = '/Users/limin/opt/anaconda3/envs/risk/limin/textCNN_demo/data/stopword.txt'
wordLabelFile = '/Users/limin/opt/anaconda3/envs/risk/limin/textCNN_demo/data/wordLabel.txt'
trainDataVecFile = '/Users/limin/opt/anaconda3/envs/risk/limin/textCNN_demo/data/traindata_vec.txt'
labelFile = '/Users/limin/opt/anaconda3/envs/risk/limin/textCNN_demo/data/label.txt'
maxLen = 30

# 获取测试样本
# trainDataVecFile = '/Users/limin/opt/anaconda3/envs/risk/limin/textCNN_demo/data/valdata_vec.txt'
# trainFile = '/Users/limin/opt/anaconda3/envs/risk/limin/textCNN_demo/data/my_validata.json'


def read_labelFile(file):
    data = open(file, "r", encoding="utf_8").read().split("\n")
    label_w2n = {}
    label_n2w = {}
    for line in data:
        line = line.split(" ")
        name_w = line[0]
        name_n = int(line[1])
        label_w2n[name_w] = name_n
        label_n2w[name_n] = name_w

    return label_w2n, label_n2w


def read_stopword(file):
    data = open(file, "r", encoding="utf_8").read().split("\n")

    return data


def get_worddict(file):
    datas = open(file, "r", encoding="utf_8").read().split("\n")
    datas = list(filter(None, datas))
    word2ind = {}
    for line in datas:
        line = line.split(" ")
        word2ind[line[0]] = int(line[1])

    ind2word = {word2ind[w]: w for w in word2ind}
    return word2ind, ind2word


def json2txt():
    label_dict, label_n2w = read_labelFile(labelFile)
    word2ind, ind2word = get_worddict(wordLabelFile)

    traindataTxt = open(trainDataVecFile, "w")
    stoplist = read_stopword(stopwordFile)
    datas = open(trainFile, "r", encoding="utf_8").read().split("\n")
    datas = list(filter(None, datas))
    random.shuffle(datas)
    for line in datas:
        line = json.loads(line)
        title = line['title']
        cla = line['category'][0:2]
        cla_ind = label_dict[cla]
        title_seg = jieba.cut(title, cut_all=False)
        title_ind = [cla_ind]
        for w in title_seg:
            if w in stoplist:
                continue
            if w in word2ind:
                title_ind.append(word2ind[w])
            else:
                title_ind.append(word2ind["unknown"])

        length = len(title_ind)
        if length > maxLen + 1:
            title_ind = title_ind[0:(maxLen+1)]
        if length < maxLen + 1:
            title_ind.extend([0] * (maxLen - length + 1))

        for n in title_ind:
            traindataTxt.write(str(n) + ',')
        traindataTxt.write('\n')


def main():
    json2txt()


if __name__ == "__main__":
    main()
