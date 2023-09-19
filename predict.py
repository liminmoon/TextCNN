# -*- coding: utf-8 -*-
import torch
import os
import torch.nn as nn
import numpy as np
import time
import math
from collections import defaultdict
from model import textCNN
import sen2inds
import logging
import yaml
from word_util import stand_sentence
# from im_rule_correction import chinese_calibration


def get_yaml(f):
    with open(f) as stream:
        yaml_data = yaml.safe_load(stream)
    return yaml_data


def read_stopword(file):
    data = open(file, "r", encoding="utf_8").read().split("\n")
    return data

config_file = "im_config.yaml"
config_dict = get_yaml("im_config.yaml")
word2ind, ind2word = sen2inds.get_worddict(config_dict["wordLabel"])
MAX_WORD_LENGTH = config_dict["wordLength"]


textCNN_param = {
    'vocab_size': len(word2ind),
    'embed_dim': 60,
    'class_num': 5,
    "kernel_num": 16,
    "kernel_size": [3, 4, 5],
    "dropout": 0.5,
}

net_chinese = textCNN(textCNN_param)
weightFile = config_dict["model"]

if os.path.exists(weightFile):
    print("load weight")
    net_chinese.load_state_dict(
        torch.load(weightFile, map_location=torch.device("cpu"))
    )
else:
    print("No weight file!")
    exit()
net_chinese.eval()


def parse_net_result(out):
    score = max(out)
    label = np.where(out == score)[0][0]
    return label, score


def gene_vec(x):
    title_seg = stand_sentence(x)
    # print("word split:", title_seg[:MAX_WORD_LENGTH])
    title_ind = []
    for w in title_seg:
        if w in word2ind:
            title_ind.append(word2ind[w])
        else:
            title_ind.append(word2ind["unknown"])
    length = len(title_ind)
    if length > MAX_WORD_LENGTH:
        title_ind = title_ind[0:MAX_WORD_LENGTH]
    if length < MAX_WORD_LENGTH:
        title_ind.extend([0] * (MAX_WORD_LENGTH - length))
    # print("embeeding split:", title_ind)
    return title_ind, title_seg

def is_chinese(x):
    cnt = 0
    chinese_cnt = 0
    for ch in x:
        # print ( u'/u4e00' <= ch<=u'/u9fa5')
        cnt += 1
        """判断一个unicode是否是汉字"""
        if "\u4e00" <= ch <= "\u9fff":
            chinese_cnt += 1
    # print (chinese_cnt,cnt)
    if chinese_cnt >= 4 and chinese_cnt / cnt >= 0.2:
        return True
    return False

def predict(x):
    data, words = gene_vec(x)
    sentence = np.array([int(x) for x in data[0:MAX_WORD_LENGTH]])
    sentence = torch.from_numpy(sentence)
    predict = (
        net_chinese(sentence.unsqueeze(0).type(torch.LongTensor))
        .cpu()
        .detach()
        .numpy()[0]
    )
    label_pre, _ = parse_net_result(predict)
    # cate0 = predict[0]
    # cate1 = predict[1]
    # prob = math.exp(cate1) / (math.exp(cate1) + math.exp(cate0))
    return label_pre ,predict

if __name__ == "__main__":
    lb = {0:'教育',1:'健康',2:'生活',3:'娱乐',4:'游戏'}
    # text = "女性经常有作乳房自查是否不用每年上医院作体检了？经常自查乳房的， "
    # text = "上海简单有效的祛斑方法_祛斑需要多少钱？"
    # text = "任九，让我优，让我喜~~~~~一次次买彩，一次次倾听那比分，一次"
    # text = "使命召唤2的问题帮帮我谢谢了！从新浪下载的使命召唤2使用了XEZ"
    text = "如果觉得自学考试的成绩与自己所做的试卷不符，应怎么处理？是山东自"
    out = predict(text)
    print(lb[out[0]])
    print(out[1])