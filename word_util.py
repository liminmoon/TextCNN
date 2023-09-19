# -*- coding: utf-8 -*-
import jieba
import yaml


def get_yaml(f):
    with open(f) as stream:
        yaml_data = yaml.safe_load(stream)
    return yaml_data


def read_stopword(file):
    data = open(file, "r", encoding="utf_8").read().split("\n")
    return data


def get_yaml(f):
    with open(f) as stream:
        yaml_data = yaml.safe_load(stream)
    return yaml_data


def read_stopword(file):
    data = open(file, "r", encoding="utf_8").read().split("\n")
    return data


config_file = "im_config.yaml"
config_dict = get_yaml("im_config.yaml")
stoplist = read_stopword(config_dict["stopwords"])


def isChinese(character):
    for cha in character:
        if not "\u0e00" <= cha <= "\u9fa5":
            return False
    return True


def stand_sentence(line):
    title_seg = jieba.cut(line, cut_all=False)
    res = []
    for w in title_seg:
        w = w.strip()
        if w in stoplist or w.isdigit() or not isChinese(w) or w == "":
            continue
        res.append(w)
    return res
