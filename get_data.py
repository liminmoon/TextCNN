# -*- coding: utf-8 -*-
# 中文数据是从https://github.com/brightmart/nlp_chinese_corpus下载的
'''
从原数据中选取部分数据；
选取数据的title前两个字符在字典WantedClass中；
且各个类别的数量为WantedNum
'''
import json

TrainJsonFile = '/Users/limin/opt/anaconda3/envs/risk/limin/textCNN_demo/data/baike_qa_train.json'
MyTainJsonFile = '/Users/limin/opt/anaconda3/envs/risk/limin/textCNN_demo/data/my_traindata.json'
StopWordFile = '/Users/limin/opt/anaconda3/envs/risk/limin/textCNN_demo/data/stopword.txt'

# 获取测试样本
# TrainJsonFile = '/Users/limin/opt/anaconda3/envs/risk/limin/textCNN_demo/data/baike_qa_valid.json'
# MyTainJsonFile = '/Users/limin/opt/anaconda3/envs/risk/limin/textCNN_demo/data/my_validata.json'

WantedClass = {'教育': 0, '健康': 0, '生活': 0, '娱乐': 0, '游戏': 0}
# WantedNum = 5000
WantedNum = 3000
numWantedAll = WantedNum * 5
def main():
    Datas = open(TrainJsonFile , 'r', encoding='utf_8').readlines()
    f = open(MyTainJsonFile , 'w', encoding='utf_8')

    numInWanted = 0
    for line in Datas:
        data = json.loads(line)
        cla = data['category'][0:2]
        if cla in WantedClass and WantedClass[cla] < WantedNum:
            json_data = json.dumps(data, ensure_ascii=False)
            f.write(json_data)
            f.write('\n')
            WantedClass[cla] += 1
            numInWanted += 1
            if numInWanted >= numWantedAll:
                break
if __name__ == "__main__":
    main()