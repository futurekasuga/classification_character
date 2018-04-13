from gensim import corpora, matutils, models
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from collections import defaultdict
from sklearn.metrics import classification_report
import codecs
import MeCab
import numpy as np
mecab = MeCab.Tagger('mecabrc -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/')

def make_name2serif(f_name,c_num):
    name2serif = defaultdict(list)
    name2id = {}
    name_id = 1
    for line in codecs.open(f_name,'r','utf-8'):
        line = line.rstrip().split(' ')
        name = line[0]
        serif = line[1]
        name2serif[name].append(serif)
        if not name2id.get(name):
            name2id[name] = name_id
            name_id += 1
        if name_id == c_num+1:
            break;
    return name2serif,name2id

def tokenize(text):
    '''
    とりあえず形態素解析して名詞だけ取り出す感じにしてる
    '''
    node = mecab.parseToNode(text)
    while node:
        if node.feature.split(',')[0] == '名詞':
            yield node.surface.lower()
        node = node.next

def get_words_main(content):
    '''
    一つの記事を形態素解析して返す
    '''
    return [token for token in tokenize(content)]

def get_vector(dictionary, content):
    '''
    ある記事の特徴語カウント
    '''
    tmp = dictionary.doc2bow(get_words_main(content))
    dense = list(matutils.corpus2dense([tmp], num_terms=len(dictionary)).T[0])
    return dense

def main():
    f_name = 'imascg_corpus.txt'
    c_num=3
    name2serif,name2id = make_name2serif(f_name,c_num)
    dictionary = corpora.Dictionary.load_from_text('imascg_dic.txt')
    # 特徴抽出
    data_train = []
    label_train = []
    for i,(name, serifs) in enumerate(name2serif.items()):
        for serif in serifs:
            try:
                data_train.append(get_vector(dictionary, serif))
                label_train.append(name2id[name])
            except:
                print('error')
    datasets = {}
    datasets['data'] = np.array(data_train)
    datasets['label'] = np.array(label_train)
    # 学習データと試験データに分けてみる
    data_train_s, data_test_s, label_train_s, label_test_s = train_test_split(datasets['data'], datasets['label'], test_size=0.15)

    # 分類器をもう一度定義
    estimator2 = RandomForestClassifier()

    # 学習
    estimator2.fit(data_train_s, label_train_s)
    print(estimator2.score(data_test_s, label_test_s))


if __name__ == '__main__':
    main()
