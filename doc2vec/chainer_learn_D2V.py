# coding: utf-8
import numpy as np
from sklearn.cross_validation import train_test_split
from collections import defaultdict
import sys
import chainer
import chainer.links as L
from chainer import Link, Chain, ChainList
from chainer import optimizers, cuda, serializers
import chainer.functions as F
import argparse
from gensim import corpora, matutils, models
from gensim.models.doc2vec import Doc2Vec
import codecs
import MeCab
mecab = MeCab.Tagger('mecabrc -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/')



#モデルの定義
class ClassificChain(Chain):
    def __init__(self,in_units,n_units,n_label):
        super(ClassificChain, self).__init__(
            l1=L.Linear(in_units, n_units),
            l2=L.Linear(n_units, n_units),
            l3=L.Linear(n_units,  n_label)
        )

    def __call__(self,x,y):
        return F.softmax_cross_entropy(self.fwd(x), y), F.accuracy(self.fwd(x), y)

    def fwd(self,x):
        h1 = self.l1(x)
        h1 = F.sigmoid(h1)
        h2 = self.l2(h1)
        h2 = F.sigmoid(h2)
        y = self.l3(h2)
        return y

def training(x_train,y_train,N,batchsize,model,optimizer):
    #ランダムな整数列リストを取得
    perm = np.random.permutation(N)
    #誤差を見れるよう
    sum_train_loss     = 0.0
    sum_train_accuracy = 0.0
    for i in range(0, N, batchsize):
        #perm を使い x_train, y_trainからデータセットを選択 (毎回対象となるデータは異なる)
        #xが入力(台詞),tが正解ラベル(キャラID)
        x = chainer.Variable(np.asarray(x_train[perm[i:i + batchsize]]))
        t = chainer.Variable(np.asarray(y_train[perm[i:i + batchsize]]))
        # 勾配をゼロ初期化
        model.zerograds()
        # 順伝搬
        loss, acc = model(x,t)
        # 平均誤差計算用
        sum_train_loss      += float(cuda.to_cpu(loss.data)) * len(t)
        # 平均正解率計算用
        sum_train_accuracy  += float(cuda.to_cpu(acc.data )) * len(t)
        # 誤差逆伝播、最適化
        loss.backward()
        optimizer.update()
    #誤差を出していき、学習の具合を確認
    print('train mean loss={}, accuracy={}'.format(sum_train_loss / N, sum_train_accuracy / N))

def evaluation(x_test,y_test,N_test,batchsize,model):
    #学習の評価(eval)
    sum_test_loss     = 0.0
    sum_test_accuracy = 0.0
    for i in range(0, N_test, batchsize):
        #testのデータをモデルに入れる
        x = chainer.Variable(np.asarray(x_test[i:i + batchsize]))
        t = chainer.Variable(np.asarray(y_test[i:i + batchsize]))
        loss, acc = model(x,t)
        sum_test_loss     += float(cuda.to_cpu(loss.data)) * len(t)
        sum_test_accuracy += float(cuda.to_cpu(acc.data))  * len(t)
    #確認
    print(' test mean loss={}, accuracy={}'.format(sum_test_loss / N_test, sum_test_accuracy / N_test))

def save(model,optimizer):
    print('save the model')
    serializers.save_npz('classifier_ffnn_d2v.model', model)
    print('save the optimizer')
    serializers.save_npz('classifier_ffnn_d2v.state', optimizer)


def make_name2serif(f_name,c_num):
    """
    自分の分類したいデータセットに応じて書き換えてください
    今回は台詞からキャラの分類をしたいのでこの形式です。
    """
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
        #本来ならすべて分類すべきだが分類先が多すぎると
        #学習がうまくいかないため上限を設けている
        if name_id == c_num+1:
            break;
    return name2serif,name2id

def tokenize(text):
    '''
    形態素解析して名詞だけを対象としている
    ストップワードの設定等は未熟なため行っていません
    '''
    node = mecab.parseToNode(text)
    while node:
        if node.feature.split(',')[0] == '名詞':
            yield node.surface.lower()
        node = node.next

def get_words_main(content):
    '''
    形態素解析して返す
    '''
    return [token for token in tokenize(content)]

def load_data(f_name,c_num):
    name2serif,name2id = make_name2serif(f_name,c_num)
    doc2vec_model = Doc2Vec.load('imascg_doc2vec.model')
    # 特徴抽出
    data_train = []
    label_train = []
    for i,(name, serifs) in enumerate(name2serif.items()):
        for serif in serifs:
            try:
                #ベクトル表現と正解ラベルの設定
                data_train.append(doc2vec_model.infer_vector(get_words_main(serif)))
                label_train.append(name2id[name])
            except:
                print('error')
    #chainerで扱えるようにnpに変換
    datasets = {}
    datasets['serif'] = np.array(data_train)
    datasets['character'] = np.array(label_train)
    return datasets

def main():
    #引数の設定
    parser = argparse.ArgumentParser()
    parser.add_argument('--data '    , dest='data'       , type=str, default='input.dat',  help='an input data file')
    parser.add_argument('--epoch'    , dest='epoch'      , type=int, default=100,          help='number of epochs to learn')
    parser.add_argument('--batchsize', dest='batchsize'  , type=int, default=10,           help='learning minibatch size')
    parser.add_argument('--units'    , dest='units'      , type=int, default=500,           help='number of hidden unit')
    parser.add_argument('--labelnum'    , dest='labelnum'      , type=int, default=3,           help='number of all label num')
    parser.add_argument('--classificnum'    , dest='classificnum'      , type=int, default=3,           help='number of classific num')
    args = parser.parse_args()

    #対象にするデータセットの準備
    dataset = load_data(args.data,args.classificnum)
    #文書のベクトルと正解ラベルの設定
    dataset['serif'] = dataset['serif'].astype(np.float32)
    dataset['character'] = dataset['character'].astype(np.int32)
    #データセットを訓練とテストに分割する(test_sizeの比率でやるので今回は85%を訓練に)
    x_train, x_test, y_train, y_test = train_test_split(dataset['serif'], dataset['character'], test_size=0.15)
    #テストと訓練のサイズ
    N_test = y_test.size
    N = len(x_train)
    #ユニット数を決める(入力は語彙数にあたるので固定)
    in_units = x_train.shape[1]
    n_units = args.units
    n_label = args.labelnum

    #モデルの定義と最適化の設定
    model = ClassificChain(in_units,n_units,n_label)
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    #ミニバッチのサイズとエポック数
    batchsize   = args.batchsize
    n_epoch     = args.epoch
    model.zerograds()
    #学習
    for epoch in range(1, n_epoch + 1):
        print('epoch', epoch)
        training(x_train,y_train,N,batchsize,model,optimizer)
        evaluation(x_test,y_test,N_test,batchsize,model)

    #modelとoptimizerを保存
    save(model,optimizer)

if __name__ == '__main__':
    main()
