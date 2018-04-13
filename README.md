# classification_character
## はじめに
character_doc2vecを用いて作成した分散表現モデルを
利用して分類問題を解こうというものになります。  

また、doc2vecの精度の確認もしたかったためBoWの分散表現での分類もしています。  
bowファイルとsklearnファイルはBoWの分類になります。  

分類にはChainerのFFNNの3層モデルを使って行っています。  
また、sklearnでも分類が解けるので精度確認のため行っています。
  
キャラクター1　台詞  
キャラクター2　台詞  
......  
というデータを用いて台詞を該当キャラに分類しようとしています。

## 使用方法  
例としてchainer_learn_BoWを使っています。  
python chainer_learn_BoW.py --data 分類したいデータ --dict 辞書データ --epoch エポック数 --batchsize バッチサイズ  --units 隠れ層のユニット数 --labelnum 分類先の数  
で学習、検証が始まります  
またbowとsklearnには辞書データが必要になりますので準備してください。  

## 実行環境  
Python 3.5.1  
mecab-python3 (0.7)  
Scrapy (1.5.0)  
requests (2.18.4)  
gensim (3.4.0)  
chainer(3.0.0)    
sklearn(0.19.1)  
  
## 実行してみて  
実行結果の予測を見るとdoc2vecの精度が70%、BoWが90%となっています。  
このためdoc2vecの学習が上手くできていない、もしくはもっと幅広いデータを使って作成する必要があることが分かりました。  
そのため次はWikiを使った分散表現を使ってみようと思います。

## 参考書籍,Webサイト  
Chainerv2による実践深層学習  
chainerで文書分類（ニュースをカテゴリ分け）　訓練後を中心:http://geomerlin-com.hatenablog.com/entry/2017/01/27/223524  
