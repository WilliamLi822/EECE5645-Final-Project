# EECE5645-Final-Project
Twitter Sentiment Analysis with Distributed Machine Learning

# You need to download train.txt(~110 MB) and test.txt from my gdrive 
https://drive.google.com/file/d/1asuqjvl-omT_fqPu0QVdweFbiGjLOccX/view?usp=sharing
# FMClassifier
Rendle, Steffen. "Factorization machines." In 2010 IEEE International Conference on Data Mining, pp. 995-1000. IEEE, 2010.

Setup on discovery:
```sh
$ python3.6 -m venv env
$ . ./env/bin/activate
$ pip install numpy pyspark
```

Run
```
$ spark-submit --executor-memory 100G --driver-memory 100G TwitterSentiment.py --traindata train_all.txt --testdata test_all.txt --kfold 2 --maxiter 5 --N 20 --classifier FMC --regParamRange 0 0.1 1 10 50 100
Runtime for cross validation: 492.71633410453796
Average AUCs for Cross-Validation: [0.6795060176988725, 0.6794446479302987, 0.6764347774945112, 0.506350101467526, 0.5077198143633668, 0.5078606751984689]
Best Params: {'regParam': 0.0}
+-----+--------------------+--------------------+--------------------+--------------------+--------------------+----------+
|label|            sentence|               words|            features|       rawPrediction|         probability|prediction|
+-----+--------------------+--------------------+--------------------+--------------------+--------------------+----------+
|    1|on verge of devel...|[on, verge, of, d...|(463857,[40808,50...|[-0.0783588907683...|[0.48042029475422...|       1.0|
|    0|you should of fou...|[you, should, of,...|(463857,[14465,63...|[0.04582694334229...|[0.51145473122397...|       0.0|
|    0|too busy at work ...|[too, busy, at, w...|(463857,[23337,27...|[0.04948095431941...|[0.51236771529287...|       0.0|
|    1|lol well maybe th...|[lol, well, maybe...|(463857,[14664,16...|[-0.0505872025448...|[0.48735589567259...|       1.0|
|    1|     i love the rain|[i, love, the, rain]|(463857,[10264,77...|[0.06566172009563...|[0.51640953469111...|       0.0|
|    0|drinking shots an...|[drinking, shots,...|(463857,[84943,11...|[-0.0326737606434...|[0.49183228646318...|       1.0|
|    0|i'm a bit hungry....|[i'm, a, bit, hun...|(463857,[66119,12...|[0.00310532079021...|[0.50077632957370...|       0.0|
|    1|        good morning|     [good, morning]|(463857,[2033,158...|[-0.0659908344783...|[0.48350827577938...|       1.0|
|    0|im supposed to go...|[im, supposed, to...|(463857,[990,2002...|[0.16816258788120...|[0.54194185524793...|       0.0|
|    0|thanx i really do...|[thanx, i, really...|(463857,[153387,1...|[0.13515769970325...|[0.53373808106447...|       0.0|
|    0|i was gonna go lo...|[i, was, gonna, g...|(463857,[19103,49...|[0.15469033486498...|[0.53859565112780...|       0.0|
|    1|aha awww you look...|[aha, awww, you, ...|(463857,[990,8494...|[-0.0835991712708...|[0.47911237075813...|       1.0|
|    0|an injured guy ir...|[an, injured, guy...|(463857,[155867,2...|[-0.0266186324023...|[0.49334573480227...|       1.0|
|    0|was going to duxf...|[was, going, to, ...|(463857,[14465,19...|[0.37787589083841...|[0.59336069165190...|       0.0|
|    0|i didn't see you ...|[i, didn't, see, ...|(463857,[33324,15...|[0.03986433438158...|[0.50996476399237...|       0.0|
|    1|just won the semi...|[just, won, the, ...|(463857,[77471,11...|[-0.0438231819849...|[0.48904595752465...|       1.0|
|    0|dontyouhate loosi...|[dontyouhate, loo...|(463857,[14465,41...|[0.00294092302666...|[0.50073523022674...|       0.0|
|    1|     rnb all the way|[rnb, all, the, way]|(463857,[13593,33...|[-0.0435714896156...|[0.48910885058920...|       1.0|
|    1|thnx part of my q...|[thnx, part, of, ...|(463857,[3744,158...|[0.11870410913464...|[0.52964123006655...|       0.0|
|    1|hello rob's daughter|[hello, rob's, da...|(463857,[120846,2...|[-0.0292160687897...|[0.49269650230432...|       1.0|
+-----+--------------------+--------------------+--------------------+--------------------+--------------------+----------+
only showing top 20 rows

AUC for Test Set 0.6792774457599059

```

# Plotting ROC curve
Create a csv file with headers 'label' and 'score' which represent the ground-truth and the score for each point when predicted by a model. You can also use the probability but you need to remove `pos_label=1`.

To install, after activating the virtualenv:
```sh
$ pip install numpy scikit-learn matplotlib
$ sudo apt-get install python3-tk
```

Then run:
```sh
$ python plot_roc.py --input scores.csv
```

The script used for storing the scores:
```python
cols = ['label', 'probability']
out = prediction.select(*cols)
scores = out.rdd.mapValues(lambda x: x[0])
out.map(lambda x: ",".join(map(str, x))).coalesce(1).saveAsTextFile("roc_data")
```
