import numpy as np
from time import time
import sys,argparse
from pyspark.ml import Pipeline
from pyspark.sql import functions as f
from pyspark import SparkConf,SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import HashingTF,Tokenizer,IDF
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

class CustomCrossValidator(CrossValidator):
    pass

def readData(path,sparkContext):
    return sparkContext.textFile(path).map(eval)
 
def createDataFrameFromRDD(dataRDD,label,sentence,sparkSession):
    return sparkSession.createDataFrame(dataRDD).toDF(label, sentence)

def getFeaturesNum(dataDF,tokenizer,sentence):
	return tokenizer.transform(dataDF)\
                    .withColumn('word', f.explode(f.split(f.col(sentence), ' ')))\
                    .groupBy('word').count().sort('count', ascending=False)\
                    .count()
def setGridSearchSpace(search_params):
    paramGrid = ParamGridBuilder()
    for _,param_val in enumerate(search_params): 
        paramGrid = paramGrid.addGrid(eval(param_val), search_params[param_val])
    paramGrid = paramGrid.build()
    return paramGrid
    


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description = 'Twitter Sentiment Analysis.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--train',default='data/train.txt', help='Input file containing (label,sentences) pairs used to train model')
	parser.add_argument('--test', default='data/test.txt', help='Input file containing  (label,sentences) pairs used to test model')
	parser.add_argument('--k',default=2,type = int,help = 'Number of folds')
	parser.add_argument('--maxIter',default=10,type=int, help='Maximum number of iterations')
	parser.add_argument('--N',default=20,type=int, help='Parallelization Level')
	parser.add_argument('--hiddenlayers',default=[6],type=list, help='Parallelization Level')
	parser.add_argument('--maxNumFeatures',default=None, help='maxNumFeatures parameter search list for cross-validation')
	parser.add_argument('--seed',default=123,type=int, help='Seed used in random number generator')
	parser.add_argument('--stepSize',default=0.1,type=float, help='')
	parser.add_argument('--blockSize',default=512,type=int, help='')
	parser.add_argument('--traindataSize',default=0,type=int, help='')


	verbosity_group = parser.add_mutually_exclusive_group(required=False)
	verbosity_group.add_argument('--verbose', dest='verbose', action='store_true')
	verbosity_group.add_argument('--silent', dest='verbose', action='store_false')
	parser.set_defaults(verbose=False)
	args = parser.parse_args()

	conf = (SparkConf().set("spark.driver.maxResultSize","50g"))
	sc = SparkContext("local[{}]".format(args.N),appName='MLP',conf=conf)
	spark = SparkSession.builder.appName('Twitter_Sentiment_MLP').getOrCreate()

	if not args.verbose :
		sc.setLogLevel("ERROR") 

	trainRDD = readData(args.train,sc)
	train = createDataFrameFromRDD(trainRDD,"label", "sentence",spark)
	if args.traindataSize:
		train = train.take(args.traindataSize)
		train = createDataFrameFromRDD(train,"label", "sentence",spark)


	tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
	
	if args.maxNumFeatures:
		numFeatures = int(args.maxNumFeatures)
	else:
		numFeatures = getFeaturesNum(train,tokenizer,'sentence')

	hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="rawfeatures",numFeatures=int(numFeatures))
	idf = IDF(inputCol=hashingTF.getOutputCol(),outputCol='features')

	mlpTrainer  = MultilayerPerceptronClassifier(
			layers=[numFeatures]+args.hiddenlayers+[2],blockSize=args.blockSize,
			featuresCol='features', labelCol='label',
			stepSize=args.stepSize, maxIter=args.maxIter,seed=args.seed
			)
	# LR = 
	# pipeline = Pipeline(stages=[tokenizer,hashingTF,idf,mlpTrainer])

	# search_params = {'mlpTrainer.stepSize':[0.1]}
	# paramGrid = setGridSearchSpace(search_params)
	# crossval = CrossValidator(
	# 		estimator=pipeline,
	# 		evaluator=BinaryClassificationEvaluator(),
	# 		estimatorParamMaps=paramGrid,
	# 		numFolds=args.k)
	# start = time()
	# cvModel = crossval.fit(train)
	# now = time()-start
	# print('Runtime of cross validation: %f' % now)
	# print('Average AUCs for Cross-Validation:',cvModel.avgMetrics)


	# testRDD = readData(args.test,sc)
	# test = createDataFrameFromRDD(testRDD,"label","sentence",spark)
	# prediction = cvModel.transform(test)
	# evaluator = BinaryClassificationEvaluator()
	# print('AUC for Test Set',evaluator.evaluate(prediction)) 
	start = time()
	pipeline = Pipeline(stages=[tokenizer,hashingTF,idf])
	preparedModel = pipeline.fit(train)
	trainPreparedData= preparedModel.transform(train)

	search_params = {'mlpTrainer.stepSize':[0.1]}
	paramGrid = setGridSearchSpace(search_params)
	crossval = CrossValidator(
			estimator=mlpTrainer,
			evaluator=BinaryClassificationEvaluator(),
			estimatorParamMaps=paramGrid,
			numFolds=args.k)
	end = time()
	cvModel = crossval.fit(trainPreparedData)
	dataTime = end-start
	mlpTime = time()-end
	print('Runtime of cross validation: %f, runtime of data processing: %f' % (mlpTime,dataTime))
	print('Average AUCs for Cross-Validation:',cvModel.avgMetrics)


	testRDD = readData(args.test,sc)
	test = createDataFrameFromRDD(testRDD,"label","sentence",spark)
	testPreparedData= preparedModel.transform(test)
	prediction = cvModel.transform(testPreparedData)
	dataCSV = prediction.select('label','rawPrediction','probability')
	# dataCSV.coalesce(1).write.csv('data/testOutput',header=True,sep=' ',mode='overwrite')
	# dataCSV.rdd.map(lambda x: ",".join(map(str, x))).coalesce(1).saveAsTextFile("output.csv")
	evaluator = BinaryClassificationEvaluator()
	print('AUC for Test Set',evaluator.evaluate(prediction)) 


# C:/Users/16176/Desktop/EECE5645