# -*- coding: utf-8 -*-
import sys,argparse
from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LinearSVC, NaiveBayes
from pyspark.ml.classification import FMClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
import numpy as np
from time import time


class CustomCrossValidator(CrossValidator):
    pass



def readData(f,sparkContext):
    """ Read the data from a file and store them in an rdd containing tuples of the form:
		(label,sentence)

	where label is an integer (0,1) and sentence is a string.

	Inputs are:
	     -f: The name of a file that contains the sentences, in form:
		   (label,sentence)

           per line
	     -sparContext: A Spark context

	The return value is the constructed rdd
    """
    return sparkContext.textFile(f).map(eval)
 
def createDataFrameFromRDD(RDD,label_column,sentence_column,sparkSession):
    """ Create dataframe from RDD and store it dataframe of the form: 
		DataFrame[label, sentence]

	where label is an integer (0,1) and sentence is a string.

    Inputs are: 
        - RDD: The RDD containing the data
        - label_column: label colunm name in string
        - sentence_column: sentence colunm name in string  
        - spark: A Spark Session
        
	The return value is the constructed dataframe
    """
    return sparkSession.createDataFrame(RDD).toDF(label_column, sentence_column)

def getmaxNumFeatures(traindata,tokenizer,sentence_column):
    """ Calcultes number of distinct words in dataframe 

    Inputs are: 
        - traindata: The dataframe containing the data
        - tokenizer: a Tokenization object 
        - sentence_name: sentence colunm name in string  
        
	The return value is max number of distinct words in dataframe
    """
    return tokenizer.transform(train)\
                    .withColumn('word', f.explode(f.split(f.col('sentence'), ' ')))\
                    .groupBy('word')\
                    .count()\
                    .sort('count', ascending=False)\
                    .count()

def setGridSearchSpace(search_params):
    """ Generates a grid search space for cross-validation in form of ParamGridBuilder. 

    Inputs are: 
        - search_params: Parameter dictionary for grid space
             i.e. search_params = {'lr.regParam':[0.1,1].,'hashingTF.numFeatures':[10,20]}

	The return is a ParamGridBuilder object
    """
    
    paramGrid = ParamGridBuilder()
    for _,param_val in enumerate(search_params): 
        paramGrid = paramGrid.addGrid(eval(param_val), search_params[param_val])
    paramGrid = paramGrid.build()

    return paramGrid
    
    

if __name__=="__main__":

    parser = argparse.ArgumentParser(description = 'Twitter Sentiment Analysis.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--traindata',default='train.txt', help='Input file containing (label,sentences) pairs used to train model')
    parser.add_argument('--testdata', default='test.txt', help='Input file containing  (label,sentences) pairs used to test model')
    parser.add_argument('--kfold',default=2,type = int,help = 'Number of folds')
    parser.add_argument('--maxiter',default=2,type=int, help='Maximum number of iterations')
    parser.add_argument('--N',default=20,type=int, help='Parallelization Level')
    parser.add_argument('--classifier',default='SVM', help='Classification Method: ["SVM", "FMC"]')
    parser.add_argument('--regParamRange',type=float, nargs='*', default=[0.1], help='regulatization parameter search list for cross-validation')
    parser.add_argument('--maxNumFeatures',default=None, help='maxNumFeatures parameter search list for cross-validation')


    verbosity_group = parser.add_mutually_exclusive_group(required=False)
    verbosity_group.add_argument('--verbose', dest='verbose', action='store_true')
    verbosity_group.add_argument('--silent', dest='verbose', action='store_false')
    parser.set_defaults(verbose=False)
 
    args = parser.parse_args()

    #sc = SparkContext(appName='Twitter_Sentiment')
    sc = SparkContext("local[{}]".format(args.N), "Twitter_Sentiment")
    spark = SparkSession.builder.appName('Twitter_Sentiment_Spark').getOrCreate()

    if not args.verbose :
        sc.setLogLevel("ERROR")        

    train_rdd = readData(args.traindata,sc)
    test_rdd  = readData(args.testdata,sc)

    train = createDataFrameFromRDD(train_rdd,"label", "sentence",spark)
    test  = createDataFrameFromRDD(test_rdd,"label", "sentence",spark)

    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
    
    if args.maxNumFeatures:
        maxNumFeatures = args.maxNumFeatures
    else:
        maxNumFeatures = getmaxNumFeatures(train,tokenizer,"sentence")
    
    hashingTF = HashingTF(inputCol="words", outputCol="features", numFeatures=int(maxNumFeatures*1.1))
    
    if args.classifier == 'SVM':
        model = LinearSVC(maxIter=args.maxiter)
        search_params = {'model.regParam':args.regParamRange}
	paramGrid = setGridSearchSpace(search_params)
    elif args.classifier == 'FMC':
        model = FMClassifier(maxIter=args.maxiter, solver='gd')
        search_params = {'model.regParam':args.regParamRange}
	paramGrid = setGridSearchSpace(search_params)
    elif args.classifier == 'NB':
	model = NaiveBayes(labelCol="label", featuresCol="features")
	paramGrid = ParamGridBuilder().addGrid(model.smoothing, eval(args.regParamRange)).build()
    else:
        print('Classification method not defined')
        exit()
        
    pipeline = Pipeline(stages=[tokenizer, hashingTF, model])
  
    crossval = CustomCrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=args.kfold)  
    start = time()
    cvModel = crossval.fit(train)
    now = time()-start
    print('Runtime for cross validation: {}'.format(now))
    
    print('Average AUCs for Cross-Validation:',cvModel.avgMetrics)
    
    params = cvModel.getEstimatorParamMaps()[np.argmax(cvModel.avgMetrics)]
    best_params = {}
    for _, val in enumerate(params):
        best_params[val.name]= params[val]
    print('Best Params:', best_params)
    
    
    
    prediction = cvModel.transform(test)
    prediction.show(20)
    evaluator = BinaryClassificationEvaluator()
    print('AUC for Test Set',evaluator.evaluate(prediction)) 
    
    # if args.classifier == 'SVM':
    #     model = LinearSVC(maxIter=args.maxiter, regParam = best_params['regParam'])
    # elif args.classifier == 'FMC':
    #     model = FMClassifier(maxIter=args.maxiter, regParam = best_params['regParam'])

    # pipeline = Pipeline(stages=[tokenizer, hashingTF, model])
    # best_model = pipeline.fit(train)
    # prediction = best_model.transform(test)
    
    # prediction.show(20)
    # print('AUC for Test Set',evaluator.evaluate(prediction))

    
    
    
        
    
    
