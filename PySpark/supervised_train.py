#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Part 2: supervised model training

Usage:

    $ spark-submit supervised_train.py hdfs:/path/to/file.parquet hdfs:/path/to/save/model

'''


# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
def main(spark, data_file, model_file):
    '''Main routine for supervised training

    Parameters
    ----------
    spark : SparkSession object

    data_file : string, path to the parquet file to load

    model_file : string, path to store the serialized model file
    '''
    
    # Load the dataset
    dataset = spark.read.parquet(data_file).sample(True,0.1,1)

    # Extract features mfcc_00, mfcc_01, ..., mfcc_19 
    assembler = VectorAssembler(inputCols = ["mfcc_00","mfcc_01","mfcc_02","mfcc_03","mfcc_04","mfcc_05","mfcc_06","mfcc_07","mfcc_08","mfcc_09","mfcc_10","mfcc_11","mfcc_12","mfcc_13","mfcc_14","mfcc_15","mfcc_16","mfcc_17","mfcc_18","mfcc_19"], outputCol = "features")

    # Normalize the features using StandardScaler
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                            withStd=True, withMean=True)
    
    # Encoded the genre target-feature
    indexer = StringIndexer(inputCol="genre",outputCol="label").setHandleInvalid("skip")    

    # Define model for Logistics Regression  
    lr = LogisticRegression(maxIter=10,featuresCol="scaledFeatures")

    # Define Pipeline
    pipeline = Pipeline(stages=[assembler, scaler, indexer, lr])

    # Define the hyperparameters
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.elasticNetParam, [0.1, 0.3, 0.5, 0.7, 0.9]) \
            .addGrid(lr.regParam, [100, 1, 0.1, 0.01, 0.001]) \
                .build()
    
    # Set cross-validate to tune the hyperparameters
    crossval = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid,
        evaluator=MulticlassClassificationEvaluator(), numFolds=5)
    
    cvModel = crossval.fit(dataset)

    # Save the best fit model
    cvModel.bestModel.save(model_file)

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('supervised_train').getOrCreate()

    # Get the filename from the command line
    data_file = sys.argv[1]

    # And the location to store the trained model
    model_file = sys.argv[2]

    # Call our main routine
    main(spark, data_file, model_file)
