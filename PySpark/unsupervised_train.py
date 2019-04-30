#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Part 1: unsupervised model training

Usage:

    $ spark-submit unsupervised_train.py hdfs:/path/to/file.parquet hdfs:/path/to/save/model

'''


# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import KMeans

def main(spark, data_file, model_file):
    '''Main routine for unsupervised training

    Parameters
    ----------
    spark : SparkSession object

    data_file : string, path to the parquet file to load

    model_file : string, path to store the serialized model file
    '''

    # Read parquet dataset
    dataset = spark.read.parquet(data_file)
    
    # Extract features mfcc_00, mfcc_01, ..., mfcc_19 
    assembler = VectorAssembler(inputCols =
    ["mfcc_00","mfcc_01","mfcc_02","mfcc_03","mfcc_04","mfcc_05","mfcc_06","mfcc_07","mfcc_08","mfcc_09","mfcc_10","mfcc_11","mfcc_12","mfcc_13","mfcc_14","mfcc_15","mfcc_16","mfcc_17","mfcc_18","mfcc_19"], outputCol = "features")

    # Normalize features using StandardScaler
    scaler = StandardScaler(inputCol="features",outputCol="scaledFeatures",withStd=True,withMean=True)
   
    # Define k-means model.
    kmeans = KMeans(featuresCol="scaledFeatures").setK(100).setSeed(0)

    # Build pipeline
    pipeline = Pipeline(stages=[assembler, scaler, kmeans])

    # Fit on Training data
    model = pipeline.fit(dataset)

    # Save model.
    model.save(model_file)

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('unsupervised_train').getOrCreate()

    # Get the filename from the command line
    data_file = sys.argv[1]

    # And the location to store the trained model
    model_file = sys.argv[2]

    # Call our main routine
    main(spark, data_file, model_file)
