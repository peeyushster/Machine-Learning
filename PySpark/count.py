#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''A simple pyspark script to count the number of rows in a parquet-backed dataframe

Usage:

    $ spark-submit count.py hdfs:/path/to/file.parquet

'''


# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession


def main(spark, filename):
    '''Main routine for the row counter

    Parameters
    ----------
    spark : SparkSession object

    filename : string, path to the parquet file to load
    '''

    # Load the dataframe
    df = spark.read.parquet(filename)

    # Give the dataframe a temporary view so we can run SQL queries
    df.createOrReplaceTempView('my_table')

    # Construct a query
    query = spark.sql('SELECT count(*) FROM my_table')

    # Print the results to the console
    query.show()


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get the filename from the command line
    filename = sys.argv[1]

    # Call our main routine
    main(spark, filename)
