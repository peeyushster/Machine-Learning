# DSGA1004 - BIG DATA
## Lab 5: Machine learning with spark
- Prof Brian McFee (bm106)
- Mayank Lamba (ml5711)
- Saumya Goyal (sg5290)

*Handout date*: 2019-04-11

*Submission deadline*: 2019-04-24


## Requirements

This lab assignment will require the use of Spark on the Dumbo cluster.

As in the previous assignment, remember to activate your environment as follows:
```
module load python/gnu/3.6.5
module load spark/2.4.0
```

Additionally, for this lab, you will be submitting non-interactive spark jobs rather than (or in addition to)
using the `pyspark` shell.  Because of how the Dumbo cluster manages python environments, you will need to
specify the python shell when submitting spark jobs.  This is done by setting the `PYSPARK_PYTHON` environment
variable to explicitly give the path to the python binary you're using (e.g., 3.6.5).  You may find it useful to
add the following shell alias to your `~/.bashrc` file:
```
alias spark-submit='PYSPARK_PYTHON=$(which python) spark-submit'
```
which will force `spark` to always match the python version to your currently loaded module when submitting a job.

## Part 0: Introduction

In this assignment, you will develop pipelines for supervised and unsupervised machine learning with Spark's MLlib package.
You will probably want to keep the [MLlib API reference](https://spark.apache.org/docs/latest/ml-guide.html) open while working through this assignment.

As a data source, you will be using a collection gathered from the [Free Music Archive](https://freemusicarchive.org).
Two files are provided for model development:

    - `hdfs:/user/bm106/pub/fma_train.parquet` - the training set
    - `hdfs:/user/bm106/pub/fma_val.parquet` - the evaluation set

A third, private test set will be used for grading purposes.

The full data consists of metadata and a sampling of acoustic features for roughly 100K recordings.
A small sample is included in the project repository as `fma_subset.parquet` which you can use to prototype your implementations below.
Before proceeding through to the rest of the assignment, you may want to pause here to load the data and familiarize yourself with its structure.

### Getting started

Unlike the previous assignments, which relied on the spark console, in this assignment you will work primarily through the `spark-submit` interface.

To help familiarize yourself with this mode of spark, a simple test script `count.py` is included in this repository.  This script instantiates a spark session, loads a parquet
file into a dataframe, computes a row count, and prints the results to the screen.
You can run it by saying (from the shell prompt)

```
spark-submit count.py hdfs:/path/to/some/file.parquet
```
where the last argument corresponds to the file you want to load.  Be sure to read through this script and understand its various components before proceeding.


## Part 1: Unsupervised model

Your task for the first section of this assignment is to implement a pipeline for K-means clustering.

Two template files are provided in the repository: `unsupervised_train.py` and `unsupervised_test.py`.  These scripts instantiate the spark session and handle command-line
arguments, but it will be up to you to implement the computation.

### Part 1a: unsupervised_train.py

The training script takes two command-line arguments:

- The path to the parquet file containing training data
- The path to store the trained model pipeline object

For example:
```
spark-submit unsupervised_train.py ./fma_subset.parquet ./kmeans_model
```
After running this script, the trained model should be saved for use with the evaluation script in the next section.


Your pipeline should consist of the following steps.

- Select out the 20 attribute columns labeled `mfcc_00, mfcc_01, ..., mfcc_19`
- Normalize the features using a `StandardScaler`
- Fit a K-means clustering model to the standardized data with K=100.


### Part 1b: unsupervised_test.py

The testing script also takes two command-line arguments:

- The path to the previously trained model
- The path to the parquet file containing the evaluation data

After loading the trained model pipeline, evaluate the clustering using the `ClusteringEvaluator`, and print the resulting score to console.


### Part 1c: evaluation

You may use the `fma_subset` file for development, but then train your model on the
full `fma_train` file and evaluate it on `fma_val`.  Document the resulting model
score in the *Part 1* section of `results.txt`.


## Part 2: Supervised model

Your task for the second part is to implement a supervised learning pipeline for
multi-class classification.

The target class is stored in the `genre` column.  **NOTE**: not every example has
an observed `genre` label!

### Part 2a: supervised_train.py

- Select and standardize the `mfcc_*` features as in Part 1
- Encode the `genre` field as a target label using a `StringIndexer` and store the
  result as `label`.
- Define a multi-class logistic regression classifier.
- Optimize the hyper-parameters (elastic net parameter and regularization weight) of your model by 5-fold cross-validation on the training set.  Use at least 5 distinct values for each parameter in your grid.

Combine the entire process into a Pipeline object.
Once the model pipeline has been fit, save it to the provided model filename.


### Part 2b: supervised_test.py

Like in Part 1b, the test script should load the pre-trained model and a parquet
file containing evaluation data.

After classifying the validation, use the `MultiClassMetrics` evaluator to report
the following:

- Overall precision, recall, and F1
- Weighted precision, recall, and F1
- Per-class precision, recall, and F1

Your script should print these results to the console.

### Part 2c: evaluation

As before, train your pipeline on `fma_train` and evaluate on `fma_val`.  Record the
output of your evaluation in `results.txt`.


## What to turn in

    - Your implementations for parts 1a, 1b, 2a, and 2b
    - A brief writeup of your findings (model evaluations) in `results.txt`
