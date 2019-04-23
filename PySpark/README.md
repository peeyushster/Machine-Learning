### Unsupervised Training Using PySpark ML Library: unsupervised_train.py

The training script takes two command-line arguments:

- The path to the parquet file containing training data
- The path to store the trained model pipeline object

For example:
```
spark-submit unsupervised_train.py ./fma_subset.parquet ./kmeans_model
```
After running this script, the trained model should be saved for use with the evaluation script.


Sample pipeline consists of the following steps.

- Select out the 20 attribute columns labeled `mfcc_00, mfcc_01, ..., mfcc_19`
- Normalize the features using a `StandardScaler`
- Fit a K-means clustering model to the standardized data with K=100.

Evaluation (unsupervised_test.py)

The testing script also takes two command-line arguments:

- The path to the previously trained model
- The path to the parquet file containing the evaluation data



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
