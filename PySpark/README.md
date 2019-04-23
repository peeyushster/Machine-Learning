# PySpark ML Library

### Unsupervised Training: unsupervised_train.py

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



### Supervised Training (code in supervised_train.py)

Script implemented a supervised learning pipeline for multi-class classification.

The target class is stored in the `genre` column.  

Steps taken to implement supervised training -
- Select and standardize the `mfcc_*` features 
- Encode the `genre` field as a target label using a `StringIndexer` and store the result as `label`.
- Define a multi-class logistic regression classifier.
- Optimize the hyper-parameters (elastic net parameter and regularization weight) of your model by 5-fold cross-validation on the training set.  Use at least 5 distinct values for each parameter in your grid.

Combine the entire process into a Pipeline object. Once the model pipeline has been fit, save it to the provided model filename.

Evaluation Metrics

After classifying the validation, use the `MultiClassMetrics` evaluator to report the following:

- Overall precision, recall, and F1
- Weighted precision, recall, and F1
- Per-class precision, recall, and F1
