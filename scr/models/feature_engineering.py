import os
import mlflow
import pandas as pd
from optbinning import BinningProcess
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark.conf import SparkConf
import pyspark.sql.functions as f

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

config = SparkConf().setAll([('spark.executor.memory', '40g'), ('spark.executor.cores', '16'), ('spark.cores.max', '16'), ('spark.driver.memory', '30g')])
sc = SparkContext(conf=config)
spark = SparkSession(sc)

DIR_LOCAL = os.path.dirname(os.path.realpath('__file__'))
DIR_PATH = os.path.join(DIR_LOCAL, 'data')
filename = 'dataset_model.parquet'
dataset = spark.read.parquet(os.path.join(DIR_PATH, filename))
dataset = (
    dataset
    .filter(f.col('dt_year')>=2022)
    .withColumn('dt_year', f.col('dt_year').cast('string'))
    .withColumn('dt_month', f.month(f.col('date')))
)

dataset.columns

# Creating Datasets
dataset = dataset.toPandas()

# dataset['dt_month'].unique().tolist()
out_of_time = [1,3,5,7,9,11]
training_dates = [2,4,6,8,12]
train = dataset[dataset['dt_month'].isin(training_dates)]
out_of_time = dataset[dataset['dt_month'].isin(out_of_time)]

# Defining Variables Columns
ident_cols = ['date', 'id_isin', 'id_cnpj', 'ticker', 'adj_close', 'close', 'dividends', 'high_price', 'low_price', 'open',
              'stock_splits', 'volume', 'dt_year', 'dt_quarter', 'id_cvm', 'id', 'txt_company_name',
              'dt_refer', 'dt_fim_exerc', 'dt_ini_exerc', 'dt_refer_lag', 'dt_month', 'processed_at', 'cat_target', 'var_target']
y_col= 'target'
X_cols = [col for col in dataset.columns if col not in ident_cols]

# Train
X = train[X_cols]
y = train[y_col].values
id = train[ident_cols]

# Out-of-Time
X_out = out_of_time[X_cols]
y_out = out_of_time[y_col].values
dataset_out = out_of_time[ident_cols]


# Spliting Train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=7)

# Getting Index Rows
dataset_train = id.loc[X_train.index, :]
dataset_test = id.loc[X_test.index, :]

# Study
binning_process = BinningProcess(variable_names=X_cols, min_bin_size=0.1)
binning_process.fit(X,y)
iv_tb = binning_process.summary().sort_values(by='iv', ascending=False)
iv_tb.to_csv(os.path.join(DIR_PATH, 'iv_tb.csv'), index=False)
