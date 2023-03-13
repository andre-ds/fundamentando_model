import os
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
import lightgbm as lgb
from utils.utils import make_predictions
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, roc_auc
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark.conf import SparkConf
import pyspark.sql.functions as f
from utils.utils import dataset_description
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pylab as plt
import seaborn as sns
from lightgbm import LGBMClassifier, plot_importance 
from plot_metric.functions import BinaryClassification
from optbinning.scorecard import plot_auc_roc, plot_cap, plot_ks
import scikitplot as skplt
from utils.utils import Modeling


def objective(trial):
    
    # Invoke suggest methods of a Trial object to generate hyperparameters.
    regressor_name = trial.suggest_categorical('classifier', ['lightGBM'])
    if regressor_name == 'RandomForest':
        max_depth = trial.suggest_int(name='max_depth', low=5, high=14)
        max_leaf_nodes = trial.suggest_int(name='max_leaf_nodes', low=10, high=40)
        min_impurity_decrease = trial.suggest_float(name='impurity', low=0.0, high=1.0, step=0.1)
        min_samples_leaf = trial.suggest_int(name='min_samples_leaf', low=10, high=60)
        n_estimators =  trial.suggest_int(name='n_estimators', low=25, high=50)
        model = RandomForestClassifier(random_state=7, class_weight='balanced_subsample',
            max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
            min_samples_leaf=min_samples_leaf, n_estimators=n_estimators)
    elif regressor_name == 'lightGBM':
        importance_type = trial.suggest_categorical('importance_type', ['gain'])
        boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart'])
        n_estimators =  trial.suggest_int(name='n_estimators', low=50, high=1000)
        objective = trial.suggest_categorical('objective', ['binary'])
        n_jobs = trial.suggest_categorical('n_jobs', [-1])
        class_weight = trial.suggest_categorical('class_weight', ['balanced'])
        metric = trial.suggest_categorical('metric', ['binary_logloss', 'cress_entropy', 'auc', 'avarege_precision'])
        reg_alpha = trial.suggest_float(name='reg_alpha', low=0.0, high=1, step=0.025)
        reg_lambda = trial.suggest_float(name='reg_lambda', low=0.0, high=1, step=0.025)
        max_depth = trial.suggest_int(name='max_depth', low=5, high=12)
        min_child_samples = trial.suggest_int(name='min_child_samples', low=1000, high=25000)
        model = lgb.LGBMClassifier(random_state=7, importance_type=importance_type, boosting_type=boosting_type,
                        objective=objective, n_jobs=n_jobs, class_weight=class_weight, metric=metric, reg_lambda=reg_lambda,
                        reg_alpha=reg_alpha, max_depth=max_depth, n_estimators=n_estimators,
                        min_child_samples=min_child_samples)

    model.fit(md.X_train, md.y_train)
    pred = model.predict(md.X_train)
    metric = roc_auc_score(md.y_train, pred)

    return metric 



config = SparkConf().setAll([('spark.executor.memory', '40g'), ('spark.executor.cores', '16'), ('spark.cores.max', '16'), ('spark.driver.memory', '30g')])
sc = SparkContext(conf=config)
spark = SparkSession(sc)

DIR_LOCAL = os.path.dirname(os.path.realpath('__file__'))
DIR_PATH = os.path.join(DIR_LOCAL, 'data')
filename = 'dataset_model.parquet'

dataset = (
    spark.read.parquet(os.path.join(DIR_PATH, filename))
    .filter(f.col('dt_year')>=2022)
)

# Convert to Pandas
dataset = dataset.toPandas()
# Missing Values
dataset_description(dataset=dataset)
# Fill NA
dataset = dataset.fillna(value=0)

# Creating Datasets
out_of_time = [1,3,5,7,9,11]
training_dates = [2,4,6,8,12]
train = dataset[dataset['dt_month'].isin(training_dates)]
out_of_time = dataset[dataset['dt_month'].isin(out_of_time)]

# Defining Variables Columns
ident_cols = ['id_cnpj', 'id_cvm', 'txt_company_name', 'ticker',  'id_isin', 'processed_at', 'date',  'dt_refer', 'dt_year', 'dt_month',
              'dt_safra_dre', 'dt_fim_exerc', 'dt_ini_exerc', 'cat_target', 'var_target', 'amplitude_close_high_price']

y_col= 'cat_target'
X_cols = [col for col in dataset.columns if col not in ident_cols]
#X_cols = [col for col in dataset.columns if col not in ident_cols]

# Train
X = train[X_cols]
y = train[y_col].values
id = train[ident_cols]

# Out-of-Time
X_out = out_of_time[X_cols]
y_out = out_of_time[y_col].values
dataset_out = out_of_time[ident_cols]

# Spliting Train
md = Modeling(X=X, y=y, X_out=X_out, y_out=y_out)
md._split_dataset()

# Getting Index Rows
dataset_train = id.loc[md.X_train.index, :]
dataset_test = id.loc[md.X_test.index, :]

# Hyperparameters Tunning
mlflow.set_experiment(experiment_name='hyperparameter-tunning-1')
with mlflow.start_run():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=500)
    best_params = study.best_params
    best_params.pop('classifier')
    mlflow.log_params(params=best_params)
    mlflow.sklearn.log_model(study, 'optuna')
    mlflow.end_run()


#best_params = {'importance_type': 'gain', 'boosting_type': 'gbdt', 'n_estimators': 1000, 'objective': 'binary', 'n_jobs': -1, 'class_weight': 'balanced', 'metric': 'cress_entropy', 'reg_alpha': 0.5750000000000001, 'reg_lambda': 0.55, 'max_depth': 10, 'min_child_samples': 1005}

mlflow.set_experiment(experiment_name='version-all-variables-1')
with mlflow.start_run():
    mlflow.log_params(params=best_params)
    model = lgb.LGBMClassifier(random_state=7, **best_params)
    model.fit(md.X_train, md.y_train)
    
    # Log Metrics
    md.prediction_train(model=model)
    md.prediction_test(model=model)
    md.prediction_out(model=model)
    md.metrics_recall()
    md.metrics_precision()
    md.metrics_roc_auc()
    md.confusion_matrics()
    dataset_train = md.train_dataset(dataset=dataset_train)
    md.metrics_ks(dataset=dataset_train, type='train', variable_period='dt_month')
    md.metrics_ks(dataset=dataset_train, type='train', variable_period='ALL')
    dataset_test = md.test_dataset(dataset=dataset_test)
    md.metrics_ks(dataset=dataset_test, type='test',variable_period='dt_month')
    md.metrics_ks(dataset=dataset_test, type='test', variable_period='ALL')
    dataset_out = md.out_dataset(dataset=dataset_out)
    md.metrics_ks(dataset=dataset_out, type='out', variable_period='dt_month')
    md.metrics_ks(dataset=dataset_out, type='out', variable_period='ALL')
    md.feature_importance(model=model, variables_names=X_cols, type='dict')
    md.feature_importance(model=model, variables_names=X_cols, type='figure')
    mlflow.sklearn.log_model(model, 'model')
    mlflow.end_run()

