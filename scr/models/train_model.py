import os
import mlflow
import optuna
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark.conf import SparkConf
import pyspark.sql.functions as f
from utils.utils import dataset_description
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pylab as plt
import seaborn as sns
from lightgbm import LGBMClassifier, plot_importance 


def objective(trial):
    
    # Invoke suggest methods of a Trial object to generate hyperparameters.
    regressor_name = trial.suggest_categorical('classifier', ['RandomForest', 'lightGBM'])
    if regressor_name == 'RandomForest':
        max_depth = trial.suggest_int(name='max_depth', low=2, high=5)
        max_leaf_nodes = trial.suggest_int(name='max_leaf_nodes', low=10, high=40)
        min_impurity_decrease = trial.suggest_float(name='impurity', low=0.0, high=1.0, step=0.1)
        min_samples_leaf = trial.suggest_int(name='min_samples_leaf', low=10, high=60)
        n_estimators =  trial.suggest_int(name='n_estimators', low=30, high=50)
        model = RandomForestClassifier(random_state=7, class_weight='balanced_subsample',
            max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
            min_samples_leaf=min_samples_leaf, n_estimators=n_estimators)
    elif regressor_name == 'lightGBM':
        objective = trial.suggest_categorical('objective', ['binary'])
        n_jobs = trial.suggest_categorical('n_jobs', [-1])
        class_weight = trial.suggest_categorical('class_weight', ['balanced'])
        metric = trial.suggest_categorical('metric', ['binary_logloss', 'cress_entropy', 'auc', 'avarege_precision'])
        lambda_l1 = trial.suggest_float(name='impurity', low=0.0, high=1e-8, step=3)
        lambda_l2 = trial.suggest_float(name='impurity', low=0.0, high=1e-8, step=3)
        max_depth = trial.suggest_int(name='max_depth', low=2, high=10)
        n_estimators =  trial.suggest_int(name='n_estimators', low=5, high=150)
        min_child_samples = trial.suggest_int(name='min_child_samples', low=50, high=15000)
        model = lgb.LGBMClassifier(random_state=7, objective=objective, n_jobs=n_jobs,
                        class_weight=class_weight, metric=metric, lambda_l1=lambda_l1,
                        lambda_l2=lambda_l2, max_depth=max_depth, n_estimators=n_estimators,
                        min_child_samples=min_child_samples)

    model.fit(X_train, y_train)
    pred = model.predict(X_train)
    metric = average_precision_score(y_train, pred)

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

# Variables IB
tb_iv = pd.read_csv(os.path.join(DIR_PATH, 'iv_tb.csv'))
tb_iv = tb_iv[tb_iv['iv']<0.54]
X_cols = tb_iv['name'].to_list()

# Creating Datasets
dataset = dataset.toPandas().fillna(value=0)
out_of_time = [1,3,5,7,9,11]
training_dates = [2,4,6,8,12]
train = dataset[dataset['dt_month'].isin(training_dates)]
out_of_time = dataset[dataset['dt_month'].isin(out_of_time)]

# Defining Variables Columns
ident_cols = ['id_cnpj', 'id_cvm', 'txt_company_name', 'ticker',  'id_isin', 'processed_at', 'date',  'dt_refer', 'dt_year', 'dt_quarter', 'dt_month',
              'dt_fim_exerc', 'dt_ini_exerc', 'dt_refer_lag', 'cat_target', 'var_target', 'amplitude_close_high_price']

y_col= 'cat_target'
X_cols = [col for col in X_cols if col not in ident_cols]
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=7)

# Getting Index Rows
dataset_train = id.loc[X_train.index, :]
dataset_test = id.loc[X_test.index, :]

# Hyperparameters
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=3)
best_params = study.best_params
best_params



# Get Predictions
model = lgb.LGBMClassifier(random_state=7, **best_params)
# Fitting
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_train_prob = model.predict_proba(X_train)[:, 1]

y_test_pred = model.predict(X_test)
y_test_prob = model.predict_proba(X_test)[:, 1]

y_out_pred = model.predict(X_out)
y_out_prob = model.predict_proba(X_out)[:, 1]


# Creating Datasets with Predictions
dataset_train['prediction'] = y_train_pred
dataset_train['prob'] = y_train_prob

dataset_test['prediction'] = y_test_pred
dataset_test['prob'] = y_test_prob

dataset_out['prediction'] = y_out_pred
dataset_out['prob'] = y_out_prob


# Analyzing test dataset results
print(confusion_matrix(y_train, y_train_pred))
print(classification_report(y_train, y_train_pred))

# Analyzing test dataset results
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

# Analyzing out-of-time dataset results
print(confusion_matrix(y_out, y_out_pred))
print(classification_report(y_out, y_out_pred))


# Feature Importance
importances = model.feature_importances_
variables_names = X_cols

dataframe = {'variables_names':variables_names, 'importances':importances}
dataframe = pd.DataFrame(dataframe)
dataframe.sort_values(by='importances', ascending=False,inplace=True)

plt.figure(figsize=(10,20))
sns.barplot(x=dataframe['importances'], y=dataframe['variables_names'][:20])
plt.title('Feature Importance')
plt.xlabel('Importances')
plt.ylabel('Variables')
plt.show()


# Plot ROC Curve

# ROC-Train
from plot_metric.functions import BinaryClassification
from optbinning.scorecard import plot_auc_roc, plot_cap, plot_ks

bc = BinaryClassification(y_train, y_train_pred, labels=["Class 1", "Class 2"])
plt.figure(figsize=(5,5))
bc.plot_roc_curve(title='ROC-Train')
plt.show()

# optbinning
plot_auc_roc(y_train, y_train_pred, title='ROC-Train')
plt.show()


# ROC-Test
bc = BinaryClassification(y_test, y_test_pred, labels=["Class 1", "Class 2"])
plt.figure(figsize=(5,5))
bc.plot_roc_curve(title='ROC-Test')
plt.show()

# optbinning
plot_auc_roc(y_test, y_test_pred, title='ROC-Test')
plt.show()


# =============================================================================
# Plot KS Curve
# =============================================================================
# import scikitplot as skplt

import scikitplot as skplt

# Train
skplt.metrics.plot_ks_statistic(y_train, model.predict_proba(X_train), title='KS-Train')
plt.show()

# optbinning
plot_ks(y_train, y_train_pred, title='ROC-Train')
plt.show()


# Test
skplt.metrics.plot_ks_statistic(y_test, model.predict_proba(X_test), title='KS-Test')
plt.show()


# optbinning
plot_ks(y_test, y_test_pred, title='ROC-Test')
plt.show()
