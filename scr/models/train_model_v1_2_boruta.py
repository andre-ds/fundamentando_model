import os
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from boruta import BorutaPy
from sklearn.metrics import roc_auc_score
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark.conf import SparkConf
import pyspark.sql.functions as f
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from utils.utils import Modeling
from utils.utils import dataset_description


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
        max_depth = trial.suggest_int(name='max_depth', low=2, high=5)
        min_child_samples = trial.suggest_int(name='min_child_samples', low=1000, high=25000)
        model = lgb.LGBMClassifier(random_state=7, importance_type=importance_type, boosting_type=boosting_type,
                        objective=objective, n_jobs=n_jobs, class_weight=class_weight, metric=metric, reg_lambda=reg_lambda,
                        reg_alpha=reg_alpha, max_depth=max_depth, n_estimators=n_estimators,
                        min_child_samples=min_child_samples)

    model.fit(boruta.X_train, boruta.y_train)
    pred = model.predict(boruta.X_train)
    metric = roc_auc_score(boruta.y_train, pred)

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


# Boruta
best_params = {'importance_type': 'gain', 'boosting_type': 'gbdt', 'n_estimators': 1000, 'objective': 'binary', 'n_jobs': -1, 'class_weight': 'balanced', 'metric': 'cress_entropy', 'reg_alpha': 0.5750000000000001, 'reg_lambda': 0.55, 'max_depth': 10, 'min_child_samples': 1005}
model = lgb.LGBMClassifier(random_state=7, **best_params)
feature_selector = BorutaPy(model, n_estimators='auto', verbose=2, random_state=1)
feature_selector.fit(md.X_train.values, md.y_train)

feature_df = md.boruta_features(dataset=md.X_train, feature_selector=feature_selector)
boruta_variables = feature_df[feature_df['support']==True]['features'].tolist()



# Boruta Selection Training
X_b = train[boruta_variables]
y_b = train[y_col].values
id = train[ident_cols]

X_out_b = out_of_time[boruta_variables]
y_out_b = out_of_time[y_col].values
dataset_out = out_of_time[ident_cols]

# Spliting Train
boruta = Modeling(X=X_b, y=y_b, X_out=X_out_b, y_out=y_out_b)
boruta._split_dataset()

# Getting Index Rows
dataset_train = id.loc[boruta.X_train.index, :]
dataset_test = id.loc[boruta.X_test.index, :]


# Hyperparameters Tunning
mlflow.set_experiment(experiment_name='boruta-selection')
with mlflow.start_run(run_name='hyperparameters-tunning-boruta'):
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=500)
    best_params = study.best_params
    best_params.pop('classifier')
    mlflow.log_params(params=best_params)
    mlflow.sklearn.log_model(study, 'optuna')
    mlflow.end_run()


with mlflow.start_run():
    mlflow.log_params(params=best_params)
    boruta_model = lgb.LGBMClassifier(random_state=7, **best_params)
    boruta_model.fit(boruta.X_train, boruta.y_train)
    
    # Log Metrics
    boruta.prediction_train(model=boruta_model)
    boruta.prediction_test(model=boruta_model)
    boruta.prediction_out(model=boruta_model)
    boruta.metrics_recall()
    boruta.metrics_precision()
    boruta.metrics_roc_auc()
    boruta.confusion_matrics()
    dataset_train = boruta.train_dataset(dataset=dataset_train)
    boruta.metrics_ks(dataset=dataset_train, type='train', variable_period='dt_month')
    boruta.metrics_ks(dataset=dataset_train, type='train', variable_period='ALL')
    dataset_test = boruta.test_dataset(dataset=dataset_test)
    boruta.metrics_ks(dataset=dataset_test, type='test',variable_period='dt_month')
    boruta.metrics_ks(dataset=dataset_test, type='test', variable_period='ALL')
    dataset_out = boruta.out_dataset(dataset=dataset_out)
    boruta.metrics_ks(dataset=dataset_out, type='out', variable_period='dt_month')
    boruta.metrics_ks(dataset=dataset_out, type='out', variable_period='ALL')
    boruta.feature_importance(model=boruta_model, variables_names=boruta_variables, type='dict')
    boruta.feature_importance(model=boruta_model, variables_names=boruta_variables, type='figure')
    mlflow.sklearn.log_model(boruta_model, 'boruta_model')
    mlflow.end_run()



