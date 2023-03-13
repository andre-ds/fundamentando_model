from pyspark.sql.dataframe import DataFrame


class Modeling():

    def __init__(self, X=None, y=None, X_out=None, y_out=None):
        self.X = X
        self.y = y
        self.X_out = X_out
        self.y_out = y_out


    def _split_dataset(self, test_size=0.3):

        from sklearn.model_selection import train_test_split    
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=7)

    
    def ks_by_date_ref(self, dataset, variable_period, variable_target_categorical, variable_target_prob):

        from scipy import stats
        
        KS_period = {} 
        for i in dataset[variable_period].unique():
            df = dataset[dataset[variable_period]==i]
            ks = stats.kstest(df[df[variable_target_categorical]==1][variable_target_prob], df[df[variable_target_categorical]==0][variable_target_prob])
            ks = round(ks[0], 4)
            KS_period[f'period_ref_{i}'] = ks

        return KS_period
        

    def prediction_train(self, model):

        self.y_train_pred = model.predict(self.X_train)
        self.y_train_prob = model.predict_proba(self.X_train)[:, 1]


    def train_dataset(self, dataset):
        
        dataset['prediction'] = self.y_train_pred
        dataset['prob'] = self.y_train_prob

        return dataset


    def prediction_test(self, model):
        
        self.y_test_pred = model.predict(self.X_test)
        self.y_test_prob = model.predict_proba(self.X_test)[:, 1]


    def test_dataset(self, dataset):    
        
        dataset['prediction'] = self.y_test_pred
        dataset['prob'] = self.y_test_prob

        return dataset


    def prediction_out(self, model):
        
        self.y_out_pred = model.predict(self.X_out)
        self.y_out_prob = model.predict_proba(self.X_out)[:, 1]


    def out_dataset(self, dataset):    
        
        dataset['prediction'] = self.y_out_pred
        dataset['prob'] = self.y_out_prob

        return dataset


    def confusion_matrics(self):
        
        import mlflow
        from sklearn.metrics import confusion_matrix

        if self.y_train_pred is not None:
            cm = confusion_matrix(self.y_train, self.y_train_pred)
            cm = {
                    '00':str(cm[0][0]),
                    '01':str(cm[0][1]),
                    '10':str(cm[1][0]),
                    '11':str(cm[1][1]),
                    }
            mlflow.log_dict(dictionary=cm, artifact_file='train_cm.json')
        if self.y_test_pred is not None:
            cm = confusion_matrix(self.y_train, self.y_train_pred)
            cm = {
                    '00':str(cm[0][0]),
                    '01':str(cm[0][1]),
                    '10':str(cm[1][0]),
                    '11':str(cm[1][1]),
                    }
            mlflow.log_dict(dictionary=cm, artifact_file='test_cm.json')
        if self.y_out_pred is not None:
            cm = confusion_matrix(self.y_train, self.y_train_pred)
            cm = {
                    '00':str(cm[0][0]),
                    '01':str(cm[0][1]),
                    '10':str(cm[1][0]),
                    '11':str(cm[1][1]),
                    }
            mlflow.log_dict(dictionary=cm, artifact_file='out_cm.json')


    def metrics_recall(self):

        import mlflow
        from sklearn.metrics import recall_score

        if self.y_train_pred is not None:
            train = recall_score(self.y_train, self.y_train_pred)
            mlflow.log_metric(key=f'train_recall', value=train)
        if self.y_test_pred is not None:
            test= recall_score(self.y_test, self.y_test_pred)
            mlflow.log_metric(key=f'test_recall', value=test)
        if self.y_out_pred is not None:
            out = recall_score(self.y_out, self.y_out_pred)
            mlflow.log_metric(key=f'out_recall', value=out)


    def metrics_precision(self):

        import mlflow
        from sklearn.metrics import precision_score

        if self.y_train_pred is not None:
            train = precision_score(self.y_train, self.y_train_pred)
            mlflow.log_metric(key=f'train_precision', value=train)
        if self.y_test_pred is not None:
            test= precision_score(self.y_test, self.y_test_pred)
            mlflow.log_metric(key=f'test_precision', value=test)
        if self.y_out_pred is not None:
            out = precision_score(self.y_out, self.y_out_pred)
            mlflow.log_metric(key=f'out_precision', value=out)


    def metrics_roc_auc(self):
        
        import mlflow
        from sklearn.metrics import roc_auc_score

        if self.y_train_pred is not None:
            train = roc_auc_score(self.y_train, self.y_train_pred)
            mlflow.log_metric(key=f'train_roc', value=train)
        if self.y_test_pred is not None:
            test= roc_auc_score(self.y_test, self.y_test_pred)
            mlflow.log_metric(key=f'test_roc', value=test)
        if self.y_out_pred is not None:
            out = roc_auc_score(self.y_out, self.y_out_pred)
            mlflow.log_metric(key=f'out_roc', value=out)


    def metrics_ks(self, dataset, type, variable_target_categorical='cat_target', variable_target_prob='prob', variable_period='ALL'):

        import mlflow

        if variable_period != 'ALL':
            if type=='train':
                train = self.ks_by_date_ref(dataset, variable_period, variable_target_categorical, variable_target_prob)
                mlflow.log_dict(dictionary=train, artifact_file='train_bydate_ks.json')
            elif type=='test':
                test = self.ks_by_date_ref(dataset, variable_period, variable_target_categorical, variable_target_prob)
                mlflow.log_dict(dictionary=test, artifact_file='test_bydate_ks.json')
            elif type=='out':
                out = self.ks_by_date_ref(dataset, variable_period, variable_target_categorical, variable_target_prob)
                mlflow.log_dict(dictionary=out, artifact_file='out_bydate_ks.json')
        else:
            from scipy import stats
            if type=='train':
                train = stats.kstest(dataset[dataset[variable_target_categorical]==1][variable_target_prob], dataset[dataset[variable_target_categorical]==0][variable_target_prob])
                train = round(train[0], 4)
                mlflow.log_metric(key='train_ks', value=train)
            elif type=='test':
                test = stats.kstest(dataset[dataset[variable_target_categorical]==1][variable_target_prob], dataset[dataset[variable_target_categorical]==0][variable_target_prob])
                test = round(test[0], 4)
                mlflow.log_metric(key='test_ks', value=test)
            elif type=='out':
                out = stats.kstest(dataset[dataset[variable_target_categorical]==1][variable_target_prob], dataset[dataset[variable_target_categorical]==0][variable_target_prob])
                out = round(out[0], 4)
                mlflow.log_metric(key='out_ks', value=out)


    def feature_importance(self, model, variables_names, type, n_variables=None):
        
        import mlflow
        importances = model.feature_importances_

        if type == 'dict':
            
            feature_importance = dict(zip(variables_names, importances))
            for i in feature_importance.keys():
                feature_importance.update({i:round(feature_importance.get(i), 2)})

            feature_importance = dict(reversed(sorted(feature_importance.items(), key=lambda item: item[1])))
            mlflow.log_dict(feature_importance, 'feature_importance.json')

        if type == 'figure':
            import pandas as pd
            import matplotlib.pylab as plt
            import seaborn as sns

            # Creating Dataframe
            dataframe = {'variables_names':variables_names, 'importances':importances}
            dataframe = pd.DataFrame(dataframe)
            dataframe.sort_values(by='importances', ascending=False,inplace=True)
            # Plotting
            fig = plt.figure(figsize=(10,8))
            sns.barplot(x=dataframe['importances'], y=dataframe['variables_names'][:n_variables])
            plt.title('Feature Importance')
            plt.xlabel('Importances')
            plt.ylabel('Variables')
            mlflow.log_figure(fig, 'feature_importance.png')


    def boruta_features(self, dataset, feature_selector):

        import pandas as pd

        feature_df = pd.DataFrame(dataset.columns.tolist(), columns=['features'])
        feature_df['rank'] = feature_selector.ranking_
        feature_df['support'] = feature_selector.support_
        feature_df.sort_values(by='rank', inplace=True)

        return feature_df


# Pandas
def dataset_description(dataset):
    import pandas as pd
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    dataset = pd.DataFrame({'Tipo': dataset.dtypes,
                    'Quantidade_Nan': dataset.isna().sum(),
                    'Percentual_Nan': (dataset.isna().sum() / dataset.shape[0]) * 100,
                    'Valores_Unicos': dataset.nunique()})
    return dataset

# Pyspark
def get_percentage_change(dataset:DataFrame, partition_var:list, order_var:list, variable:str, nlag:int, variable_name:str=None):

    from pyspark.sql.window import Window
    import pyspark.sql.functions as f

    windowSpec  = Window.partitionBy(partition_var).orderBy(order_var)
    
    dataset = (
        dataset
        .withColumn('var_lag',f.lag(variable, nlag).over(windowSpec))
        .withColumn(variable_name, (f.col(variable)-f.col('var_lag'))/f.abs(f.col('var_lag')))
        .drop('var_lag')
        )

    return dataset




['date', 'id_isin', 'id_cnpj', 'ticker', 'adj_close', 'close', 'dividends', 'high_price', 'low_price', 'open', 'stock_splits',
'volume', 'dt_year', 'dt_quarter', 'amplitude_close_high_price', 'amt_avg_adj_close_5', 'amt_std_adj_close_5',
'amt_min_adj_close_5', 'amt_max_adj_close_5', 'amt_avg_adj_close_10', 'amt_std_adj_close_10', 'amt_min_adj_close_10',
'amt_max_adj_close_10', 'amt_avg_adj_close_15', 'amt_std_adj_close_15', 'amt_min_adj_close_15', 'amt_max_adj_close_15',
'amt_avg_adj_close_20', 'amt_std_adj_close_20', 'amt_min_adj_close_20', 'amt_max_adj_close_20', 'amt_avg_adj_close_25',
'amt_std_adj_close_25', 'amt_min_adj_close_25', 'amt_max_adj_close_25', 'amt_avg_adj_close_30', 'amt_std_adj_close_30',
'amt_min_adj_close_30', 'amt_max_adj_close_30', 'amt_avg_adj_close_60', 'amt_std_adj_close_60', 'amt_min_adj_close_60',
'amt_max_adj_close_60', 'amt_avg_adj_close_90', 'amt_std_adj_close_90', 'amt_min_adj_close_90', 'amt_max_adj_close_90',
'amt_avg_adj_close_120', 'amt_std_adj_close_120', 'amt_min_adj_close_120', 'amt_max_adj_close_120', 'amt_avg_adj_close_160',
'amt_std_adj_close_160', 'amt_min_adj_close_160', 'amt_max_adj_close_160', 'amt_avg_adj_close_180', 'amt_std_adj_close_180',
'amt_min_adj_close_180', 'amt_max_adj_close_180', 'amt_avg_adj_close_360', 'amt_std_adj_close_360', 'amt_min_adj_close_360',
'amt_max_adj_close_360', 'pct_tx_adj_close_5d', 'pct_tx_adj_close_10d', 'pct_tx_adj_close_15d', 'pct_tx_adj_close_20d',
'pct_tx_adj_close_25d', 'pct_tx_adj_close_30d', 'pct_tx_adj_close_60d', 'pct_tx_adj_close_90d', 'pct_tx_adj_close_120d',
'pct_tx_adj_close_160d', 'pct_tx_adj_close_180d', 'pct_tx_adj_close_360d', 'pct_tx_high_price_5d', 'pct_tx_high_price_10d',
'pct_tx_high_price_15d', 'pct_tx_high_price_20d', 'pct_tx_high_price_25d', 'pct_tx_high_price_30d', 'pct_tx_high_price_60d',
'pct_tx_high_price_90d', 'pct_tx_high_price_120d', 'pct_tx_high_price_160d', 'pct_tx_high_price_180d', 'pct_tx_high_price_360d',
'pct_tx_low_price_5d', 'pct_tx_low_price_10d', 'pct_tx_low_price_15d', 'pct_tx_low_price_20d', 'pct_tx_low_price_25d',
'pct_tx_low_price_30d', 'pct_tx_low_price_60d', 'pct_tx_low_price_90d', 'pct_tx_low_price_120d', 'pct_tx_low_price_160d',
'pct_tx_low_price_180d', 'pct_tx_low_price_360d', 'pct_tx_amplitude_close_high_price_5d', 'pct_tx_amplitude_close_high_price_10d',
'pct_tx_amplitude_close_high_price_15d', 'pct_tx_amplitude_close_high_price_20d', 'pct_tx_amplitude_close_high_price_25d',
'pct_tx_amplitude_close_high_price_30d', 'pct_tx_amplitude_close_high_price_60d', 'pct_tx_amplitude_close_high_price_90d',
'pct_tx_amplitude_close_high_price_120d', 'pct_tx_amplitude_close_high_price_160d', 'pct_tx_amplitude_close_high_price_180d',
'pct_tx_amplitude_close_high_price_360d', 'amt_beta_adj_close_11', 'amt_beta_adj_close_16', 'amt_beta_adj_close_21',
'amt_beta_adj_close_26', 'amt_beta_adj_close_31', 'amt_beta_adj_close_61', 'amt_beta_adj_close_91', 'amt_beta_adj_close_121',
'amt_beta_adj_close_161', 'amt_beta_adj_close_181', 'amt_beta_adj_close_361', 'cat_adj_close_broke_high_5',
'cat_adj_close_broke_high_10', 'cat_adj_close_broke_high_15', 'cat_adj_close_broke_high_20', 'cat_adj_close_broke_high_25',
'cat_adj_close_broke_high_30', 'cat_adj_close_broke_high_60', 'cat_adj_close_broke_high_90', 'cat_adj_close_broke_high_120',
'cat_high_price_broke_high_5', 'cat_high_price_broke_high_10', 'cat_high_price_broke_high_15', 'cat_high_price_broke_high_20',
'cat_high_price_broke_high_25', 'cat_high_price_broke_high_30', 'cat_high_price_broke_high_60', 'cat_high_price_broke_high_90',
'cat_high_price_broke_high_120', 'cat_adj_close_broke_low_5', 'cat_low_price_broke_low_5', 'cat_adj_close_broke_low_10',
'cat_low_price_broke_low_10', 'cat_adj_close_broke_low_15', 'cat_low_price_broke_low_15', 'cat_adj_close_broke_low_20',
'cat_low_price_broke_low_20', 'cat_adj_close_broke_low_25', 'cat_low_price_broke_low_25', 'cat_adj_close_broke_low_30',
'cat_low_price_broke_low_30', 'cat_adj_close_broke_low_60', 'cat_low_price_broke_low_60', 'cat_adj_close_broke_low_90',
'cat_low_price_broke_low_90', 'cat_adj_close_broke_low_120', 'cat_low_price_broke_low_120', 'amt_IFR_adj_close_10',
'amt_IFR_adj_close_15', 'amt_IFR_adj_close_20', 'amt_IFR_adj_close_25', 'amt_IFR_adj_close_30', 'amt_IFR_adj_close_60',
'amt_IFR_adj_close_90', 'amt_IFR_adj_close_120', 'id_cvm', 'id', 'txt_company_name', 'dt_refer', 'dt_fim_exerc',
'dt_ini_exerc', 'amt_cost_goods_and_services', 'amt_earnings_before_income_tax_and_social_contribution',
'amt_earnings_before_interest_and_taxes', 'amt_financial_results', 'amt_groos_revenue', 'amt_net_profit',
'amt_operating_revenues_and_expenses', 'amt_sales_revenue', 'processed_at', 'amt_tot_amt_sales_revenue_1m_lag',
'amt_tot_amt_sales_revenue_1y_lag', 'amt_tot_amt_cost_goods_and_services_1m_lag', 'amt_tot_amt_cost_goods_and_services_1y_lag',
'amt_tot_amt_earnings_before_interest_and_taxes_1m_lag', 'amt_tot_amt_earnings_before_interest_and_taxes_1y_lag',
'amt_tot_amt_financial_results_1m_lag', 'amt_tot_amt_financial_results_1y_lag', 'amt_tot_amt_net_profit_1m_lag',
'amt_tot_amt_net_profit_1y_lag', 'pct_tx_amt_sales_revenue_1m', 'pct_tx_amt_sales_revenue_1y', 'pct_tx_amt_cost_goods_and_services_1m',
'pct_tx_amt_cost_goods_and_services_1y', 'pct_tx_amt_earnings_before_interest_and_taxes_1m', 'pct_tx_amt_earnings_before_interest_and_taxes_1y',
'pct_tx_amt_financial_results_1m', 'pct_tx_amt_financial_results_1y', 'pct_tx_amt_net_profit_1m', 'pct_tx_amt_net_profit_1y', 'amt_avg_amt_sales_revenue_1',
'amt_avg_amt_sales_revenue_2', 'amt_avg_amt_sales_revenue_3', 'amt_avg_amt_sales_revenue_4', 'amt_avg_amt_cost_goods_and_services_1',
'amt_avg_amt_cost_goods_and_services_2', 'amt_avg_amt_cost_goods_and_services_3', 'amt_avg_amt_cost_goods_and_services_4',
'amt_avg_amt_earnings_before_interest_and_taxes_1', 'amt_avg_amt_earnings_before_interest_and_taxes_2', 'amt_avg_amt_earnings_before_interest_and_taxes_3',
'amt_avg_amt_earnings_before_interest_and_taxes_4', 'amt_avg_amt_financial_results_1', 'amt_avg_amt_financial_results_2', 'amt_avg_amt_financial_results_3',
'amt_avg_amt_financial_results_4', 'amt_avg_amt_net_profit_1', 'amt_avg_amt_net_profit_2', 'amt_avg_amt_net_profit_3', 'amt_avg_amt_net_profit_4',
'amt_std_amt_sales_revenue_1', 'amt_std_amt_sales_revenue_2', 'amt_std_amt_sales_revenue_3', 'amt_std_amt_sales_revenue_4', 'amt_std_amt_cost_goods_and_services_1',
'amt_std_amt_cost_goods_and_services_2', 'amt_std_amt_cost_goods_and_services_3', 'amt_std_amt_cost_goods_and_services_4',
'amt_std_amt_earnings_before_interest_and_taxes_1', 'amt_std_amt_earnings_before_interest_and_taxes_2', 'amt_std_amt_earnings_before_interest_and_taxes_3',
'amt_std_amt_earnings_before_interest_and_taxes_4', 'amt_std_amt_financial_results_1', 'amt_std_amt_financial_results_2', 'amt_std_amt_financial_results_3', 
'amt_std_amt_financial_results_4', 'amt_std_amt_net_profit_1', 'amt_std_amt_net_profit_2', 'amt_std_amt_net_profit_3', 'amt_std_amt_net_profit_4',
'amt_beta_amt_sales_revenue_3', 'amt_beta_amt_cost_goods_and_services_3', 'amt_beta_amt_earnings_before_interest_and_taxes_3',
'amt_beta_amt_financial_results_3', 'amt_beta_amt_net_profit_3', 'amt_beta_amt_sales_revenue_6', 'amt_beta_amt_cost_goods_and_services_6',
'amt_beta_amt_earnings_before_interest_and_taxes_6', 'amt_beta_amt_financial_results_6', 'amt_beta_amt_net_profit_6', 'amt_beta_amt_sales_revenue_12',
'amt_beta_amt_cost_goods_and_services_12', 'amt_beta_amt_earnings_before_interest_and_taxes_12', 'amt_beta_amt_financial_results_12',
'amt_beta_amt_net_profit_12', 'dt_refer_lag', 'target', 'dt_month']