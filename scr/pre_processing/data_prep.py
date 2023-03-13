import os
#import boto3
#from dotenv import load_dotenv
from pyspark.sql.window import Window
from utils.utils import download_bucket_s3, s3_connection
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark.conf import SparkConf
import pyspark.sql.functions as f 
from utils.utils import get_percentage_change
from utils.documents import all_varlist


#load_dotenv()
DIR_LOCAL = os.path.dirname(os.path.realpath('__file__'))
DIR_PATH = os.path.join(DIR_LOCAL, 'data')

#spark = SparkSession.builder.getOrCreate()
config = SparkConf().setAll([('spark.executor.memory', '40g'), ('spark.executor.cores', '16'), ('spark.cores.max', '16'), ('spark.driver.memory', '30g')])
sc = SparkContext(conf=config)
spark = SparkSession(sc)

'''
def download_bucket_s3(s3, bucket, path):
    
    import os
    import boto3

    def _get_objects(s3, response, bucket, path):
        for i in response['Contents']:
            source = i['Key']
            destination = os.path.join(path, source)
            if not os.path.exists(os.path.dirname(destination)):
                os.makedirs(os.path.dirname(destination))
            s3.download_file(Bucket=bucket,Key=source, Filename=destination)

    
    response = s3.list_objects_v2(Bucket=bucket)
    _get_objects(s3=s3,response=response, bucket=bucket, path=path)
    while response.get('IsTruncated'):
        TOKEN = response.get('NextContinuationToken')
        response = s3.list_objects_v2(Bucket=bucket, ContinuationToken=TOKEN)
        _get_objects(s3=s3, response=response, bucket=bucket, path=path)

import boto3
s3 = boto3.client(
    service_name='s3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# Updating Analytical Files
download_bucket_s3(s3=s3, bucket='fundamentus-analytical', path=DIR_PATH)

df_stock.agg(f.max(f.col('date'))).show()
'''

# Datasets
filename = 'analytical_stock_price.parquet'
df_stock = spark.read.parquet(os.path.join(DIR_PATH, filename)).orderBy(['id_cnpj', 'date'])

filename = 'analytical_dre.parquet'
df_dre = spark.read.parquet(os.path.join(DIR_PATH, filename)).orderBy(['id_cnpj', 'dt_refer'])
#df_dre.toPandas().to_csv(os.path.join(DIR_PATH, 'dataset_dre.csv'), index=False)

df_stock = (
    df_stock
    .dropDuplicates(subset=['date', 'id_cnpj', 'ticker'])
    .withColumn('dt_year', f.year(f.col('date')))
    .withColumn('dt_month', f.month(f.col('date')))
    .withColumn('dt_date_lag', f.date_sub(f.col('date'), 90))
    .withColumn('dt_year_lag', f.year(f.col('dt_date_lag')))
    .withColumn('dt_month_lag', f.month(f.col('dt_date_lag')))
    .withColumn('dt_safra_dre', f.col('dt_year_lag')*100+f.col('dt_month_lag'))
    .filter(f.col('id_cnpj') != '00000000000000')
    .filter(f.col('date')>'2011-03-31')
    .orderBy('dt_date_lag')
)

df_dre = (
    df_dre
    .filter(df_dre['dt_refer'].isNotNull())
    .withColumn('dt_month', f.month(f.col('dt_refer')))
    .withColumn('dt_safra_dre', f.col('dt_year')*100+f.col('dt_month'))
    .dropDuplicates(subset=['id_cnpj', 'dt_refer'])
    .drop('dt_year', 'dt_quarter', 'dt_month')
    .orderBy('dt_refer')
)

df = (
    df_stock
    .join(df_dre, on=(df_stock.id_cnpj==df_dre.id_cnpj)&(df_stock.dt_safra_dre>=df_dre.dt_safra_dre), how='left')
    .withColumn('target_lag', f.lag(f.col('adj_close'), 1).over(Window.partitionBy(df_stock.id_cnpj, 'ticker').orderBy('date')))
    .dropDuplicates(subset=['date', 'id_isin', 'id_cnpj', 'ticker', 'adj_close'])
    .drop(df_dre.id_cnpj)
    .drop(df_dre.dt_safra_dre)
    .drop('dt_date_lag', 'dt_year_lag', 'dt_month_lag')
)

# Defining Target
df = get_percentage_change(dataset=df, partition_var=['id_cnpj', 'ticker'], order_var='date', variable='target_lag', nlag=1, variable_name='var_target')

df = (
    df
    .withColumn('dt_month', f.month(f.col('date')))
    .withColumn('cat_target', f.when(f.col('var_target')>=0, 1)
    .when(f.col('var_target')<0, 0))
    .filter(f.col('cat_target').isNotNull())
    .drop('target_lag')
    .select(all_varlist)
)

# Remove Financial Informartions
df = (
    df
    .filter(f.col('txt_company_name').isNotNull())
)

# Saving
df.write.mode('overwrite').parquet(os.path.join(DIR_PATH, 'dataset_model.parquet'))

'''
var_test = ['id_cnpj', 'id_cvm', 'txt_company_name', 'ticker',  'id_isin', 'processed_at', 'date',  'dt_refer', 'dt_year', 'dt_month',
'dt_fim_exerc', 'dt_ini_exerc', 'dt_safra_dre', 'cat_target', 'var_target']
df.select(var_test).toPandas().to_csv(os.path.join(DIR_PATH, 'dataset_model.csv'), index=False)
'''
