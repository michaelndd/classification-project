# Prepare the environment

from env import host, user, password

import numpy as np
import pandas as pd

from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score

import matplotlib.pyplot as plt
import seaborn as sns

from pandas.tools.plotting import scatter_matrix
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter


# Acquisition

def get_db_url(
    hostname: str, username: str, password: str, db_name: str
) -> str:
    """
    return url for accessing a mysql database
    """
    return f"mysql+pymysql://{username}:{password}@{hostname}/{db_name}"

def get_sql_conn(hostname: str, username: str, password: str, db_name: str):
    """
    return a mysql connection object
    """
    return create_engine(get_db_url(host, user, password, db_name))

def df_from_sql(query: str, url: str) -> pd.DataFrame:
    """
    return a Pandas DataFrame resulting from a sql query
    """
    return pd.read_sql(query, url)

def get_telco_data() -> pd.DataFrame:
    db = "telco_churn"
    query = ("SELECT * "
             f"FROM customers;")
    url = get_db_url(host, user, password, db)
    return df_from_sql(query, url)

# Data Preparation

def replace_empty(df):
	df['total_charges'] = np.where(df['total_charges'] == ' ',
								  (df.monthly_charges * df.tenure),
								  df['total_charges'])
	return df

def transform_churn(df):
	df['churn'] = df['churn'].map( {'No': 0, 'Yes': 1} ).astype(int)
	return df

def convert_tenure(df):
	df['tenure_year'] = df.tenure / 12
	return df

def add_phone_id(df):
    '''
    return DataFrame with a new colum phone_id that combines phone_service and multiple_lines as follows:
    # 0 = no phone service
    # 1 = One line
    # 2 = Two+ lines
    '''
    df_temp = df.copy()
    df_temp.loc[(df_temp.phone_service == 'No'), 'phone_id'] = 0
    df_temp.loc[(df_temp.phone_service == 'Yes') & (df_temp.multiple_lines == 'No'), 'phone_id'] = 1
    df_temp.loc[(df_temp.phone_service == 'Yes') & (df_temp.multiple_lines == 'Yes'), 'phone_id'] = 2
    df_temp = df_temp.astype({'phone_id': int})
    return df_temp

def add_household_type_id(df):
    '''
    return DataFrame with a new colum household_type_id that combines partner and dependents as follows:
    # 0 = no partner, no dependents
    # 1 = has partner, no dependents
    # 2 = no partner, has dependents
    # 3 = has partner, has dependents
    '''
    df_temp = df.copy()
    df_temp.loc[(df_temp.partner == 'No')& (df_temp.dependents == 'No'), 'household_type_id'] = 0
    df_temp.loc[(df_temp.partner == 'Yes') & (df_temp.dependents == 'No'), 'household_type_id'] = 1
    df_temp.loc[(df_temp.partner == 'No') & (df_temp.dependents == 'Yes'), 'household_type_id'] = 2
    df_temp.loc[(df_temp.partner == 'Yes') & (df_temp.dependents == 'Yes'), 'household_type_id'] = 3
    df_temp = df_temp.astype({'household_type_id': int})
    return df_temp

def add_streaming_services(df):
    """
    return DataFrame with a new column streaming_services that combines streaming_tv and streaming_movies columns as follows:
    # 0 = no internet service
    # 1 = no streaming_tv, no streaming_movies
    # 2 = has streaming_tv, no streaming_movies
    # 3 = no streaming_tv, has streaming_movies
    # 4 = has streaming_tv, has streaming_movies
    """
    df_temp = df.copy()
    df_temp.loc[(df_temp.streaming_tv == "No internet service") & (df_temp.streaming_movies == 'No internet service'), "streaming_services"] = int(0)
    df_temp.loc[(df_temp.streaming_tv == "No") & (df_temp.streaming_movies == 'No'), "streaming_services"] = int(1)
    df_temp.loc[(df_temp.streaming_tv == "Yes") & (df_temp.streaming_movies == 'No'), "streaming_services"] = int(2)
    df_temp.loc[(df_temp.streaming_tv == "No") & (df_temp.streaming_movies == 'Yes'), "streaming_services"] = int(3)
    df_temp.loc[(df_temp.streaming_tv == "Yes") & (df_temp.streaming_movies == 'Yes'), "streaming_services"] = int(4)
    df_temp = df_temp.astype({"streaming_services": int})
    return df_temp

def add_online_security_backup(df):
    """
    return DataFrame with a new column streaming_services that combines online_security and online_backup columns as follows:
    # 0 = no internet service
    # 1 = no online_security, no online_backup
    # 2 = has online_security, no online_backup
    # 3 = no online_security, has online_backup
    # 4 = has online_security, has online_backup
    """
    df_temp = df.copy()
    df_temp.loc[(df_temp.online_security == "No internet service") & (df_temp.online_backup == "No internet service"), "online_security_backup"] = 0
    df_temp.loc[(df_temp.online_security == "No") & (df_temp.online_backup == "No"), "online_security_backup"] = 1
    df_temp.loc[(df_temp.online_security == "Yes") & (df_temp.online_backup == "No"), "online_security_backup"] = 2
    df_temp.loc[(df_temp.online_security == "No") & (df_temp.online_backup == "Yes"), "online_security_backup"] = 3
    df_temp.loc[(df_temp.online_security == "Yes") & (df_temp.online_backup == "Yes"), "online_security_backup"] = 4
    df_temp = df_temp.astype({"online_security_backup": int})
    return df_temp

def fix_internet_service_type_id(df):
    """
    0 = no internet
    1 = DSL
    2 = fiber
    """
    df_temp = df.replace({"internet_service_type_id": 3}, 0)
    return df_temp

def predict_churn(input_df):
	'''
	return a DataFrame that predicts churn.
	'''
	df = input_df
	df = replace_empty(df)
	df = transform_churn(df)
	df = convert_tenure(df)
	df_sql = add_phone_id(df)
	df_sql = add_household_type_id(df_sql)
	df_sql = add_streaming_services(df_sql)
	df_sql = add_online_security_backup(df_sql)
	df_sql = fix_internet_service_type_id(df_sql)

	# Split the data
	xcols = ['gender', 'senior_citizen', 'internet_service_type_id', 'device_protection', 'tech_support',
	         'contract_type_id', 'paperless_billing', 'payment_type_id', 'monthly_charges', 'total_charges',
	         'tenure_year', 'phone_id', 'household_type_id', 'streaming_services', 'online_security_backup']
	X = df_sql[xcols]
	y = df_sql[['churn']]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 123, stratify=y)

	# Encode
	to_encode = ["gender", "device_protection", "tech_support", "paperless_billing"]
	encoders = {}
	for col in to_encode:
	    encoder = LabelEncoder()
	    encoder.fit(X_train[col])
	    X_train[col] = encoder.transform(X_train[col])
	    X_test[col] = encoder.transform(X_test[col])
	    encoders[col] = encoder

	# MinMax Scale
	scaler = MinMaxScaler()
	scaler.fit(X_train[["monthly_charges", "total_charges"]])
	X_train[["monthly_charges", "total_charges"]] = scaler.transform(X_train[["monthly_charges", "total_charges"]])
	X_test[["monthly_charges", "total_charges"]] = scaler.transform(X_test[["monthly_charges", "total_charges"]])

	# Make train_df and test_df
	train_df = pd.concat([X_train, y_train], axis=1)
	test_df = pd.concat([X_test, y_test], axis=1)




















predict_churn(get_telco_data())