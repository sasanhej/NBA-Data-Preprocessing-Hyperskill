import pandas as pd
import os
import requests
import re
import numpy as np
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
# Checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# Download data if it is unavailable.
if 'nba2k-full.csv' not in os.listdir('../Data'):
    print('Train dataset loading.')
    url = "https://www.dropbox.com/s/wmgqf23ugn9sr3b/nba2k-full.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/nba2k-full.csv', 'wb').write(r.content)
    print('Loaded.')

data_path = "../Data/nba2k-full.csv"

# write your code here


def clean_data(path):
    df = pd.read_csv(path)
    df.b_day = pd.to_datetime(df.b_day, format='%m/%d/%y')
    df.draft_year = pd.to_datetime(df.draft_year, format='%Y')
    df.team = df.team.fillna('No Team')
    df.height = df.height.apply(lambda x: re.search(r'(?<=/ )\d*.\d*', x).group()).astype('float')
    df.weight = df.weight.apply(lambda x: re.search(r'(?<=/ )\d*.\d*', x).group()).astype('float')
    df.salary=df.salary.map(lambda x: x.strip('$')).astype('float')
    df.country = df.country.map(lambda x: x if x == 'USA' else 'Not-USA')
    df.draft_round = df.draft_round.map(lambda x: '0' if x == 'Undrafted' else x)
    return df


def feature_data(df):
    df.version = pd.to_datetime(df.version.map(lambda x: ('20'+re.search(r'(?<=k)\d*', x).group())))
    df['age'] = pd.DatetimeIndex(df.version).year-pd.DatetimeIndex(df.b_day).year
    df['experience'] = pd.DatetimeIndex(df.version).year-pd.DatetimeIndex(df.draft_year).year
    df['bmi'] = df.weight/(df.height**2).astype(float)
    df.drop(columns=['version', 'b_day', 'draft_year', 'weight', 'height'], inplace=True)
    for i in df.columns:
        if df[i].nunique() > 50 and df[i].dtype == object:
            df.drop(columns=[i], inplace=True)
    return df


def multicol_data(df, target='salary'):
    corrcols = df.describe().drop(columns=[target]).columns
    corrs = {}
    drop = {}
    for i in itertools.combinations(corrcols, 2):
        corrs[i] = abs(df[i[1]].corr(df[i[0]]))
    for j in max(corrs, key=corrs.get):
        drop[j] = abs(df[j].corr(df[target]))
    return df.drop(columns=[min(drop, key=drop.get)])


def transform_data(df, target='salary'):
    numerical = df.describe().drop(columns=target).columns
    scaler = StandardScaler()
    scaler.fit(df[numerical])
    df[numerical] = scaler.transform(df[numerical])
    categorical = df.drop(columns=target).drop(columns=numerical).columns
    encoder = OneHotEncoder()
    encoder.fit(df[categorical])
    gr = encoder.categories_
    list_of_columns = np.concatenate(gr).ravel().tolist()
    tn = list_of_columns
    X = pd.DataFrame(df[numerical])
    X2 = pd.DataFrame.sparse.from_spmatrix(encoder.transform(df[categorical]))
    X2 = X2.set_axis(tn, axis=1)
    X = X.join(X2)
    y = df[target]
    return X, y