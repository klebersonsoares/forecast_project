# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 13:13:28 2025

@author: kleberson.soares
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import early_stopping, log_evaluation

# ===============================
# 1️⃣ Carregar dados
# ===============================
df_trans = pd.read_parquet('transactions.parquet')
df_products = pd.read_parquet('products.parquet')
df_pdv = pd.read_parquet('stores.parquet')

# ===============================
# 2️⃣ Preparar datas e agregação semanal
# ===============================
df_trans['transaction_date'] = pd.to_datetime(df_trans['transaction_date'])
df_trans['weekofyear'] = df_trans['transaction_date'].dt.isocalendar().week.astype(int)
df_trans['year'] = df_trans['transaction_date'].dt.year
df_trans['month'] = df_trans['transaction_date'].dt.month
df_trans['dayofweek'] = df_trans['transaction_date'].dt.dayofweek

# Agregar quantidade total por PDV × SKU × semana
df_agg = df_trans.groupby(
    ['internal_store_id','internal_product_id','year','weekofyear','month','dayofweek'],
    as_index=False
)['quantity'].sum()

# ===============================
# 3️⃣ Merge com produtos e PDVs
# ===============================
df_agg = df_agg.merge(df_products[['produto']], left_on='internal_product_id', right_on='produto', how='left')
df_agg = df_agg.merge(df_pdv[['pdv']], left_on='internal_store_id', right_on='pdv', how='left')

# ===============================
# 4️⃣ Features históricas por SKU
# ===============================
sku_stats = df_agg.groupby('internal_product_id')['quantity'].agg(['mean','std']).reset_index()
sku_stats.rename(columns={'mean':'sku_mean','std':'sku_std'}, inplace=True)
df_agg = df_agg.merge(sku_stats, on='internal_product_id', how='left')

# ===============================
# 5️⃣ Criar lags e médias móveis
# ===============================
df_agg = df_agg.sort_values(['internal_product_id','internal_store_id','year','weekofyear'])

# Lags
for lag in [1,2,3,4,12]:
    df_agg[f'lag_{lag}'] = df_agg.groupby(['internal_product_id','internal_store_id'])['quantity'].shift(lag)

# Médias móveis
for window in [3,4,12]:
    df_agg[f'rolling_mean_{window}'] = (
        df_agg.groupby(['internal_product_id','internal_store_id'])['quantity']
        .shift(1).rolling(window).mean()
    )

# ===============================
# 6️⃣ Encoding simples e sazonalidade cíclica
# ===============================
df_agg['sku_freq'] = df_agg['internal_product_id'].map(df_agg['internal_product_id'].value_counts())
df_agg['pdv_freq'] = df_agg['internal_store_id'].map(df_agg['internal_store_id'].value_counts())

# Sazonalidade cíclica da semana
df_agg['week_sin'] = np.sin(2*np.pi*df_agg['weekofyear']/52)
df_agg['week_cos'] = np.cos(2*np.pi*df_agg['weekofyear']/52)

# ===============================
# 7️⃣ Remover linhas iniciais com NaN (lags)
# ===============================
df_agg = df_agg.dropna().reset_index(drop=True)

# ===============================
# 8️⃣ Preparar features e target
# ===============================
features = [
    'weekofyear','month','dayofweek','sku_mean','sku_std',
    'lag_1','lag_2','lag_3','lag_4','lag_12',
    'rolling_mean_3','rolling_mean_4','rolling_mean_12',
    'sku_freq','pdv_freq','week_sin','week_cos'
]
y = df_agg['quantity']
X = df_agg[features]

# ===============================
# 9️⃣ Validação temporal com TimeSeriesSplit
# ===============================
ts_split = TimeSeriesSplit(n_splits=3)

def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)

wmape_scores = []
for fold, (train_idx, val_idx) in enumerate(ts_split.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        feature_fraction=0.8,
        subsample=0.8,
        random_state=42
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='l1',
        callbacks=[early_stopping(stopping_rounds=100), log_evaluation(0)]
    )

    y_pred = model.predict(X_val)
    score = wmape(y_val, y_pred)
    wmape_scores.append(score)
    print(f"Fold {fold+1} - WMAPE: {score:.4f}")

print("Média WMAPE cross-val:", np.mean(wmape_scores))

# ===============================
# 🔟 Refit do modelo em todo dataset
# ===============================
model.fit(X, y)
df_agg['quantidade'] = model.predict(X)

# ===============================
# 11️⃣ Preparar CSV para submissão
# ===============================
df_forecast = df_agg[['year','weekofyear','internal_store_id','internal_product_id','quantidade']].copy()
df_forecast.rename(columns={
    'weekofyear': 'semana',
    'internal_store_id': 'pdv',
    'internal_product_id': 'produto'
}, inplace=True)

# Filtrar semanas 1–5 de janeiro/2023
df_forecast = df_forecast[df_forecast['semana'].isin([1,2,3,4,5])]

# Arredondar quantidade para inteiro
df_forecast['quantidade'] = df_forecast['quantidade'].round().astype(int)

# Salvar CSV com separador ; e encoding UTF-8
df_forecast.to_csv('forecast_jan2023_V04.csv', sep=';', index=False, encoding='utf-8')
print("Arquivo forecast_jan2023_V04.csv criado com sucesso!")
