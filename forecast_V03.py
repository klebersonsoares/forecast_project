# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 10:46:24 2025

@author: kleberson.soares
"""

import pandas as pd
import numpy as np
import lightgbm as lgb

# 1️⃣ Carregar dados
df_trans = pd.read_parquet('transactions.parquet')
df_products = pd.read_parquet('products.parquet')
df_pdv = pd.read_parquet('stores.parquet')

# 2️⃣ Datas e agregação semanal
df_trans['transaction_date'] = pd.to_datetime(df_trans['transaction_date'])
df_trans['week'] = df_trans['transaction_date'].dt.isocalendar().week
df_trans['month'] = df_trans['transaction_date'].dt.month
df_trans['dayofweek'] = df_trans['transaction_date'].dt.dayofweek

# Agregar: quantidade total por PDV × SKU × semana
df_agg = df_trans.groupby(['internal_store_id','internal_product_id','week','month','dayofweek'], as_index=False)['quantity'].sum()

# 3️⃣ Merge com produtos e PDVs
df_agg = df_agg.merge(df_products[['produto']], left_on='internal_product_id', right_on='produto', how='left')
df_agg = df_agg.merge(df_pdv[['pdv']], left_on='internal_store_id', right_on='pdv', how='left')

# 4️⃣ Features históricas por SKU (estatísticas globais)
sku_stats = df_agg.groupby('internal_product_id')['quantity'].agg(['mean','std']).reset_index()
sku_stats.rename(columns={'mean':'sku_mean','std':'sku_std'}, inplace=True)
df_agg = df_agg.merge(sku_stats, on='internal_product_id', how='left')

# 5️⃣ Criar lags e médias móveis (histórico recente por SKU e PDV)
df_agg = df_agg.sort_values(['internal_product_id','internal_store_id','week'])
for lag in [1, 2, 3]:
    df_agg[f'lag_{lag}'] = df_agg.groupby(['internal_product_id','internal_store_id'])['quantity'].shift(lag)

df_agg['rolling_mean_3'] = (
    df_agg.groupby(['internal_product_id','internal_store_id'])['quantity']
    .shift(1).rolling(3).mean()
)

# 6️⃣ Encoding simples para produto e loja
df_agg['sku_freq'] = df_agg['internal_product_id'].map(df_agg['internal_product_id'].value_counts())
df_agg['pdv_freq'] = df_agg['internal_store_id'].map(df_agg['internal_store_id'].value_counts())

# 7️⃣ Remover linhas iniciais com NaN (por causa dos lags)
df_agg = df_agg.dropna().reset_index(drop=True)

# 8️⃣ Preparar treino e validação temporal
features = [
    'week','month','dayofweek','sku_mean','sku_std',
    'lag_1','lag_2','lag_3','rolling_mean_3',
    'sku_freq','pdv_freq'
]
y = df_agg['quantity']

max_week = df_agg['week'].max()
train_weeks = df_agg['week'] < max_week - 2   # treino até penúltimas semanas
val_weeks   = df_agg['week'] >= max_week - 2  # valida últimas semanas

X_train, y_train = df_agg[train_weeks][features], y[train_weeks]
X_val, y_val     = df_agg[val_weeks][features], y[val_weeks]

# 9️⃣ Treinar modelo LightGBM
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31
)
model.fit(X_train, y_train)

# 🔟 Avaliar com WMAPE
def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)

y_pred = model.predict(X_val)
print("WMAPE validação:", wmape(y_val, y_pred))

# 11️⃣ Previsão final em todo dataset
df_agg['quantidade'] = model.predict(df_agg[features])

# 12️⃣ Ajustar nomes do CSV
df_forecast = df_agg[['week','internal_store_id','internal_product_id','quantidade']].copy()
df_forecast.rename(columns={
    'week': 'semana',
    'internal_store_id': 'pdv',
    'internal_product_id': 'produto'
}, inplace=True)

# Filtrar semanas 1–5 (janeiro)
df_forecast = df_forecast[df_forecast['semana'].isin([1,2,3,4,5])]

# 13️⃣ Salvar CSV
df_forecast.to_csv('forecast_jan2023_V03.csv', index=False)
print("Arquivo forecast_jan2023_V03.csv criado com sucesso!")
