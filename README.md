# Forecast de Vendas - Janeiro 2023 (Versão V04)

Este projeto realiza previsão de vendas semanais por PDV e SKU para as semanas 1 a 5 de janeiro de 2023, utilizando LightGBM e features temporais/lags.

## Estrutura do projeto

- `forecast_v04.py` - Script principal de forecast.
- `transactions.parquet` - Dados de transações.
- `products.parquet` - Dados de produtos.
- `stores.parquet` - Dados de PDVs.
- `forecast_jan2023_V04.csv` - Resultado da previsão.

## Requisitos

- Python 3.9+
- Bibliotecas Python:
  ```bash
  pip install pandas numpy scikit-learn lightgbm
