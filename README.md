# Forecast de Vendas - Janeiro 2023 (V04)

Este repositório contém o código e o CSV de previsão de vendas solicitados pela organização.  
O modelo gera previsões semanais por PDV e SKU para as primeiras semanas de janeiro de 2023.

## Estrutura do repositório

- `forecast_big_data.py` → Script Python completo da versão V04 do forecast.
- `forecast_jan2023_V04.csv` → Resultado final do forecast (CSV enviado à organização).
- `README.md` → Documentação e instruções de execução.
- `.gitignore` → Ignora arquivos grandes, como os arquivos `.parquet`.

> Os arquivos de dados brutos (`transactions.parquet`, `products.parquet`, `stores.parquet`) não estão incluídos no repositório por excederem o limite do GitHub e não serem necessários para execução do script.

---
## Requisitos e dependências

- **Python 3.9+**
- Bibliotecas Python necessárias:

```bash
pip install pandas numpy scikit-learn lightgbm

---

## Como executar

1. Certifique-se de ter o Python 3 instalado.  
2. Instale as dependências (veja acima).  
3. Execute o script no terminal:

```bash
python forecast_big_data.py

---
## Formato do CSV de saída

| semana | pdv  | produto | quantidade |
|--------|------|---------|------------|
| 1      | 101  | 1001    | 5          |
| 2      | 101  | 1001    | 4          |
| …      | …    | …       | …          |

- `semana` → número da semana do ano  
- `pdv` → identificador do ponto de venda  
- `produto` → identificador do produto  
- `quantidade` → previsão de vendas (inteiro)

---

## Observações

- O repositório está preparado para ser público e contém apenas o necessário para reproduzir o forecast.  
- Para reproduzir com os dados brutos, será necessário solicitar os arquivos `.parquet` à organização ou gerar os dados correspondentes.


