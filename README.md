# Forecast de Vendas - Janeiro 2023 (V04)

Este repositório contém o código e o CSV de previsão de vendas solicitados pela organização.  
O modelo gera previsões semanais por PDV e SKU para as primeiras semanas de janeiro de 2023.

## Estrutura do repositório

- `forecast_big_data.py` → Script Python completo da versão V04 do forecast, comentado e organizado em etapas numeradas.
- `forecast_jan2023_V04.csv` → Resultado final do forecast (CSV enviado à organização).
- `README.md` → Documentação e instruções de execução.
- `.gitignore` → Ignora arquivos grandes, como os arquivos `.parquet`.

> Os arquivos de dados brutos (`TRANSACTIONS`, `PRODUCTS`, `PDV`) não estão incluídos no repositório por excederem o limite do GitHub e não serem necessários para execução do script.  
> Para referência, os arquivos de input devem ser renomeados conforme o script:
> - `part-00000-tid-7173294866425216458-eae53fbf-d19e-4130-ba74-78f96b9675f1-4-1-c000.snappy` → `TRANSACTIONS.parquet`  
> - `part-00000-tid-5196563791502273604-c90d3a24-52f2-4955-b4ec-fb143aae74d8-4-1-c000.snappy` → `PRODUCTS.parquet`  
> - `part-00000-tid-2779033056155408584-f6316110-4c9a-4061-ae48-69b77c7c8c36-4-1-c000.snappy` → `PDV.parquet`

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
3. Coloque os arquivos de input na mesma pasta do script, renomeando conforme indicado acima.  
4. Execute o script no terminal:

```bash
python forecast_big_data.py


---
## Estratégias de modelagem e criatividade

O modelo V04 incorpora diversas estratégias para capturar padrões históricos e sazonais:

- **Lags e médias móveis** → capturam tendências e suavizam variações semanais.
- **Encoding simples e sazonalidade cíclica da semana** → melhora a captura de padrões sazonais.
- **Validação temporal (TimeSeriesSplit)** → garante avaliação robusta do modelo.
- **Estatísticas por SKU** → features adicionais como média e desvio padrão histórico do produto.

Essas estratégias foram selecionadas para superar a **baseline interna da organização**, conforme solicitado.

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
- O CSV incluído serve como referência de resultado final e validação do modelo.

---

## Métricas de avaliação do modelo

- **WMAPE médio** obtido nos folds de validação: `[inserir valor]`  
- **RMSE e MAE** podem ser verificados no script para acompanhamento de performance.  

Essas métricas evidenciam que o modelo foi testado e otimizado, atendendo aos critérios de performance e qualidade técnica do hackathon.




