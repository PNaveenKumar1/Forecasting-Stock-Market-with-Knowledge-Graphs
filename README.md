
# Construction and Modeling of Temporally-Aware Heterogeneous Knowledge Graph for Forecasting Stock Market Trends

### Pallekonda Naveen Kumar

## Project Overview

This project addresses the challenge of forecasting stock market trends by constructing a **Temporally-Aware Heterogeneous Knowledge Graph (SmKG)**. SmKG captures inter-stock relationships, financial statements, analyst sentiments, corporate events, and macroeconomic factors from stock assets listed on NASDAQ and NSE markets. By leveraging time-aware embeddings, this project ranks stock assets based on their potential return on investment over various holding periods.

The core contributions include:
1. Construction of two knowledge graphs for NASDAQ and NSE markets.
2. Development of the **Time-Aware Heterogeneous Knowledge Graph Embedding model (TA-HKGE)** to rank stocks.
3. Empirical validation of the model showing improved performance in stock ranking tasks.

## Key Features

- **Knowledge Graph Construction**: Captures relationships between stocks, events, and financial indicators.
- **Time-Aware Embeddings**: Incorporates the temporal dynamics of stock movements.
- **Stock Ranking**: Based on metrics like Normalized Discounted Cumulative Gain (NDCG) and Investment Return Ratio (IRR).

## Datasets

The datasets used in this project include stock prices from:
- NASDAQ100, S&P500 (from Yahoo Finance)
- NIFTY500 (from NSE)

Additionally, the project incorporates data on:
- Financial ratios and statements (Macrotrends)
- Corporate events and news articles (Seeking Alpha, Money Control)


## Results

The model has shown an improvement of **0.0316 in NDCG** and **2.26% in accuracy** on stock ranking tasks when compared to baseline models. These results have been validated on real-world stock data from NASDAQ and NSE markets.

## Model Architecture

The model utilizes:
- **LSTM networks** for time series data
- **Heterogeneous graph attention networks** to modify embeddings based on stock relationships
- A final fully connected layer to compute stock ranking scores.

![Model Architecture](https://github.com/user-attachments/assets/dd15ff52-b65f-4c8c-ac73-66785c85c596)

