# Stock Market Prediction & Sentimental Analysis

\<div align="center"\>

[](https://opensource.org/licenses/MIT)
[](https://www.python.org/downloads/)

\</div\>

A comprehensive project to predict the Dow Jones Industrial Average (DJIA) stock prices by leveraging financial news sentiment with deep learning and ensemble models. This repository contains the code and detailed methodology for integrating natural language processing (NLP) with traditional time-series analysis.

## Table of Contents

  - [About The Project](https://www.google.com/search?q=%23about-the-project)
  - [Key Features](https://www.google.com/search?q=%23key-features)
  - [Methodology](https://www.google.com/search?q=%23methodology)
  - [Tech Stack](https://www.google.com/search?q=%23tech-stack)
  - [Getting Started](https://www.google.com/search?q=%23getting-started)
      - [Prerequisites](https://www.google.com/search?q=%23prerequisites)
      - [Installation](https://www.google.com/search?q=%23installation)
  - [Results](https://www.google.com/search?q=%23results)
  - [Future Work](https://www.google.com/search?q=%23future-work)
  - [Authors](https://www.google.com/search?q=%23authors)

## About The Project

Traditional stock market prediction models often rely solely on historical price data, which may not capture the full spectrum of market dynamics. This project addresses that limitation by incorporating market sentiment, a critical driver of investor behavior.

By analyzing the sentiment of daily financial news headlines, we aim to quantify market psychology and integrate it as a powerful feature into our predictive models. The core hypothesis is that combining sentiment analysis with technical indicators will lead to a more robust and accurate prediction of stock price movements.

This project was developed based on the work presented in the "Stock Market Prediction & Sentimental Analysis" report.

## Key Features

  * **Sentiment Analysis**: Utilizes **FinBERT**, a state-of-the-art language model pre-trained on financial text, to accurately score news sentiment.
  * **Extensive Feature Engineering**: Constructs a rich feature set including:
      * Technical indicators (SMA, EMA, MACD, RSI).
      * Sentiment-derived features (e.g., daily average sentiment, count of positive/negative articles).
      * Lag features from historical price data.
  * **Multi-Model Evaluation**: Trains, evaluates, and compares a diverse range of models, from traditional machine learning to deep learning, including XGBoost, LightGBM, Prophet, LSTM, and GRU.
  * **Advanced Ensemble Modeling**: Implements a **Stacking Ensemble Regressor** to combine the predictive power of the best-performing base models, further enhancing accuracy.

## Methodology

The project follows a structured, multi-stage workflow:

1.  **Data Sourcing and Integration**: Historical DJIA price data and daily news headlines (from sources including Reddit) are collected, cleaned, and merged into a unified dataset.
2.  **Sentiment Scoring**: Each news headline is processed by the FinBERT model to generate sentiment scores (positive, negative, neutral). These scores are then aggregated per day.
3.  **Feature Construction**: The integrated dataset is enriched with the engineered features described above.
4.  **Model Training and Evaluation**: The dataset is split into training and testing sets. Each model is trained on the historical data and evaluated based on its ability to predict future closing prices using the **Mean Squared Error (MSE)** metric.
5.  **Ensemble Optimization**: The final Stacking model is built using the most effective base learners to produce the final prediction.

## Tech Stack

This project is built with Python and leverages the following powerful libraries and frameworks:

| Technology                               | Description                                         |
| ---------------------------------------- | --------------------------------------------------- |
| **Python** | Core programming language                           |
| **Pandas & NumPy** | Data manipulation and numerical analysis            |
| **PyTorch** | Deep learning framework for LSTM/GRU models         |
| **Hugging Face Transformers** | For implementing the FinBERT sentiment model        |
| **Scikit-learn** | For machine learning utilities and ensemble methods |
| **XGBoost & LightGBM** | High-performance gradient boosting models           |
| **Prophet** | Time-series forecasting library by Facebook         |
| **Matplotlib & Seaborn** | Data visualization and plotting                     |

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

Make sure you have Python 3.8+ and pip installed on your system.

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/NguyenAnhDucDev/Stock-Market-Prediction-Sentimental-Analysis.git
    cd Stock-Market-Prediction-Sentimental-Analysis
    ```
2.  **Install dependencies:**
    The required libraries are listed in the Jupyter Notebook. It is recommended to create a `requirements.txt` file and install from it.
    ```sh
    pip install pandas numpy torch transformers scikit-learn xgboost lightgbm prophet matplotlib seaborn jupyter
    ```
3.  **Run the Jupyter Notebook:**
    Launch the notebook to explore the entire workflow.
    ```sh
    jupyter notebook BA_code.ipynb
    ```

## Results

The evaluation results showed that models incorporating sentiment features generally outperformed those based on price data alone. The **Stacking Ensemble model achieved the lowest Mean Squared Error (MSE)**, confirming its superior predictive accuracy compared to any single model.
