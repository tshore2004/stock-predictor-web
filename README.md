# Stock Predictor Web

[Live Demo](https://stock-predictor-web.vercel.app)

This is a personal project created to learn about APIs, machine learning, and full-stack web development. It combines a Next.js frontend with a FastAPI backend to predict stock prices using a neural network trained on historical data and technical indicators.

## Project Purpose

The main goal of this project is to explore how modern web APIs, machine learning, and deployment platforms work together. It is not intended for financial advice or production use, but as a playground for learning and experimentation.

## Tech Stack

- **Frontend:** Next.js (React, TypeScript, Tailwind CSS)
- **Backend:** FastAPI (Python)
- **Machine Learning:** TensorFlow, scikit-learn
- **Data Sources:** Yahoo Finance, Stooq, Polygon (optional)

## Features

- Enter a stock ticker and get a next-day closing price prediction
- View model validation error (MAE) for the latest trained model
- On-demand model training and prediction for any ticker
- Uses technical indicators (MA, RSI, MACD, Bollinger Bands) as features
- Pretrained models for fast cold starts

## How to Run Locally

### 1. Backend (API)

```bash
cd api
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

The API will be available at `http://localhost:8000`.

### 2. Frontend (Web)

```bash
cd web
npm install
npm run dev
```

The web app will be available at `http://localhost:3000`.

### 3. Environment Variables

- `POLYGON_API_KEY` (optional): Polygon.io API key for enhanced data fetching
- `NEXT_PUBLIC_API`: Base URL of the FastAPI backend (e.g. `http://localhost:8000`). The frontend falls back to `/api` if undefined.

## Machine Learning Model

- Sequence model (LSTM) trained on 60 days of technical indicators
- Features: Close, MA10, MA50, RSI, MACD, MACD Signal, Bollinger Bands
- Trains on-demand for any ticker, or uses a cached/pretrained model

## Disclaimer

This project is for educational purposes only. Predictions are not financial advice.
