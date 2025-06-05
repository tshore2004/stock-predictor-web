# Stock Predictor Web

This project includes a Next.js front-end and a FastAPI backend for stock price prediction. When deployed to Vercel, be sure to set the following environment variables:

- `POLYGON_API_KEY` – optional Polygon API key used for fetching market data.
- `NEXT_PUBLIC_API` – base URL of the FastAPI functions (e.g. `https://your-deployment.vercel.app/api`). The client falls back to `/api` when undefined.

Pretrained models are stored under `src/ml/models`. The API loads these models before training new ones, which keeps cold start times low on serverless deployments.
