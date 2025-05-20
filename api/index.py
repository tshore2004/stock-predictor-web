"""
Vercel Serverless entry-point.
Every route defined in src.api.main ( /predict, /metrics, … )
will be reachable under /api/* in production.
"""
from src.api.main import app as handler  # fastapi.FastAPI instance