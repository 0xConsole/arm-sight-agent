"""
ArmSight API entry point for Vercel serverless functions.
Exposes the FastAPI app from app.main as the handler.
"""
from app.main import app

# Vercel expects a callable named `handler`
handler = app
