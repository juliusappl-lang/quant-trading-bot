from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import signals, watchlist, train

app = FastAPI(title="Signal Engine API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(signals.router, prefix="/api")
app.include_router(watchlist.router, prefix="/api")
app.include_router(train.router, prefix="/api")
