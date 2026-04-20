# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import uvicorn
from openenv.core.env_server.http_server import create_app

try:
    from .support_triage_environment import SupportTriageEnvironment
except (ImportError, ValueError):
    from server.support_triage_environment import SupportTriageEnvironment

try:
    from ..models import SupportTriageAction, SupportTriageObservation
except (ImportError, ValueError):
    from models import SupportTriageAction, SupportTriageObservation

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Create the app
app = create_app(
    SupportTriageEnvironment,
    SupportTriageAction,
    SupportTriageObservation,
    env_name="support_triage",
    max_concurrent_envs=1,
)

# Mount frontend files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

@app.get("/style.css")
async def style():
    return FileResponse("frontend/style.css")

@app.get("/script.js")
async def script():
    return FileResponse("frontend/script.js")

@app.get("/health")
async def health():
    return {"status": "ok"}

def main(host: str = "0.0.0.0", port: int = None):
    """Entry point for direct execution."""
    if port is None:
        port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
