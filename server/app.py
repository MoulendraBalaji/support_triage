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

# Create the app
app = create_app(
    SupportTriageEnvironment,
    SupportTriageAction,
    SupportTriageObservation,
    env_name="support_triage",
    max_concurrent_envs=1,
)

@app.get("/")
async def root():
    return {"status": "running", "environment": "SupportTriageEnvironment"}

@app.get("/health")
async def health():
    return {"status": "ok"}

def main():
    """Entry point for direct execution."""
    host = "0.0.0.0"
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
