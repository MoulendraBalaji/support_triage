# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

"""
FastAPI application for the Support Triage Environment.

This module creates an HTTP server that exposes the SupportTriageEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from .support_triage_environment import SupportTriageEnvironment
except (ImportError, ValueError):
    from server.support_triage_environment import SupportTriageEnvironment

try:
    from ..models import SupportTriageAction, SupportTriageObservation
except (ImportError, ValueError):
    from models import SupportTriageAction, SupportTriageObservation


# Create the app with web interface and README integration
app = create_app(
    SupportTriageEnvironment,
    SupportTriageAction,
    SupportTriageObservation,
    env_name="support_triage",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


def main():
    """
    Entry point for direct execution.
    """
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the Support Triage Environment server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 7860)), help="Port number to listen on")
    
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


@app.get("/")
async def root():
    return {"status": "running", "environment": "SupportTriageEnvironment"}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    main()
