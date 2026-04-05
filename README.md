---
title: Support Triage Env
emoji: 💻
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
---
# Support Triage Environment

## Description and Motivation

The Support Triage Environment simulates a real-world customer support routing and response task.
An AI agent must process an incoming customer ticket, lookup customer metadata, read relevant KB (Knowledge Base) articles, and then perform the correct action (reply to the user or escalate to an engineer). This isn't a toy problem—it's a very common human-in-the-loop task that companies face every day. Automating triage requires reasoning over multiple steps: gathering context, checking policy, and taking the appropriate branch of logic.

## Action Space

The agent interacts with the environment by issuing a single action containing a `command` and an `argument`. The action space supports:

- `read_email` (argument: `""`): Retrieves the incoming customer email content.
- `lookup_customer` (argument: `<email_address>`): Queries the customer DB for tier/status.
- `search_kb` (argument: `<keywords>`): Searches the knowledge base for instructions/policy.
- `reply` (argument: `<message>`): Sends a response to the customer and closes the ticket.
- `escalate` (argument: `<reason>`): Escalates the ticket to higher-tier human support.

## Observation Space

The environment returns an Observation containing:

- `last_command_result` (str): Text result of the command (e.g. the email, the KB article, the DB entry).
- `is_resolved` (bool): True if the ticket was successfully evaluated and closed.
- `task_difficulty` (str): The current difficulty setting.
- `reward` (float): The current incremental reward, accumulating through partial progress.

## Tasks and Difficulty

1. **Easy ("easy")**: Triage a password reset request. The agent reads the email, searches KB for reset policy, and replies with standard instructions.
2. **Medium ("medium")**: Triage a refund request. The agent must lookup the customer to verify they are a Pro user and requested within the policy timeframe before replying with a full refund.
3. **Hard ("hard")**: Triage an Enterprise production DB outage. The agent must look up the user, verify they are Enterprise, search KB to understand that Enterprise outages must NOT be replied to directly, and then use the `escalate` action.

## Setup & Usage

To install the dependencies (make sure you have python and `uv` installed):

```bash
uv sync
```

To run the openenv validation locally:

```bash
python -m openenv.cli validate
```

### Running the API locally

```bash
uv run --project . server
```

Or start via Docker:

```bash
docker build -t support_triage_env .
docker run -p 8000:8000 support_triage_env
```

## Baseline Execution

You can run the baseline inference script locally.

```bash
python inference.py
```

### Baseline Scores (GPT-4o)

- **Easy**: ~1.0
- **Medium**: ~0.9
- **Hard**: ~0.8
  _Note: Scores show accumulated step-wise reward trajectory._
