import asyncio
import os
import json
import textwrap
from typing import List, Optional
from openai import AsyncOpenAI

# Importing our environment and models
try:
    from .client import SupportTriageEnv
    from .models import SupportTriageAction
except (ImportError, ValueError):
    from client import SupportTriageEnv
    from models import SupportTriageAction

# Configuration from Environment Variables
HF_TOKEN = os.getenv("HF_TOKEN")
# Note: Use router.huggingface.co as the base URL for the API. This returns 404 in browser but is required for code.
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.1-8B-Instruct"
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or "support_triage_env:latest"
# Default to your Space URL if no environment URL is provided
ENV_URL = os.getenv("ENV_URL") or os.getenv("SUPPORT_TRIAGE_URL") or "https://moulendrabalaji-2007-support-triage.hf.space"
TASK_NAME = os.getenv("SUPPORT_TRIAGE_TASK", "easy")
BENCHMARK = "support_triage"

# Parameters
MAX_STEPS = 10
TEMPERATURE = 0.0 # Stick to deterministic for triage
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.5 

SYSTEM_PROMPT = textwrap.dedent('''
    You are a Customer Support Triage AI. 
    Your goal is to process tickets accurately and efficiently.
    
    COMMANDS:
    - read_email (arg: ""): Call this first to see the ticket.
    - lookup_customer (arg: "<email_address>"): Call this with the email found in the ticket.
    - search_kb (arg: "<keywords>"): Call this with keywords related to the problem.
    - reply (arg: "<message>"): Send the final solution to the customer.
    - escalate (arg: "<reason>"): Send to engineer if it's an Enterprise outage.
    
    WORKFLOW:
    1. Call 'read_email' first.
    2. Then 'lookup_customer' with the email found.
    3. Then 'search_kb' with problem keywords.
    4. Finally 'reply' or 'escalate'.
    
    CRITICAL:
    - NEVER put the "Current Observation" or "Result of last command" in the 'argument' field.
    - Only use the arguments defined above.
    - ALWAYS respond with a JSON object:
    {"command": "command_name", "argument": "argument_text"}
''').strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Flatten action for cleaner logs
    action_flat = action.replace('\n', ' ').strip()
    print(
        f"[STEP] step={step} action={action_flat} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

async def get_model_action(client: AsyncOpenAI, observation: str) -> Optional[SupportTriageAction]:
    try:
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Current Observation: {observation}\nWhat is your next action?"},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"}
        )
        content = (completion.choices[0].message.content or "").strip()
        data = json.loads(content)
        return SupportTriageAction(
            command=data.get("command", ""),
            argument=data.get("argument", "")
        )
    except Exception as exc:
        print(f"[ERROR] get_model_action failed: {exc}", flush=True)
        return None

async def main() -> None:
    # Try to read token from token.txt or environment
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPEN_AI_API_KEY")
    
    if not api_key:
        token_file = os.path.join(os.path.dirname(__file__), "token.txt")
        if os.path.exists(token_file):
            with open(token_file, "r") as f:
                api_key = f.read().strip()
    
    if not api_key or api_key == "none":
        print("="*60)
        print("[CRITICAL] API Token Missing!")
        print("Either set the HF_TOKEN environment variable or put it in token.txt")
        print("  Windows: $env:HF_TOKEN='your_token'")
        print("  Linux/Mac: export HF_TOKEN='your_token'")
        print("="*60)
        return
        
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=api_key)

    # We will run all 3 tasks (easy, medium, hard) to satisfy the "at least 3 tasks" requirement
    tasks_to_run = ["easy", "medium", "hard"]
    
    for task_id in tasks_to_run:
        rewards = []
        steps_taken = 0
        score = 0.0
        success = False

        log_start(task_id, BENCHMARK, MODEL_NAME)

        try:
            # Initialize Environment
            if os.getenv("LOCAL_IMAGE_NAME"):
                env = await SupportTriageEnv.from_docker_image(LOCAL_IMAGE_NAME)
            elif ENV_URL:
                env = SupportTriageEnv(base_url=ENV_URL)
            else:
                env = SupportTriageEnv(base_url=f"http://localhost:{os.getenv('PORT', '7860')}")

            # reset with specific task
            result = await env.reset(options={"task_id": task_id})
            obs = result.observation
            
            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                obs_context = f"Result of last command: {obs.last_command_result} | Resolved: {obs.is_resolved} | Task: {obs.task_difficulty}"
                action = await get_model_action(client, obs_context)
                if not action:
                    break
                
                result = await env.step(action)
                obs = result.observation
                reward = result.reward or 0.0
                
                rewards.append(reward)
                steps_taken = step
                log_step(step=step, action=action.command, reward=reward, done=result.done, error=None)

                if result.done:
                    break

            total_reward = sum(rewards)
            # Map reward to (0, 1) range
            # Easy/Medium/Hard have different max rewards. 
            # We use a formula that ensures result is strictly in (0, 1)
            raw_score = max(0.0, min(1.0, (total_reward + 1.0) / 3.0))
            score = 0.05 + (raw_score * 0.9) # Range [0.05, 0.95]
            
            success = obs.is_resolved and score >= 0.4

        except Exception as e:
            print(f"[ERROR] Task {task_id} failed: {e}", flush=True)
            score = 0.01 # Minimum non-zero score for failure
        finally:
            log_end(success, steps_taken, score, rewards)
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
