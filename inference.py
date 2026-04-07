import asyncio
import os
import json
import textwrap
from typing import List, Optional
from openai import OpenAI

# Importing our environment and models
from client import SupportTriageEnv
from models import SupportTriageAction

# Configuration from Environment Variables
HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://api-inference.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Meta-Llama-3.1-8B-Instruct"
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or "support_triage_env:latest"
TASK_NAME = os.getenv("SUPPORT_TRIAGE_TASK", "easy")
BENCHMARK = "support_triage"

# Parameters
MAX_STEPS = 10
TEMPERATURE = 0.0 # Stick to deterministic for triage
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.5 

# For Support Triage, max reward is approx 1.2 across all steps
MAX_TOTAL_REWARD = 1.0 

SYSTEM_PROMPT = textwrap.dedent("""
    You are a Customer Support Triage AI. 
    Your goal is to process tickets accurately.
    You have these commands:
    - read_email (arg: ""): Initial reading of the ticket.
    - lookup_customer (arg: "<email>"): Find customer tier/details.
    - search_kb (arg: "<query>"): Find policy articles.
    - reply (arg: "<msg>"): Reply to customer (ends task).
    - escalate (arg: "<reason>"): Send to engineer (ends task).
    
    CRITICAL: ALWAYS respond with a JSON object:
    {"command": "command_name", "argument": "argument_text"}
""").strip()

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

def get_model_action(client: OpenAI, observation: str) -> Optional[SupportTriageAction]:
    try:
        completion = client.chat.completions.create(
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
        # Fallback or log error
        return None

async def main() -> None:
    # Use HF_TOKEN as API key if provided
    api_key = HF_TOKEN or os.getenv("OPEN_AI_API_KEY") or "none"
    client = OpenAI(base_url=API_BASE_URL, api_key=api_key)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    error_msg = None

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Initialize Environment
        # We prioritize LOCAL_IMAGE_NAME if set, else assume a local server for testing
        if os.getenv("LOCAL_IMAGE_NAME"):
            env = await SupportTriageEnv.from_docker_image(LOCAL_IMAGE_NAME)
        else:
            # Fallback to local server running on fixed port (defaulting to our fixed 7860)
            env = SupportTriageEnv(base_url=f"http://localhost:{os.getenv('PORT', '7860')}")

        # reset with specific task
        result = await env.reset(options={"task_id": TASK_NAME})
        obs = result.observation
        
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # Format observation for the model
            obs_context = f"Result of last command: {obs.last_command_result} | Resolved: {obs.is_resolved} | Task: {obs.task_difficulty}"
            
            action = get_model_action(client, obs_context)
            if not action:
                error_msg = "Model failed to generate valid JSON action"
                break
            
            action_str = f"{action.command}('{action.argument}')"

            result = await env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps_taken = step
            
            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            if done:
                break

        total_reward = sum(rewards)
        score = total_reward / MAX_TOTAL_REWARD
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] {error_msg}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals() and env:
            try:
                await env.close()
            except:
                pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
