import os
import json
import logging
import asyncio
from openai import AsyncOpenAI

from server.support_triage_environment import SupportTriageEnvironment
from models import SupportTriageAction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_baseline_on_task(task_id: str, client: AsyncOpenAI, env: SupportTriageEnvironment) -> float:
    logger.info(f"--- Starting Task: {task_id} ---")
    observation = env.reset(task_id=task_id)
    
    messages = [
        {"role": "system", "content": """You are an AI support triage agent.
Your goal is to resolve customer support tickets according to the task difficulty.
You have the following tools available through your action output:
- command: "read_email", argument: ""
- command: "lookup_customer", argument: "<email>"
- command: "search_kb", argument: "<keywords>"
- command: "reply", argument: "<response text>"
- command: "escalate", argument: "<reason>"

Respond with JSON format strictly matching:
{"command": "<command_name>", "argument": "<arg>"}"""}
    ]
    
    for step in range(1, 15):
        # Format observation
        obs_text = f"Result: {observation.last_command_result}\nResolved: {observation.is_resolved}\nReward: {observation.reward}"
        messages.append({"role": "user", "content": obs_text})
        
        logger.info(f"Step {step} Obs: {obs_text}")
        
        if observation.is_resolved or observation.done:
            logger.info(f"Task {task_id} completed with final reward: {observation.reward}")
            return observation.reward
            
        try:
            response = await client.chat.completions.create(
                model="gpt-4o",  # or gpt-4o-mini
                messages=messages,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            logger.info(f"Agent Action: {content}")
            action_data = json.loads(content)
            
            action = SupportTriageAction(
                command=action_data.get("command", ""),
                argument=action_data.get("argument", "")
            )
            
            messages.append({"role": "assistant", "content": content})
            observation = env.step(action)
            
        except Exception as e:
            logger.error(f"Error during agent step: {e}")
            break
            
    return observation.reward

async def main():
    if "OPENAI_API_KEY" not in os.environ:
        logger.warning("OPENAI_API_KEY not found in environment. Baseline might fail if missing.")
        
    client = AsyncOpenAI()
    env = SupportTriageEnvironment()
    
    tasks = ["easy", "medium", "hard"]
    scores = {}
    
    for t in tasks:
        score = await run_baseline_on_task(t, client, env)
        scores[t] = score
        
    logger.info("=============================")
    logger.info(f"BASELINE SCORES: {scores}")
    logger.info("=============================")

if __name__ == "__main__":
    asyncio.run(main())
