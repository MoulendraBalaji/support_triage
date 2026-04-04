import os
import json
import logging
import asyncio
from openai import AsyncOpenAI

from server.support_triage_environment import SupportTriageEnvironment
from models import SupportTriageAction

# Mute standard logging as strictly stdout format using START/STEP/END is required
logging.getLogger().setLevel(logging.ERROR) 

async def run_baseline_on_task(task_id: str, client: AsyncOpenAI, env: SupportTriageEnvironment, model_name: str) -> float:
    print(f"[START] Task run for {task_id}")
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
        obs_text = f"Result: {observation.last_command_result}\nResolved: {observation.is_resolved}\nReward: {observation.reward}"
        messages.append({"role": "user", "content": obs_text})
        
        # Flattening newlines for cleaner STEP logs
        obs_flat = obs_text.replace('\n', ' | ')
        print(f"[STEP] {step} Observation: {obs_flat}")
        
        if observation.is_resolved or observation.done:
            print(f"[END] Task {task_id} completed. Final Reward: {observation.reward}")
            return observation.reward
            
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            
            # Print agent action carefully structured
            print(f"[STEP] {step} Agent Action: {content.replace(chr(10), '')}")
            action_data = json.loads(content)
            
            action = SupportTriageAction(
                command=action_data.get("command", ""),
                argument=action_data.get("argument", "")
            )
            
            messages.append({"role": "assistant", "content": content})
            observation = env.step(action)
            
        except Exception as e:
            print(f"[END] Task {task_id} Failed due to exception: {e}")
            break
            
    return observation.reward

async def main():
    hf_token = os.environ.get("HF_TOKEN")
    api_base_url = os.environ.get("API_BASE_URL", "http://0.0.0.0:8000")
    model_name = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")

    # The prompt explicitly requires strictly following these env variables
    if not hf_token:
        print("WARNING: HF_TOKEN not found in environment. OpenAI client connection might fail.")
    if not api_base_url:
        print("WARNING: API_BASE_URL not found in environment. OpenAI client will default to standard endpoint if not present.")

    # Initialize OpenAI client conforming strictly to the variables
    client_kwargs = {}
    if hf_token:
        client_kwargs['api_key'] = hf_token
    else:
        # Fallback for standard key if HF_TOKEN isn't set
        client_kwargs['api_key'] = os.environ.get("OPENAI_API_KEY", "dummy_key")
        
    if api_base_url:
        client_kwargs['base_url'] = api_base_url
        
    client = AsyncOpenAI(**client_kwargs)
    
    env = SupportTriageEnvironment()
    
    tasks = ["easy", "medium", "hard"]
    scores = {}
    
    for t in tasks:
        score = await run_baseline_on_task(t, client, env, model_name)
        scores[t] = score
        
    print(f"[END] Evaluation complete. Scores: {scores}")

if __name__ == "__main__":
    asyncio.run(main())
