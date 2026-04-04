from uuid import uuid4
import re
from typing import Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SupportTriageAction, SupportTriageObservation
except ImportError:
    from models import SupportTriageAction, SupportTriageObservation

TASKS = {
    "easy": {
        "email_content": "Hi, I forgot my password and am locked out. How do I reset it? - Alice (alice@example.com)",
        "customer": {"email": "alice@example.com", "tier": "free", "status": "active"},
        "expected_kb_query": ["password", "reset"],
        "max_steps": 5,
    },
    "medium": {
        "email_content": "User bob@test.com purchased Pro yesterday but wants a full refund due to a feature missing.",
        "customer": {"email": "bob@test.com", "tier": "pro", "purchase_days_ago": 1},
        "expected_kb_query": ["refund", "policy", "pro"],
        "max_steps": 7,
    },
    "hard": {
        "email_content": "URGENT: My production database went down! We are losing money! - Charlie (charlie@enterprise.com)",
        "customer": {"email": "charlie@enterprise.com", "tier": "enterprise", "status": "critical"},
        "expected_kb_query": ["down", "enterprise", "outage", "escalate"],
        "max_steps": 8,
    }
}

KB_ARTICLES = {
    "password_reset": "To reset your password, visit the /reset-password link and enter your email.",
    "refund_policy": "Pro and Enterprise users are eligible for a full refund if requested within 14 days of purchase. Free users do not get refunds.",
    "enterprise_support": "For any Enterprise user reporting a production outage (system down), DO NOT reply directly. Escalate the ticket immediately with high priority.",
}

class SupportTriageEnvironment(Environment):
    """
    Customer Support Triage Environment.
    The agent acts as a customer support router/responder.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.current_task_id = "easy"
        self.task_data = {}
        self.has_looked_up = False
        self.has_searched_kb = False

    def reset(self, task_id: str = "easy") -> SupportTriageObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        if task_id not in TASKS:
             task_id = "easy"
             
        self.current_task_id = task_id
        self.task_data = TASKS[task_id]
        
        self.has_looked_up = False
        self.has_searched_kb = False

        return SupportTriageObservation(
            last_command_result="Environment reset successful. Use 'read_email' to start processing the ticket.",
            is_resolved=False,
            task_difficulty=task_id,
            done=False,
            reward=0.0,
        )

    def step(self, action: SupportTriageAction) -> SupportTriageObservation:
        self._state.step_count += 1
        
        cmd = action.command.lower()
        arg = action.argument.lower()
        
        result = ""
        reward = 0.0
        done = False
        
        if cmd == "read_email":
            result = f"Email Subject/Body: {self.task_data['email_content']}"
            
        elif cmd == "lookup_customer":
            email = arg.strip()
            if email == self.task_data["customer"]["email"]:
                result = f"Customer DB Record: {self.task_data['customer']}"
                if not self.has_looked_up:
                    reward += 0.2
                    self.has_looked_up = True
            else:
                result = f"Customer {email} not found in DB."
                reward -= 0.1
                
        elif cmd == "search_kb":
            query_words = arg.split()
            matched = False
            for word in query_words:
                if word in ["password", "reset"]:
                    result += "Found KB Article (Password): " + KB_ARTICLES["password_reset"] + "\n"
                    matched = True
                elif word in ["refund", "policy", "pro"]:
                    result += "Found KB Article (Refunds): " + KB_ARTICLES["refund_policy"] + "\n"
                    matched = True
                elif word in ["down", "enterprise", "outage", "escalate"]:
                    result += "Found KB Article (Enterprise Outage): " + KB_ARTICLES["enterprise_support"] + "\n"
                    matched = True
                    
            if not matched:
                result = "No KB articles found matching your query."
            else:
                if not self.has_searched_kb:
                    # Check if query matches expected queries for current task
                    expected = self.task_data["expected_kb_query"]
                    if any(w in expected for w in query_words):
                        reward += 0.2
                        self.has_searched_kb = True
                        
        elif cmd == "reply":
            # Grader logic for replying
            if self.current_task_id == "easy":
                if "reset" in arg or "link" in arg:
                    reward += 0.8
                    result = "Successfully replied and resolved the ticket."
                else:
                    reward -= 0.5
                    result = "Reply sent but did not address the customer's need (missing reset instructions)."
            elif self.current_task_id == "medium":
                if "full refund" in arg or "processed" in arg:
                    # if they didn't lookup, maybe they guessed? We penalize slightly if premature
                    if self.has_looked_up and self.has_searched_kb:
                        reward += 0.5
                        result = "Successfully processed the full refund as per policy."
                    else:
                        reward -= 0.2
                        result = "You replied without verifying customer info or policy!"
                else:
                    reward -= 0.5
                    result = "Incorrect reply. Agent failed to provide full refund per policy."
            elif self.current_task_id == "hard":
                # For hard task, agent shouldn't reply directly
                reward -= 1.0
                result = "FATAL MISTAKE: You replied directly to an Enterprise outage without escalating!"
            
            done = True
            
        elif cmd == "escalate":
            if self.current_task_id == "hard":
                if self.has_looked_up and self.has_searched_kb:
                    reward += 0.6
                    result = "Successfully escalated the critical Enterprise incident."
                else:
                    reward += 0.2
                    result = "Escalated, but without proper verification. Next time look up and search KB first."
            else:
                reward -= 0.8
                result = "Incorrect action. You escalated a low-priority ticket that you could have resolved."
                
            done = True
            
        else:
            result = f"Unknown command: {cmd}. Valid commands are read_email, lookup_customer, search_kb, reply, escalate."
            reward -= 0.1

        # Check max steps
        if self._state.step_count >= self.task_data["max_steps"] and not done:
            done = True
            result = "Maximum steps reached without resolution."
            reward -= 0.5

        return SupportTriageObservation(
            last_command_result=result,
            is_resolved=done,
            task_difficulty=self.current_task_id,
            done=done,
            reward=reward,
            metadata={"step": self._state.step_count}
        )

    @property
    def state(self) -> State:
        return self._state
