from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class SupportTriageAction(Action):
    """Action for the Support Triage environment."""
    
    command: str = Field(
        ..., 
        description="Command to execute: 'read_email', 'search_kb', 'lookup_customer', 'reply', 'escalate'."
    )
    argument: str = Field(
        default="", 
        description="Argument for the command. For search_kb: query string. For lookup_customer: email address. For reply: the response message. For escalate: reason."
    )


class SupportTriageObservation(Observation):
    """Observation from the Support Triage environment."""
    
    last_command_result: str = Field(default="", description="Result of the last executed command.")
    is_resolved: bool = Field(default=False, description="Whether the ticket has been resolved.")
    task_difficulty: str = Field(default="easy", description="Difficulty level of the current task.")
