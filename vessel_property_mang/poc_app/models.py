from typing import Optional
from pydantic import BaseModel, Field
from typing import List 


class TaskInstruction(BaseModel):

    task: List[str]= Field(description="The task mentioned by the user")
    instructions: List[str]= Field(description="The instructions indicated by the user on how to complete the task")

class TasksInstructions(BaseModel):
    """User's task and its instructions."""

    tasks_and_instructions: List[TaskInstruction]= Field(description="The list of tasks mentioned by the user and their corresponding The instructions indicated by the user on how to complete the task")
