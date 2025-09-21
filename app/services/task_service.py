# app/services/task_service.py
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from app.models.schemas import TaskStatus, TaskStatusResponse
import json
import logging

logger = logging.getLogger(__name__)

class TaskService:
    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}
    
    def create_task(self, task_type: str, **kwargs) -> str:
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {
            'task_id': task_id,
            'task_type': task_type,
            'status': TaskStatus.PENDING,
            'progress': 0.0,
            'message': 'Task created',
            'result': None,
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            **kwargs
        }
        logger.info(f"Created task {task_id} of type {task_type}")
        return task_id
    
    def update_task(self, task_id: str, status: TaskStatus = None, 
                   progress: float = None, message: str = None, 
                   result: Any = None):
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        if status:
            task['status'] = status
        if progress is not None:
            task['progress'] = progress
        if message:
            task['message'] = message
        if result is not None:
            task['result'] = result
        
        task['updated_at'] = datetime.now()
        logger.info(f"Updated task {task_id}: status={status}, progress={progress}")
        return True
    
    def get_task(self, task_id: str) -> Optional[TaskStatusResponse]:
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        return TaskStatusResponse(**task)
    
    def delete_task(self, task_id: str) -> bool:
        if task_id in self.tasks:
            del self.tasks[task_id]
            logger.info(f"Deleted task {task_id}")
            return True
        return False
task_service = TaskService()