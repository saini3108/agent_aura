import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
import streamlit as st

class WorkflowManager:
    """Manages the agent workflow execution and state"""
    
    def __init__(self, audit_logger):
        self.audit_logger = audit_logger
        self.workflow_steps = [
            'analyst_agent',
            'validator_agent', 
            'documentation_agent',
            'human_review',
            'reviewer_agent',
            'auditor_agent'
        ]
        self.step_names = [
            'Analyst Agent',
            'Validator Agent',
            'Documentation Agent', 
            'Human Review',
            'Reviewer Agent',
            'Auditor Agent'
        ]
    
    def get_workflow_status(self, workflow_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get current workflow status"""
        return {
            'current_step': workflow_state.get('current_step', 0),
            'completed_steps': workflow_state.get('completed_steps', []),
            'total_steps': len(self.workflow_steps),
            'progress_percentage': len(workflow_state.get('completed_steps', [])) / len(self.workflow_steps) * 100,
            'current_step_name': self.step_names[workflow_state.get('current_step', 0)] if workflow_state.get('current_step', 0) < len(self.step_names) else 'Complete'
        }
    
    def can_execute_step(self, step_index: int, context: Dict[str, Any]) -> tuple[bool, str]:
        """Check if a workflow step can be executed"""
        
        # Check if step is valid
        if step_index < 0 or step_index >= len(self.workflow_steps):
            return False, "Invalid step index"
        
        step_name = self.workflow_steps[step_index]
        
        # Check prerequisites for each step
        if step_name == 'analyst_agent':
            if context.get('data') is None:
                return False, "Upload validation data before running Analyst Agent"
            return True, ""
        
        elif step_name == 'validator_agent':
            if 0 not in context.get('completed_steps', []):
                return False, "Complete Analyst Agent step first"
            if context.get('data') is None:
                return False, "Validation data required"
            return True, ""
        
        elif step_name == 'documentation_agent':
            if not context.get('files', {}):
                return False, "Upload documentation files before running Documentation Agent"
            return True, ""
        
        elif step_name == 'human_review':
            # Human review can be executed if at least one previous agent has completed
            completed_steps = context.get('completed_steps', [])
            if len(completed_steps) == 0:
                return False, "Complete at least one agent step before human review"
            return True, ""
        
        elif step_name == 'reviewer_agent':
            # Reviewer agent needs previous outputs
            completed_steps = context.get('completed_steps', [])
            if len(completed_steps) < 2:
                return False, "Complete at least Analyst and Validator agents before review"
            return True, ""
        
        elif step_name == 'auditor_agent':
            # Auditor agent needs most steps completed
            completed_steps = context.get('completed_steps', [])
            if len(completed_steps) < 3:
                return False, "Complete more validation steps before audit"
            return True, ""
        
        return True, ""
    
    def execute_step(self, step_index: int, agents: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific workflow step"""
        
        # Check if step can be executed
        can_execute, error_message = self.can_execute_step(step_index, context)
        if not can_execute:
            return {
                'success': False,
                'error': error_message,
                'step_index': step_index
            }
        
        step_name = self.workflow_steps[step_index]
        agent_name = step_name.replace('_agent', '') if step_name.endswith('_agent') else step_name
        
        # Log step execution start
        self.audit_logger.log_action(
            f"Starting {self.step_names[step_index]}",
            f"Executing workflow step {step_index + 1}",
            {'step_name': step_name, 'agent': agent_name}
        )
        
        try:
            # Handle human review step separately
            if step_name == 'human_review':
                return self._handle_human_review_step(step_index, context)
            
            # Get the appropriate agent
            if agent_name not in agents:
                return {
                    'success': False,
                    'error': f"Agent {agent_name} not found",
                    'step_index': step_index
                }
            
            agent = agents[agent_name]
            
            # Prepare context for agent
            agent_context = {
                'data': context.get('data'),
                'files': context.get('files', {}),
                'previous_outputs': context.get('agent_outputs', {}),
                'workflow_state': context.get('workflow_state', {})
            }
            
            # Execute agent
            result = agent.run(agent_context)
            
            # Log successful execution
            self.audit_logger.log_action(
                f"Completed {self.step_names[step_index]}",
                f"Successfully executed {agent_name} agent",
                {
                    'step_index': step_index,
                    'agent': agent_name,
                    'result_status': result.get('status', 'unknown')
                }
            )
            
            return {
                'success': True,
                'result': result,
                'step_index': step_index,
                'agent_name': agent_name
            }
            
        except Exception as e:
            # Log execution error
            self.audit_logger.log_action(
                f"Failed {self.step_names[step_index]}",
                f"Error executing {agent_name} agent: {str(e)}",
                {
                    'step_index': step_index,
                    'agent': agent_name,
                    'error': str(e)
                }
            )
            
            return {
                'success': False,
                'error': f"Error executing {agent_name} agent: {str(e)}",
                'step_index': step_index
            }
    
    def _handle_human_review_step(self, step_index: int, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the human review step"""
        
        # Check if human feedback exists
        human_feedback = context.get('human_feedback', {})
        step_key = f'step_{step_index}'
        
        if step_key in human_feedback:
            # Human review already completed
            feedback = human_feedback[step_key]
            assessment = feedback.get('assessment', 'Unknown')
            
            self.audit_logger.log_action(
                "Human Review Completed",
                f"Human review assessment: {assessment}",
                feedback
            )
            
            return {
                'success': True,
                'result': {
                    'status': 'completed',
                    'assessment': assessment,
                    'feedback': feedback,
                    'timestamp': feedback.get('timestamp', datetime.now().isoformat())
                },
                'step_index': step_index,
                'agent_name': 'human_reviewer'
            }
        else:
            # Human review pending
            return {
                'success': False,
                'error': "Human review pending - please complete review on Human Review page",
                'step_index': step_index,
                'pending_human_input': True
            }
    
    def advance_workflow(self, workflow_state: Dict[str, Any], step_index: int) -> Dict[str, Any]:
        """Advance the workflow to the next step"""
        
        # Mark current step as completed
        completed_steps = workflow_state.get('completed_steps', [])
        if step_index not in completed_steps:
            completed_steps.append(step_index)
        
        # Determine next step
        next_step = step_index + 1
        if next_step >= len(self.workflow_steps):
            next_step = len(self.workflow_steps)  # Workflow complete
        
        # Update workflow state
        updated_state = workflow_state.copy()
        updated_state['completed_steps'] = completed_steps
        updated_state['current_step'] = next_step
        
        # Log workflow advancement
        self.audit_logger.log_action(
            "Workflow Advanced",
            f"Advanced from step {step_index + 1} to step {next_step + 1}",
            {
                'previous_step': step_index,
                'next_step': next_step,
                'completed_steps': completed_steps
            }
        )
        
        return updated_state
    
    def reset_workflow(self) -> Dict[str, Any]:
        """Reset the workflow to initial state"""
        
        initial_state = {
            'current_step': 0,
            'completed_steps': [],
            'agent_outputs': {},
            'human_feedback': {},
            'validation_results': {},
            'audit_trail': []
        }
        
        self.audit_logger.log_action(
            "Workflow Reset",
            "Workflow has been reset to initial state",
            {'reset_timestamp': datetime.now().isoformat()}
        )
        
        return initial_state
    
    def get_step_dependencies(self, step_index: int) -> List[int]:
        """Get the dependencies for a specific step"""
        
        if step_index < 0 or step_index >= len(self.workflow_steps):
            return []
        
        step_name = self.workflow_steps[step_index]
        
        # Define step dependencies
        dependencies = {
            'analyst_agent': [],
            'validator_agent': [0],  # Depends on analyst
            'documentation_agent': [],
            'human_review': [0, 1, 2],  # Can review after any agent
            'reviewer_agent': [0, 1, 2],  # Needs multiple agents completed
            'auditor_agent': [0, 1, 2, 4]  # Needs most steps completed
        }
        
        return dependencies.get(step_name, [])
    
    def validate_workflow_state(self, workflow_state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the current workflow state"""
        
        validation_result = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        current_step = workflow_state.get('current_step', 0)
        completed_steps = workflow_state.get('completed_steps', [])
        
        # Check if current step is valid
        if current_step < 0 or current_step > len(self.workflow_steps):
            validation_result['is_valid'] = False
            validation_result['issues'].append(f"Invalid current step: {current_step}")
        
        # Check if completed steps are valid
        for step in completed_steps:
            if step < 0 or step >= len(self.workflow_steps):
                validation_result['is_valid'] = False
                validation_result['issues'].append(f"Invalid completed step: {step}")
        
        # Check step dependencies
        for step in completed_steps:
            dependencies = self.get_step_dependencies(step)
            for dep in dependencies:
                if dep not in completed_steps and dep != step:
                    validation_result['warnings'].append(
                        f"Step {step + 1} completed without dependency {dep + 1}"
                    )
        
        # Check if current step should be advanced
        if current_step in completed_steps and current_step < len(self.workflow_steps) - 1:
            validation_result['warnings'].append(
                "Current step is completed but workflow hasn't advanced"
            )
        
        return validation_result
    
    def get_workflow_summary(self, workflow_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get a comprehensive workflow summary"""
        
        status = self.get_workflow_status(workflow_state)
        validation = self.validate_workflow_state(workflow_state)
        
        # Count agent outputs and human feedback
        agent_outputs = workflow_state.get('agent_outputs', {})
        human_feedback = workflow_state.get('human_feedback', {})
        
        # Determine overall status
        if status['current_step'] >= len(self.workflow_steps):
            overall_status = 'Complete'
        elif len(validation['issues']) > 0:
            overall_status = 'Invalid'
        elif status['current_step'] == 3 and 'step_3' not in human_feedback:
            overall_status = 'Pending Human Review'
        elif status['progress_percentage'] > 0:
            overall_status = 'In Progress'
        else:
            overall_status = 'Not Started'
        
        return {
            'overall_status': overall_status,
            'progress_percentage': status['progress_percentage'],
            'current_step_name': status['current_step_name'],
            'completed_steps_count': len(status['completed_steps']),
            'total_steps': status['total_steps'],
            'agent_outputs_count': len(agent_outputs),
            'human_feedback_count': len(human_feedback),
            'validation_issues': len(validation['issues']),
            'validation_warnings': len(validation['warnings']),
            'is_valid': validation['is_valid']
        }
