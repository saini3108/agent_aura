import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import streamlit as st

class AuditLogger:
    """Handles audit trail logging for the validation system"""
    
    def __init__(self):
        self.session_key = 'audit_trail'
        self.max_entries = 1000  # Limit to prevent memory issues
        
        # Initialize audit trail in session state if not exists
        if self.session_key not in st.session_state.workflow_state:
            st.session_state.workflow_state[self.session_key] = []
    
    def log_action(self, action: str, details: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log an action to the audit trail"""
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details,
            'metadata': metadata or {},
            'session_id': self._get_session_id(),
            'entry_id': self._generate_entry_id()
        }
        
        # Add to session state audit trail
        audit_trail = st.session_state.workflow_state.get(self.session_key, [])
        audit_trail.append(entry)
        
        # Limit the number of entries to prevent memory issues
        if len(audit_trail) > self.max_entries:
            audit_trail = audit_trail[-self.max_entries:]
        
        st.session_state.workflow_state[self.session_key] = audit_trail
    
    def log_agent_execution(self, agent_name: str, status: str, 
                           execution_time: float, result: Dict[str, Any]) -> None:
        """Log agent execution details"""
        
        metadata = {
            'agent_name': agent_name,
            'execution_status': status,
            'execution_time_seconds': execution_time,
            'result_summary': self._summarize_result(result)
        }
        
        if status == 'success':
            action = f"Agent Executed Successfully"
            details = f"{agent_name} completed execution in {execution_time:.2f} seconds"
        else:
            action = f"Agent Execution Failed"
            details = f"{agent_name} failed execution after {execution_time:.2f} seconds"
            if 'error' in result:
                metadata['error_details'] = result['error']
        
        self.log_action(action, details, metadata)
    
    def log_human_interaction(self, interaction_type: str, user_input: Dict[str, Any]) -> None:
        """Log human interactions"""
        
        metadata = {
            'interaction_type': interaction_type,
            'user_input_summary': self._summarize_user_input(user_input)
        }
        
        action = f"Human Interaction: {interaction_type}"
        details = f"User provided {interaction_type.lower()} input"
        
        self.log_action(action, details, metadata)
    
    def log_validation_metric(self, metric_name: str, metric_value: Any, 
                             threshold: Optional[float] = None, 
                             assessment: Optional[str] = None) -> None:
        """Log validation metrics"""
        
        metadata = {
            'metric_name': metric_name,
            'metric_value': metric_value,
            'threshold': threshold,
            'assessment': assessment
        }
        
        action = f"Validation Metric Calculated"
        details = f"{metric_name}: {metric_value}"
        
        if threshold is not None:
            details += f" (threshold: {threshold})"
        if assessment:
            details += f" - {assessment}"
        
        self.log_action(action, details, metadata)
    
    def log_workflow_event(self, event_type: str, step_name: str, 
                          step_index: int, additional_info: Optional[Dict[str, Any]] = None) -> None:
        """Log workflow-related events"""
        
        metadata = {
            'event_type': event_type,
            'step_name': step_name,
            'step_index': step_index,
            'additional_info': additional_info or {}
        }
        
        action = f"Workflow Event: {event_type}"
        details = f"{event_type} for {step_name} (Step {step_index + 1})"
        
        self.log_action(action, details, metadata)
    
    def log_data_operation(self, operation: str, data_info: Dict[str, Any]) -> None:
        """Log data-related operations"""
        
        metadata = {
            'operation': operation,
            'data_info': data_info
        }
        
        action = f"Data Operation: {operation}"
        details = f"Data {operation.lower()}"
        
        if 'filename' in data_info:
            details += f" - {data_info['filename']}"
        if 'shape' in data_info:
            details += f" (Shape: {data_info['shape']})"
        
        self.log_action(action, details, metadata)
    
    def log_error(self, error_type: str, error_message: str, 
                  context: Optional[Dict[str, Any]] = None) -> None:
        """Log errors with context"""
        
        metadata = {
            'error_type': error_type,
            'error_message': error_message,
            'context': context or {}
        }
        
        action = f"Error: {error_type}"
        details = f"Error occurred: {error_message}"
        
        self.log_action(action, details, metadata)
    
    def log_system_event(self, event: str, description: str, 
                        system_info: Optional[Dict[str, Any]] = None) -> None:
        """Log system-level events"""
        
        metadata = {
            'system_event': event,
            'system_info': system_info or {},
            'timestamp_utc': datetime.utcnow().isoformat()
        }
        
        action = f"System Event: {event}"
        details = description
        
        self.log_action(action, details, metadata)
    
    def get_audit_trail(self, limit: Optional[int] = None, 
                       filter_actions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Retrieve audit trail entries"""
        
        audit_trail = st.session_state.workflow_state.get(self.session_key, [])
        
        # Apply action filter if specified
        if filter_actions:
            audit_trail = [
                entry for entry in audit_trail 
                if any(action.lower() in entry['action'].lower() for action in filter_actions)
            ]
        
        # Apply limit if specified
        if limit:
            audit_trail = audit_trail[-limit:]
        
        return audit_trail
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the audit trail"""
        
        audit_trail = st.session_state.workflow_state.get(self.session_key, [])
        
        if not audit_trail:
            return {
                'total_entries': 0,
                'date_range': None,
                'action_types': {},
                'error_count': 0,
                'agent_executions': 0,
                'human_interactions': 0
            }
        
        # Count action types
        action_types = {}
        error_count = 0
        agent_executions = 0
        human_interactions = 0
        
        for entry in audit_trail:
            action = entry['action']
            
            # Categorize actions
            if 'error' in action.lower():
                error_count += 1
            elif 'agent' in action.lower() and 'executed' in action.lower():
                agent_executions += 1
            elif 'human' in action.lower():
                human_interactions += 1
            
            # Count action types
            action_category = action.split(':')[0] if ':' in action else action
            action_types[action_category] = action_types.get(action_category, 0) + 1
        
        # Get date range
        timestamps = [entry['timestamp'] for entry in audit_trail]
        date_range = {
            'start': min(timestamps),
            'end': max(timestamps)
        } if timestamps else None
        
        return {
            'total_entries': len(audit_trail),
            'date_range': date_range,
            'action_types': action_types,
            'error_count': error_count,
            'agent_executions': agent_executions,
            'human_interactions': human_interactions
        }
    
    def export_audit_trail(self, format_type: str = 'json') -> str:
        """Export audit trail in specified format"""
        
        audit_trail = self.get_audit_trail()
        
        if format_type.lower() == 'json':
            return json.dumps(audit_trail, indent=2, default=str)
        elif format_type.lower() == 'csv':
            # Convert to CSV format
            import pandas as pd
            
            # Flatten the audit trail for CSV
            flattened_entries = []
            for entry in audit_trail:
                flat_entry = {
                    'timestamp': entry['timestamp'],
                    'action': entry['action'],
                    'details': entry['details'],
                    'session_id': entry.get('session_id', ''),
                    'entry_id': entry.get('entry_id', '')
                }
                
                # Add metadata fields
                metadata = entry.get('metadata', {})
                for key, value in metadata.items():
                    flat_entry[f'metadata_{key}'] = str(value)
                
                flattened_entries.append(flat_entry)
            
            df = pd.DataFrame(flattened_entries)
            return df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def clear_audit_trail(self) -> None:
        """Clear the audit trail (use with caution)"""
        
        # Log the clearing action first
        self.log_action(
            "Audit Trail Cleared",
            "Audit trail has been manually cleared",
            {'cleared_at': datetime.now().isoformat()}
        )
        
        # Clear all but the last entry (the clearing action)
        st.session_state.workflow_state[self.session_key] = st.session_state.workflow_state[self.session_key][-1:]
    
    def _get_session_id(self) -> str:
        """Get or create a session ID"""
        
        if 'session_id' not in st.session_state:
            st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return st.session_state.session_id
    
    def _generate_entry_id(self) -> str:
        """Generate a unique entry ID"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        return f"entry_{timestamp}"
    
    def _summarize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize agent result for logging"""
        
        summary = {
            'status': result.get('status', 'unknown'),
            'timestamp': result.get('timestamp', ''),
            'agent': result.get('agent', '')
        }
        
        # Add key metrics if present
        if 'metrics' in result:
            metrics = result['metrics']
            summary['key_metrics'] = {
                'auc': metrics.get('auc'),
                'ks_statistic': metrics.get('ks_statistic'),
                'psi': metrics.get('psi')
            }
        
        # Add error information if present
        if 'error' in result:
            summary['error'] = result['error']
        
        return summary
    
    def _summarize_user_input(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize user input for logging"""
        
        summary = {}
        
        # Summarize different types of user input
        for key, value in user_input.items():
            if isinstance(value, str):
                summary[key] = value[:100] + '...' if len(value) > 100 else value
            elif isinstance(value, (int, float, bool)):
                summary[key] = value
            elif isinstance(value, (list, dict)):
                summary[key] = f"{type(value).__name__} with {len(value)} items"
            else:
                summary[key] = str(type(value).__name__)
        
        return summary
