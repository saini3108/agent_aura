"""
ValiCred-AI Main Application Entry Point
Clean architecture with agent_aura structure
"""
import sys
import os
from pathlib import Path

# Add paths to system path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / "frontend"))
sys.path.append(str(current_dir / "agent-service"))
sys.path.append(str(current_dir / "shared"))

# Import and run the frontend application
from frontend.app import main

if __name__ == "__main__":
    main()