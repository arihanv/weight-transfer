#!/usr/bin/env python3
"""
Launch script for run_forge.py with proper environment setup.
This script sets up the Python path and environment variables needed for Forge.
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """Set up the environment for running Forge scripts."""
    
    # Get the project root directory
    project_root = Path(__file__).parent.absolute()
    forge_dir = project_root / "forge"
    
    # Add forge src to Python path
    forge_src = forge_dir / "src"
    if forge_src.exists():
        sys.path.insert(0, str(forge_src))
    
    # Add forge apps to Python path
    forge_apps = forge_dir / "apps"
    if forge_apps.exists():
        sys.path.insert(0, str(forge_apps))
    
    # Set environment variables
    env = os.environ.copy()
    
    # Required environment variables for Forge
    env.update({
        "PYTHONPATH": f"{forge_src}:{forge_apps}:{env.get('PYTHONPATH', '')}",
        "HYPERACTOR_CODEC_MAX_FRAME_LENGTH": "134217728",  # 128MB for large RPC calls
        "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"),  # Adjust based on your GPUs
        "WANDB_MODE": "offline",  # Use offline mode for wandb
    })
    
    # Optional: Set model and checkpoint paths
    env.update({
        "MODEL_ID": env.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct"),
        "CHECKPOINT_ROOT": env.get("CHECKPOINT_ROOT", "./checkpoints"),
        "LATEST_POINTER": env.get("LATEST_POINTER", "./checkpoints/LATEST"),
        "TOTAL_OUTER_STEPS": env.get("TOTAL_OUTER_STEPS", "100"),
        "CHECKPOINT_EVERY": env.get("CHECKPOINT_EVERY", "20"),
        "LOCAL_BATCH_SIZE": env.get("LOCAL_BATCH_SIZE", "2"),
        "SEQ_LEN": env.get("SEQ_LEN", "1024"),
        "LEARNING_RATE": env.get("LEARNING_RATE", "5e-6"),
    })
    
    return env

def main():
    """Main entry point."""
    print("Setting up Forge environment...")
    
    # Setup environment
    env = setup_environment()
    
    # Get the script path
    script_path = Path(__file__).parent / "arihan" / "run_forge.py"
    
    if not script_path.exists():
        print(f"Error: Script not found at {script_path}")
        sys.exit(1)
    
    print(f"Launching {script_path}")
    print(f"Environment variables:")
    for key, value in env.items():
        if key.startswith(("PYTHONPATH", "MODEL_ID", "CHECKPOINT_ROOT", "CUDA_VISIBLE_DEVICES")):
            print(f"  {key}={value}")
    
    # Run the script
    try:
        result = subprocess.run([
            sys.executable, str(script_path)
        ], env=env, cwd=Path(__file__).parent)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error running script: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
