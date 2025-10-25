#!/usr/bin/env python3
"""
Proper Forge launcher that follows the correct pattern.
This script should be run from the forge directory.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main entry point that runs the script properly."""
    
    # Get the project root
    project_root = Path(__file__).parent.absolute()
    forge_dir = project_root / "forge"
    script_path = project_root / "arihan" / "run_forge.py"
    
    # Check if we're in the right directory
    if not forge_dir.exists():
        print(f"Error: Forge directory not found at {forge_dir}")
        sys.exit(1)
    
    if not script_path.exists():
        print(f"Error: Script not found at {script_path}")
        sys.exit(1)
    
    # Set up environment
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": f"{forge_dir / 'src'}:{forge_dir / 'apps'}",
        "HYPERACTOR_CODEC_MAX_FRAME_LENGTH": "134217728",
        "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"),
        "WANDB_MODE": "offline",
        "MODEL_ID": env.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct"),
        "CHECKPOINT_ROOT": env.get("CHECKPOINT_ROOT", "./checkpoints"),
        "LATEST_POINTER": env.get("LATEST_POINTER", "./checkpoints/LATEST"),
        "TOTAL_OUTER_STEPS": env.get("TOTAL_OUTER_STEPS", "100"),
        "CHECKPOINT_EVERY": env.get("CHECKPOINT_EVERY", "20"),
        "LOCAL_BATCH_SIZE": env.get("LOCAL_BATCH_SIZE", "2"),
        "SEQ_LEN": env.get("SEQ_LEN", "1024"),
        "LEARNING_RATE": env.get("LEARNING_RATE", "5e-6"),
    })
    
    print("Setting up environment...")
    print(f"PYTHONPATH: {env['PYTHONPATH']}")
    print(f"MODEL_ID: {env['MODEL_ID']}")
    print(f"CUDA_VISIBLE_DEVICES: {env['CUDA_VISIBLE_DEVICES']}")
    
    # Change to forge directory and run the script
    print(f"Changing to forge directory: {forge_dir}")
    os.chdir(forge_dir)
    
    print(f"Running script: {script_path}")
    try:
        # Run the script with the proper environment
        result = subprocess.run([
            sys.executable, str(script_path)
        ], env=env, cwd=forge_dir)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error running script: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
