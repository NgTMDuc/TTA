import os
import subprocess
import sys

def create_virtualenv(env_name = "ducntm"):
    print(f"Creating virtual environment: {env_name}")
    subprocess.run([sys.executable, "-m", "venv", env_name])
    
    # activate_script = ""
    activate_script = os.path.join(env_name, "bin", "activate")
    print(f"Activating virtual environment: {activate_script}")
    requirements_file = "requirements.txt"
    if not os.path.exists(requirements_file):
        print(f"{requirements_file} not found! Please make sure it exists in the current directory.")
        return
    
    pip_executable = os.path.join(env_name, "bin", "pip")
    print()