import subprocess
import sys
from pathlib import Path

# Get the directory where this runner script lives
REPO_ROOT = Path(__file__).resolve().parent

# Define the relative scripts to run in order
SCRIPTS = [
    "plot_midterm_map.py",
    "plot_temperature_correlation.py",
    "plot_time_series.py",
    "police_boundaries.py",
    "seasonal_volatility.py",
    "temperature_correlation_residuals.py",
    "tourism_correlation_residuals.py",
    "visual_analysis.py"
]

def run_script(script_name):
    script_path = REPO_ROOT / script_name
    
    if not script_path.exists():
        print(f"[-] Skipped: {script_name} (File not found at {script_path})")
        return False

    print(f"\n==================================================")
    print(f"RUNNING: {script_name}")
    print(f"==================================================")
    
    try:
        # Run the script using the current Python interpreter
        # cwd=REPO_ROOT ensures relative data paths like 'data/SHP/...' work perfectly
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(REPO_ROOT),
            check=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"[-] Error executing {script_name}: {e}")
        return False

if __name__ == "__main__":
    print("Starting master visualization suite...")
    success_count = 0
    
    for script in SCRIPTS:
        if run_script(script):
            success_count += 1
            
    print(f"\n==================================================")
    print(f"Execution finished. Successfully ran {success_count}/{len(SCRIPTS)} scripts.")
    print(f"==================================================")