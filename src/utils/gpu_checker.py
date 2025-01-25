import subprocess
from src.utils.updates_sender import send_email

def get_gpu_utilization():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        utilization = [int(x) for x in result.stdout.splitlines()]
        return utilization
    except Exception as e:
        print(f"Error querying GPU: {e}")
        return None

def monitor_gpu():
    utilization = get_gpu_utilization()
    if utilization is not None:
        print(f"GPU Utilization: {utilization}")
        if all(u == 0 for u in utilization):  # Check if all GPUs are at 0%
            send_email(
                subject="GPU Utilization Alert",
                body="GPU utilization has dropped to 0%!",
                recipient_email="betello@diag.uniroma1.it"
            )
            send_email(
                subject="GPU Utilization Alert",
                body="GPU utilization has dropped to 0%!",
                recipient_email="purificato@diag.uniroma1.it"
            )
