import subprocess
from typing import Optional


def run_command(command: str, cwd: Optional[str] = None):
    output = subprocess.check_output(command, cwd=cwd)
    print(output.decode())
