import subprocess
from typing import Optional


def run_command(command: list[str], cwd: Optional[str] = None, raise_on_error: bool = True):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd
    )
    output, error = process.communicate()
    if process.returncode != 0 and raise_on_error:
        raise Exception("File handling failed %d %s %s" %
                        (process.returncode, output, error)
                        )

    print(output.decode())