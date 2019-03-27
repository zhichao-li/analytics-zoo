import subprocess

from ray_poc.util import safe_shell_exec


def simple_execute(command, env=None, stdout=None, stderr=None, timeout=None):
    # safe_shell_exec.execute(command=command, env=env, stdout=stdout, stderr=stderr)
    process = subprocess.Popen(
        command,
        shell=True,
        env=env,
        cwd=None,
        stdout=stdout,
        stderr=stderr)
    print("child process id: {}".format(process.pid))
    process.wait(timeout)