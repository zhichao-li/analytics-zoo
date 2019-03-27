import os
import subprocess
import signal
import atexit

def session_execute(command, env=None):
    pro = subprocess.Popen(
        command,
        shell=True,
        env=env,
        cwd=None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid)
    out, err = pro.communice()
    errcode = pro.returncode

    def _shutdown():
        os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
    # TODO: are there any other signal we want to handle?
    atexit.register(_shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    if errcode != 0:
        # https://bip.weizmann.ac.il/course/python/PyMOTW/PyMOTW/docs/atexit/index.html
        # http://www.pybloggers.com/2016/02/how-to-always-execute-exit-functions-in-python/    register for signal handling
        raise Exception(err)
    return out, err
