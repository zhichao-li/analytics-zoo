import os
import subprocess
import signal
import atexit

class ProcessResource(object):
    pgids=[]
    exception=[]

def session_execute(command, env=None):
    pro = subprocess.Popen(
        command,
        shell=True,
        env=env,
        cwd=None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid)
    pgid = os.getpgid(pro.pid)
    ProcessResource.pgids.append(pgid)
    print("---------------" + str(pgid))
    out, err = pro.communicate()
    out=out.decode("utf-8")
    err=err.decode("utf-8")  # converting bytes to string otherwise \n would not be recognized using str(err)
    errcode = pro.returncode
    if errcode != 0:
        # https://bip.weizmann.ac.il/course/python/PyMOTW/PyMOTW/docs/atexit/index.html
        # http://www.pybloggers.com/2016/02/how-to-always-execute-exit-functions-in-python/    register for signal handling
        raise Exception(err)
    else:
        print(out)
    return out, err
