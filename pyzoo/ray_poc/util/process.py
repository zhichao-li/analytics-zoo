import os
import subprocess
import signal
import atexit
import sys

class ProcessInfo(object):
    def __init__(self, out, err, errorcode, pgid, pid=None, node_ip=None):
        self.out=out
        self.err=err
        self.pgid=pgid
        self.pid=pid
        self.errorcode=errorcode
        self.tag="default"
        self.master_addr=None
        self.node_ip=node_ip

def session_execute(command, env=None, tag=None):
    pro = subprocess.Popen(
        command,
        shell=True,
        env=env,
        cwd=None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid)
    pgid = os.getpgid(pro.pid)
    print("The pgid for the current session is: {}".format(pgid))
    out, err = pro.communicate()
    out=out.decode("utf-8")
    err=err.decode("utf-8")  # converting bytes to string otherwise \n would not be recognized using str(err)
    errorcode=pro.returncode
    if errorcode != 0:
        # https://bip.weizmann.ac.il/course/python/PyMOTW/PyMOTW/docs/atexit/index.html
        # http://www.pybloggers.com/2016/02/how-to-always-execute-exit-functions-in-python/    register for signal handling
        print(err)
    else:
        print(out)
    return ProcessInfo(out, err, pro.returncode, pgid, tag)


def _gen_shutdown_per_node(pgids):
    def _shutdown_per_node(iter):
        print("shutting down pgid: {}".format(pgids))
        for pgid in pgids:
            print("killing {}".format(pgid))
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                print("WARNING: cannot find pgid: {}".format(pgid))

    return _shutdown_per_node

class ProcessMonitor:
    # what if half of the services is started and then exception happen?
    def __init__(self, process_infos, sc):
        self.master = []
        self.slaves = []
        self.pgdis=[] # TODO: change me to dict
        for process_info in process_infos:
            self.pgids.append(process_info.pgid)
            if process_info.master_addr:
                self.master.append(process_info)
            else:
                self.slaves.append(process_info)
        self.register_shutdown_hook()
        # throw exception if any
        assert len(self.master) == 1, "We should got 1 master only, but we got {}".format(len(self.master))

        if self.master.errorcode != 0:
            raise Exception(self.master.err)
        for slave in self.slaves:
            if slave.errorcode != 0:
                raise Exception(slave.err)

        print(self.master.out)
        for slave in self.slaves:
            # TODO: implement __str__ for class ProcessInfo
            print(slave.out)
        return self.master.master_addr

    def register_shutdown_hook(self):
        def _shutdown():
            self.sc.range(0,
                          self.num_executors,
                          numSlices=self.num_executors).map(_gen_shutdown_per_node(self.pgdis)).collect()

        def _signal_shutdown(_signo, _stack_frame):
            _shutdown()
            sys.exit(0)

        # TODO: are there any other signal we want to handle?
        atexit.register(_shutdown)
        signal.signal(signal.SIGTERM, _signal_shutdown)
        signal.signal(signal.SIGINT, _signal_shutdown)

        # pgids=list(itertools.chain.from_iterable(pgids))
        # exceptions=list(itertools.chain.from_iterable(exceptions))
