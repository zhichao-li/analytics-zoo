import os
import os
import subprocess
import signal
import atexit
import sys

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

def is_local(sc):
    master = sc._conf.get("spark.master")
    return master == "local" or master.startswith("local[")



# TODO: remove this, as it's copy from yahoo
def get_ip_address():
  """Simple utility to get host IP address."""
  try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip_address = s.getsockname()[0]
  except socket_error as sockerr:
    if sockerr.errno != errno.ENETUNREACH:
      raise sockerr
    ip_address = socket.gethostbyname(socket.getfqdn())
  finally:
    s.close()

  return ip_address