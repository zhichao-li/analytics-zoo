# cleaner
# def cleaner(_):
#     try:
#         os.remove(os.path.join(os.getcwd(), master_file))
#     except OSError:
#         pass
#     yield []
#
# sc.range(0, NUM_WORKERS + 1, numSlices=NUM_WORKERS + 1).barrier().mapPartitions(
#     cleaner).collect()


def check_if_redis_ready():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((master_ip, int(REDIS_PORT)))
    sock.close()
    if result == 0:
        print("Port is open")
        return True
    else:
        print("Port is not open")
        return False


import psutil
import socket


def addresses():
    result = {}
    for intf, intf_addresses in psutil.net_if_addrs().items():
        for addr in intf_addresses:
            if addr.family == socket.AF_INET:
                if intf not in result:
                    result[intf] = []
                result[intf].append(addr.address)
    return result


# check if file exist

try:
    print("working dir: {}".format(os.getcwd()))
    if (not os.path.isfile(master_file)):
        with open(os.path.join(os.getcwd(), master_file), 'w') as f:
            f.write('master is created.')
            f.close()
        command = "{} start --head --redis-port {}".format(RAY_COMMAND, REDIS_PORT)
        print(command)
        # TODO redis port should be randomly searched
        execute(command)
        return
finally:
    print("Master and worker are sharing the same node")