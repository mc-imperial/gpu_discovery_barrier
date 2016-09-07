import sys
import os
import subprocess
import re
import pdb
import platform
import time
from operator import itemgetter

def fake_placeholder(x):
    return

EXE_PATH = ""
DATA_PATH = ""
ITERS = 10
PRINT = fake_placeholder
DATA_PRINT = fake_placeholder
MAX_WGS_SIZE = ""
NAME_OF_CHIP = ""
OPTIONAL_ARG = ""
log_file_handle = ""
data_file_handle = ""

PROGRAMS = {
    "bfs",
    "mst",
    "sssp",

    # Because dmr fails on so many systems it is disabled by
    # default. Uncomment this line to include dmr 
#   "dmr"
}

PROGRAM_DATA = {
    "mst" : ["2d-2e20.sym.gr", "USA-road-d.FLA.sym.gr", "rmat12.sym.gr"],
    "bfs" : ["USA-road-d.USA.gr", "r4-2e23.gr", "rmat22.gr"],
    "sssp" : ["USA-road-d.USA.gr", "r4-2e23.gr", "rmat22.gr"],

    # USA-road dataset uses more memory than some of the GPUs have. These work okay with the
    # modified work list sizes
    #"sssp" : ["r4-2e23.gr", "rmat22.gr"],
    #"bfs" : ["r4-2e23.gr", "rmat22.gr"],

    # The r1M and r2M datasets use a ton of memory (~2GB ~8GB), consider just using 250k.2
    # for smaller chips
    #    "dmr" : ["250k.2"]
    #    "dmr" : ["250k.2", "r1M"]
    #    "dmr" : ["250k.2", "r1M", "r2M"]
}

EXTRA_ARGS = {
    ("dmr", "250k.2") : "20",
    ("dmr", "r1M") : "20",
    ("dmr", "r5M") : "12"
}

def my_print(file_handle, data):
    print data
    file_handle.write(data + os.linesep)

def get_avg(l):
    return reduce(lambda x, y: x + y, l) / float(len(l))

def execute_prog(prog, data, gb, wgs):
    global log_file_handle
    global data_file_handle

    exe = os.path.join(EXE_PATH,prog+gb+"-port")
    data_path = os.path.join(DATA_PATH,data)
    cmd = [exe,data_path,str(wgs)]

    if (prog, data) in EXTRA_ARGS:
        extra = EXTRA_ARGS[(prog, data)]
        cmd = cmd + [extra]

    success = 0
    
    while success == 0:
        PRINT("running " + " ".join(cmd))
        p_obj = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ret_code = p_obj.wait()
        sout, serr = p_obj.communicate()
        PRINT("* stdout:")
        PRINT(sout)
        PRINT("* stderr:")
        PRINT(serr)
        if (ret_code != 0):
            PRINT("Error running " + " ".join(cmd))
        else:
            success = 1
            return sout

def parse_res(res, reg, output):
    my_re = res + reg
    ret = re.findall(my_re, output)
    assert(len(ret) == 1)
    ret2 = re.findall(reg, ret[0])
    assert(len(ret2) == 1)
    return ret2[0]

def get_info(output):
    time = float(parse_res("app runtime = ", "\d+.\d+", output))
    part_groups = int(parse_res("number of participating groups = ", "\d+", output))
    print "found time of:" + str(time)
    print "found active groups: " + str(part_groups)
    return (time,part_groups)

def get_avg_and_max(outputs):
    times = [x[0] for x in outputs]
    p_grps = [x[1] for x in outputs]
    avg_time = get_avg(times)
    p_grps = max(p_grps)
    return (avg_time,p_grps)

def exe_program_data(prog,data,gb,wgs):
    outputs = []
    for i in range(ITERS):
        output = execute_prog(prog, data, gb, wgs)
        outputs.append(get_info(output))
        return get_avg_and_max(outputs)

def mk_line(p,d,wgs,groups):
    tup_list = ["\"" + x + "\"" for x in [NAME_OF_CHIP, p, d]]
    key = "(" + ",".join(tup_list) + ")"
    value = "(" + str(wgs) + "," + str(groups) + ")"
    return key + " : " + value

def record_data(p_data, p, d):
    best_wgs = min(p_data,key=lambda x: x[1][0])[0]
    p_groups = min(p_data,key=lambda x: x[1][0])[1][1]
    line_no_gb = mk_line(p,d,best_wgs,p_groups) + ","
    PRINT("recording: " + line_no_gb)
    DATA_PRINT(line_no_gb)

def tune_wg_size():
    for p in PROGRAMS:
        for d in PROGRAM_DATA[p]:
            p_data = []
            i = 32
            divisor = 1
            while i <= MAX_WGS_SIZE/divisor:
                p_data.append((i, exe_program_data(p,d,"", i)))
                i*=2

            record_data(p_data, p, d)

def main():

    global EXE_PATH
    global DATA_PATH
    global PRINT
    global DATA_PRINT
    global MAX_WGS_SIZE
    global NAME_OF_CHIP
    global OPTIONAL_ARG
    global log_file_handle
    global data_file_handle

    if len(sys.argv) < 6:
        print "Please provide the follwing arguments:"
        print "path to executables, path to data, name of run, max work group size, name of chip"
        return 1

    if len(sys.argv) == 7:
        OPTIONAL_ARG = sys.argv[6]

    EXE_PATH = sys.argv[1]
    DATA_PATH = sys.argv[2]
    MAX_WGS_SIZE = int(sys.argv[4])
    NAME_OF_CHIP = sys.argv[5]
    log_file = sys.argv[3] + "_exec.log"
    print "recording all to " + log_file
    log_file_handle = open(log_file, "w")
    PRINT = lambda x : my_print(log_file_handle,x)

    data_file = sys.argv[3] + "_data.txt"
    print "recording data to " + data_file
    data_file_handle = open(data_file, "w")
    DATA_PRINT = lambda x : my_print(data_file_handle,x)
    DATA_PRINT("#Copy this python dictionary entry into wg_size_data.py")

    tune_wg_size()

    log_file_handle.close()
    data_file_handle.close()

if __name__ == '__main__':
    sys.exit(main())
