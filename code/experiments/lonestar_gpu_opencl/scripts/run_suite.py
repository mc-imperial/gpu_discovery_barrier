import sys
import os
import subprocess
import re
import pdb
import platform
import math
from operator import itemgetter
from wg_size_data import *

def fake_placeholder(x):
    return

EXE_PATH = ""
DATA_PATH = ""
ITERS = 1
PRINT = fake_placeholder
DATA_PRINT = fake_placeholder
MAX_WGS_SIZE = ""
NAME_OF_CHIP = ""
OPTIONAL_ARG = ""
log_file_handle = ""
data_file_handle = ""

HEADER="app-data avg-port min-port max-port stddev-port wgs-port avg-np min-np max-np stddev-np wgs-np wgn-np iters"

PROGRAMS = {
    "bfs",
    "mst",
    "sssp",

    # Because dmr fails on so many systems it is disabled by
    # default. Uncomment this line to include dmr 
#   "dmr"

}

PROGRAM_DATA = {
    "mst" : ["2d-2e20.sym.gr", "USA-road-d.FLA.sym.gr"],
    "bfs" : ["USA-road-d.USA.gr", "r4-2e23.gr", "rmat22.gr"],
    "sssp" : ["USA-road-d.USA.gr", "r4-2e23.gr", "rmat22.gr"],
    #    "dmr" : ["250k.2", "r1M", "r2M"]

    # USA-road dataset uses more memory than some of the GPUs have. These work okay with the
    # modified work list sizes
    #    "sssp" : ["r4-2e23.gr", "rmat22.gr"],
    #    "bfs" : ["r4-2e23.gr", "rmat22.gr"],

    # The r1M and r2M datasets use a ton of memory (~2GB ~8GB), consider just using 250k.2 for
    # smaller chips
    #    "dmr" : ["250k.2"]
    #    "dmr" : ["250k.2", "r1M"]
    #    "dmr" : ["250k.2", "r1M", "r2M"]

}

EXTRA_ARGS = {
    ("dmr-port", "250k.2") : "20",
    ("dmr-port", "r1M") : "20",
    ("dmr-port", "r5M") : "12",

    ("dmr-non-port", "250k.2") : "20",
    ("dmr-non-port", "r1M") : "20",
    ("dmr-non-port", "r5M") : "12"
}

def my_print(file_handle, data):
    print data
    file_handle.write(data + os.linesep)

def get_avg(l):
    return reduce(lambda x, y: x + y, l) / float(len(l))

def execute_prog(prog, data, gb, wgs):
    global log_file_handle
    global data_file_handle

    exe = os.path.join(EXE_PATH,prog+gb)
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

def execute_prog2(prog, data, gb, wgs):
    global log_file_handle
    global data_file_handle

    exe = os.path.join(EXE_PATH,prog)
    data_path = os.path.join(DATA_PATH,data)
    cmd = [exe,data_path,str(wgs[0]),str(wgs[1])]
    if (prog, data) in EXTRA_ARGS:
        extra = EXTRA_ARGS[(prog, data)]
        cmd = cmd + [extra]

    success = 0;        
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

def get_info2(output):
    time = float(parse_res("app runtime = ", "\d+.\d+", output))
    print "found time of:" + str(time)
    return (time,-1)

def variance(l):
    orig_avg = get_avg(l)
    new_l = [(x - orig_avg)**2 for x in l]
    var = get_avg(new_l)

    return math.sqrt(var)

def get_avg_min_max_std(outputs, wgs):
    times = [x[0] for x in outputs]
    avg_time = get_avg(times)
    min_time = min(times)
    max_time = max(times)
    std_dev = variance(times)
    return [str(avg_time), str(min_time), str(max_time), str(std_dev), str(wgs)]

def get_avg_min_max_std2(outputs, wgs):
    times = [x[0] for x in outputs]
    avg_time = get_avg(times)
    min_time = min(times)
    max_time = max(times)
    std_dev = variance(times)
    return [str(avg_time), str(min_time), str(max_time), str(std_dev), str(wgs[0]), str(wgs[1])]

def exe_program_data(prog,data,gb,wgs):
    outputs = []
    for i in range(ITERS):
        output = execute_prog(prog, data, gb, wgs)
        outputs.append(get_info(output))
        return get_avg_min_max_std(outputs, wgs)

def exe_program_data2(prog,data,gb,wgs):
    outputs = []
    for i in range(ITERS):
        output = execute_prog2(prog, data, gb, wgs)
        outputs.append(get_info2(output))
        return get_avg_min_max_std2(outputs, wgs)

def mk_line(p,d,wgs,groups):
    tup_list = ["\"" + x + "\"" for x in [NAME_OF_CHIP, p, d]]
    key = "(" + ",".join(tup_list) + ")"
    value = "(" + str(wgs) + "," + str(groups) + ")"
    return key + " : " + value

def record_data(p_data, p_data2, p, d):
    title = p + "-" + d
    line = " ".join(([title] + p_data + p_data2 + [str(ITERS)]))
    PRINT("recording: " + line)
    DATA_PRINT(line)

def tune_wg_size():
    for p in PROGRAMS:
        for d in PROGRAM_DATA[p]:
            wgs = WGS_TUNED_DATA[(NAME_OF_CHIP,p,d)]

            p_data = exe_program_data(p + "-port",d,"", wgs[0])
            p_data2 = exe_program_data2(p + "-non-port",d,"", wgs)

            record_data(p_data, p_data2, p, d)

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

    if len(sys.argv) < 5:
        print "Please provide the follwing arguments:"
        print "path to portable exe, path to data, name of run, name of chip"
        return 1

    EXE_PATH = sys.argv[1]
    DATA_PATH = sys.argv[2]
    MAX_WGS_SIZE = None
    NAME_OF_CHIP = sys.argv[4]
    log_file = sys.argv[3] + "_exec.log"
    print "recording all to " + log_file
    log_file_handle = open(log_file, "w")
    PRINT = lambda x : my_print(log_file_handle,x)

    data_file = sys.argv[4] + "_data.txt"
    print "recording data to " + data_file
    data_file_handle = open(data_file, "w")
    DATA_PRINT = lambda x : my_print(data_file_handle,x)

    DATA_PRINT(HEADER)

    tune_wg_size()

    log_file_handle.close()
    data_file_handle.close()

if __name__ == '__main__':
    sys.exit(main())
