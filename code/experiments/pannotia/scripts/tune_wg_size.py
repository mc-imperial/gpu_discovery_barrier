# Script by Tyler Sorensen

# Script for finding a good workgroup size for the pannotia applications.
# As arguments, provide "path to executables" "path to data" "name of run" "max workgroup size" and "name of chip"
# The name of the chip and the name of the run are used for the output files and do
# not need to match the chip exactly. 

import sys
import os
import subprocess
import re
import pdb
import platform
from operator import itemgetter

def fake_placeholder(x):
    return

EXE_PATH     = ""
DATA_PATH    = ""
ITERS        = 10
PRINT        = fake_placeholder
DATA_PRINT   = fake_placeholder
MAX_WGS_SIZE = ""
NAME_OF_CHIP = ""
OPTIONAL_ARG = ""

PROGRAMS = {
    "sssp",
    "bc",
    "color",
    "mis"
}

PROGRAM_DATA = {
    "sssp"  : [os.path.join("sssp", "USA-road-d.NW.gr")],
    "bc"    : [os.path.join("bc", "1k_128k.gr"), os.path.join("bc", "2k_1M.gr")],
    "color" : [os.path.join("color", "G3_circuit.graph"), os.path.join("color", "ecology1.graph")],
    "mis"   : [os.path.join("color", "G3_circuit.graph"), os.path.join("color", "ecology1.graph")]
}

GRAPH_TYPE = {
    "sssp"       : "0",
    "bc"         : "0",
    "color"      : "1",
    "mis"        : "1"}

def my_print(file_handle, data):
    print data
    file_handle.write(data + os.linesep)    

def get_avg(l):
    return reduce(lambda x, y: x + y, l) / float(len(l))

def execute_prog(prog, data, gb, wgs):
    exe = os.path.join(EXE_PATH,prog+gb)
    data_path = os.path.join(DATA_PATH,data)
    cmd = [exe,data_path,GRAPH_TYPE[prog],str(wgs)]
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
        exit(ret_code)
        
    return sout

def get_time(output):
    ret = re.findall("kernel time = \d+.\d+", output)
    assert(len(ret) == 1)
    ret2 = re.findall("\d+.\d+", ret[0])
    assert(len(ret2) == 1)
    PRINT("found time of " + ret2[0])
    return float(ret2[0])

def exe_program_data(prog,data,gb,wgs):
    times = []
    for i in range(ITERS):
        output = execute_prog(prog, data, gb, wgs)
        times.append(get_time(output))
        return get_avg(times)

def mk_line(p,d,wgs,gb):
    tup_list = ["\"" + x + "\"" for x in [NAME_OF_CHIP, p, d, gb]]
    key = "(" + ",".join(tup_list) + ")"
    value = str(wgs)
    return key + " : " + value

def record_data(times_no_gb, times_gb, p, d):
    wgs_no_gb = min(times_no_gb,key=itemgetter(1))[0]
    line_no_gb = mk_line(p,d,wgs_no_gb,"") + ","
    PRINT("recording: " + line_no_gb)
    DATA_PRINT(line_no_gb)

    wgs_gb = min(times_gb,key=itemgetter(1))[0]
    line_gb = mk_line(p,d,wgs_gb,"gb") + ","

    PRINT("recording: " + line_gb)
    DATA_PRINT(line_gb)    
    
def tune_wg_size():
    for p in PROGRAMS:
        for d in PROGRAM_DATA[p]:
            times_no_gb = []
            times_gb = []
            i = 32
            divisor = 1
            if (OPTIONAL_ARG == "ARM" and p in ["bc","sssp-opt"]):
                divisor = 2
            while i <= MAX_WGS_SIZE/divisor:
                times_no_gb.append((i, exe_program_data(p,d,"", i)))
                times_gb.append((i,exe_program_data(p,d,"-gb", i)))
                i*=2
                    
            ref = (p + ".out").replace("-","_");
            gb_out = (p + "_gb.out").replace("-","_")
            os.remove(ref)
            os.remove(gb_out)
            record_data(times_no_gb, times_gb, p, d)
            
def main():

    global EXE_PATH
    global DATA_PATH
    global PRINT
    global DATA_PRINT
    global MAX_WGS_SIZE
    global NAME_OF_CHIP
    global OPTIONAL_ARG

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
