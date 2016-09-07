# Script by Tyler Sorensen

# Script for running the pannotia applications.
# As arguments, provide "path to executables" "path to data" "name of run" and "name of chip"
# The name of the chip and the name of the run are used for the output files and do
# not need to match the chip exactly. 

import sys
import os
import subprocess
import re
import pdb
import platform
from wg_size_data import *

def fake_placeholder(x):
    return

EXE_PATH     = ""
DATA_PATH    = ""
ITERS        = 1
PRINT        = fake_placeholder
DATA_PRINT   = fake_placeholder
NAME_OF_CHIP = ""

PROGRAMS = {
    "sssp",
    "bc",
    "color",
    "mis"
}

PROGRAM_DATA = {
    "sssp"   : [os.path.join("sssp", "USA-road-d.NW.gr")],
    "bc"     : [os.path.join("bc", "1k_128k.gr"), os.path.join("bc", "2k_1M.gr")],
    "color"  : [os.path.join("color", "G3_circuit.graph"), os.path.join("color", "ecology1.graph")],
    "mis"    : [os.path.join("color", "G3_circuit.graph"), os.path.join("color", "ecology1.graph")]
}

GRAPH_TYPE = {
    "sssp"  : "0",
    "bc"    : "0",
    "color" : "1",
    "mis"   : "1"}

COMP_METHODS = {
    "sssp"   : "diff",
    "bc"     : "aprx",
    "color"  : "diff",
    "mis"    : "diff" }

def file_to_string(f):
    f = open(f,'r')
    data = f.read()
    f.close()
    return data

def get_avg(l):
    return reduce(lambda x, y: x + y, l) / float(len(l))

def get_max(l):
    return max(l)

def get_min(l):
    return min(l)

def record_data(times_no_gb, times_gb, p, d, occ):
    name = p + "-" + os.path.basename(d)
    wgs_no_gb = get_wgs_size(p,d,"")
    wgs_gb = get_wgs_size(p,d,"-gb")

    avg_no_gb = get_avg(times_no_gb)
    avg_gb = get_avg(times_gb)

    min_no_gb = get_min(times_no_gb)
    min_gb = get_min(times_gb)
    
    max_no_gb = get_max(times_no_gb)
    max_gb = get_max(times_gb)

    data_list = [name, avg_no_gb, avg_gb, min_no_gb, min_gb, max_no_gb, max_gb, wgs_no_gb, wgs_gb, str(occ)]
    line = " ".join([str(x) for x in data_list])
    PRINT("recording: " + line)
    DATA_PRINT(line)

def execute_linux_diff(f1,f2):
    cmd = ["diff", f1, f2]
    PRINT("running " + " ".join(cmd))
    p_obj = subprocess.Popen(cmd)
    ret_code = p_obj.wait()
    if (ret_code != 0):
        PRINT("VALIDATION FAILED")
        exit(1)    

def get_wgs_size(p,d,gb):
    key = (NAME_OF_CHIP,p,d,gb.replace("-",""))
    if key in WGS_TUNED_DATA:
        return str(WGS_TUNED_DATA[key])
    return "256"

def execute_prog(prog, data, gb):
    exe = os.path.join(EXE_PATH,prog+gb)
    data_path = os.path.join(DATA_PATH,data)
    wgs = get_wgs_size(prog,data,gb)
    cmd = [exe,data_path,GRAPH_TYPE[prog],wgs]
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

def get_time_and_groups(output,occ):
    occ_ret = ""
    ret = re.findall("kernel time = \d+.\d+", output)
    assert(len(ret) == 1)
    ret2 = re.findall("\d+.\d+", ret[0])
    assert(len(ret2) == 1)
    PRINT("found time of " + ret2[0])

    if occ:
        ret3 = re.findall("kernel ran with a total of \d+ workgroups", output)
        assert(len(ret3) == 1)
        ret4 = re.findall("\d+", ret3[0])
        assert(len(ret4) == 1)
        PRINT("found active groups of " + ret4[0])
        occ_ret = int(ret4[0])
        

    return float(ret2[0]),occ_ret

def compare_aprx(f1,f2):
    PRINT("doing approximate floating point comparison between " + f1 + " and " + f2)
    f1_data = file_to_string(f1).split('\n')
    f2_data = file_to_string(f2).split('\n')
    if len(f1_data) != len(f2_data):
        PRINT("files have a different number of lines")
        PRINT("VALIDATION FAILED")
        exit(1)

    f1_data = [x for x in f1_data if x != '']
    f2_data = [x for x in f2_data if x != '']

    allowed_error = .01
    for i in range(len(f1_data)):
        l1 = float(f1_data[i])
        l2 = float(f2_data[i])
        if abs(l1 - l2) > allowed_error:
            PRINT("> line " + str(i) + " differs")
            PRINT("> >" + str(l1) + " " + str(l2))
            PRINT("VALIDATION FAILED")
            exit(1)

def execute_windows_diff(f1,f2):
    cmd = ["fc.exe", f1, f2]
    PRINT("running " + " ".join(cmd))
    p_obj = subprocess.Popen(cmd)
    ret_code = p_obj.wait()
    if (ret_code != 0):
        PRINT("VALIDATION FAILED")
        exit(1)        
        

def compare(f1, f2, method):
    if method == "diff" and platform.system() == "Linux":
        execute_linux_diff(f1,f2)
        return
        if method == "aprx":
            compare_aprx(f1,f2)
            return
            if method == "diff" and platform.system() == "Windows":
                execute_windows_diff(f1,f2)
                return
                
    PRINT("UNABLE TO COMPARE")
    exit(1)        

def validate(p,gb):
    if gb == "":
        return 
        ref = (p + ".out").replace("-","_");
        gb_out = (p + "_gb.out").replace("-","_")
        comp_method = COMP_METHODS[p]
        compare(ref, gb_out, comp_method)
        
def exe_program_data(prog,data,gb,occ):
    times = []
    g = ""
    for i in range(ITERS):
        output = execute_prog(prog, data, gb)
        t,g = get_time_and_groups(output, occ)
        times.append(t)        
        validate(prog,gb)
        return times,g
        
def run_suite():
    for p in PROGRAMS:
        for d in PROGRAM_DATA[p]:
            times_no_gb,occ = exe_program_data(p,d,"", False)
            times_gb,occ = exe_program_data(p,d,"-gb", True)
            record_data(times_no_gb, times_gb, p, d, occ)

        ref = (p + ".out").replace("-","_");
        gb_out = (p + "_gb.out").replace("-","_")
        os.remove(ref)
        os.remove(gb_out)


def my_print(file_handle, data):
    print data
    file_handle.write(data + os.linesep)    

def main():

    global EXE_PATH
    global DATA_PATH
    global PRINT
    global DATA_PRINT
    global NAME_OF_CHIP

    if len(sys.argv) != 5:
        print "Please provide the follwing arguments:"
        print "path to executables, path to data, name of run, name of chip"
        return 1
        
    EXE_PATH = sys.argv[1]
    DATA_PATH = sys.argv[2]
    NAME_OF_CHIP = sys.argv[4]
    log_file = sys.argv[3] + "_exec.log"
    print "recording all to " + log_file    
    log_file_handle = open(log_file, "w")
    PRINT = lambda x : my_print(log_file_handle,x)

    data_file = sys.argv[3] + "_data.txt"
    print "recording data to " + data_file
    data_file_handle = open(data_file, "w")
    DATA_PRINT = lambda x : my_print(data_file_handle,x)
    DATA_PRINT("name avg-no-gb avg-gb max-no-gb max-gb min-no-gb min-gb wgs-no-gb wgs-gb pg")

    run_suite()
    log_file_handle.close()
    data_file_handle.close()
    
if __name__ == '__main__':
    sys.exit(main())
