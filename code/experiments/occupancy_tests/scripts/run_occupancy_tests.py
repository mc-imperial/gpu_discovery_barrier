# Script to run occupancy tests for the discovery protocol and the GPU. 
# args are: 'path to the executables' and 'number of iterations'

import sys
import platform
import subprocess
import pdb
import re
import os
import time
import math

TIMEOUT = 1
SUCCESS = 0
EXE_PATH = ""

LOCAL_CONFIG = ["MIN_LOC", "MAX_LOC"]
WGS_CONFIG = ["MAX_WGS", "MIN_WGS"]

ITERATIONS = 0

def PRINT(s):
    print s

def timeout_cmd_linux(cmd,seconds):
    cmd = ["timeout", str(seconds)] + cmd
    PRINT("running " + " ".join(cmd))
    p_obj = subprocess.Popen(cmd)
    ret_code = p_obj.wait()
    if ret_code == 124:
        return TIMEOUT
    return SUCCESS

def timeout_cmd_windows(cmd,seconds):
    p_obj = subprocess.Popen(cmd)
    time.sleep(seconds)
    if p_obj.poll() == None:
        p_obj.kill()
        time.sleep(3)
        return TIMEOUT
    else:
        return SUCCESS
        

def timeout_cmd(cmd, seconds):
    if platform.system() == "Linux":
        return timeout_cmd_linux(cmd, seconds)
    if platform.system() == "Windows":
        return timeout_cmd_windows(cmd,seconds)
    PRINT("ERROR: unsupported system")
    exit(1)

def check(cmd, wgs, lms, wgn):
    return timeout_cmd([cmd, str(wgn), wgs, lms, "0", "0"], 7)

def find_occupancy(cmd, wgs, lms, search_low, search_high):

    PRINT("Running search with low: " + str(search_low) + " and high: " + str(search_high))
    while 1:
        mid = (search_low + search_high) / 2
        PRINT("mid calculated as: " + str(mid))
        success = check(cmd,  wgs, lms, mid)
        if success == SUCCESS:
            success = check(cmd,  wgs, lms, mid + 1)
            if success == TIMEOUT:
                return mid;
            return find_occupancy(cmd,  wgs, lms, mid+1, search_high)
        if success == TIMEOUT:
            return find_occupancy(cmd,  wgs, lms, search_low, mid-1)

def run_device_query():
    exe = os.path.join(EXE_PATH,"device_query")
    p_obj = subprocess.Popen(exe, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ret_code = p_obj.wait()
    sout, serr = p_obj.communicate()
    print sout
    return sout            

def get_gpu_info():

    data = run_device_query()
    ret = re.findall("DEVICE_NAME: .*", data)
    assert(len(ret) == 1)
    device = ret[0]
    device = " ".join(device.split(" ")[1:]).lstrip()

    ret = re.findall("DEVICE_VENDOR: .*", data)
    assert(len(ret) == 1)
    vendor = ret[0]
    vendor= " ".join(vendor.split(" ")[1:]).lstrip()

    ret = re.findall("DEVICE_LOCAL_MEM_SIZE: .*", data)
    assert(len(ret) == 1)
    local = ret[0]
    ret2 = re.findall("\d+", local)
    assert(len(ret2) == 1)
    local = ret2[0]

    ret = re.findall("DEVICE_MAX_WORK_GROUP_SIZE: .*", data)
    assert(len(ret) == 1)
    wgs = ret[0]
    ret2 = re.findall("\d+", wgs)
    assert(len(ret2) == 1)
    wgs = ret2[0]
        
    ret = re.findall("DEVICE_MAX_COMPUTE_UNITS: .*", data)
    assert(len(ret) == 1)
    sm_num = ret[0]
    ret2 = re.findall("\d+", sm_num)
    assert(len(ret2) == 1)
    sm_num = ret2[0]
    
    return device,local,wgs,sm_num,vendor

def get_wgs(c, gpu_data):
    conf = c[1]
    if conf == "MAX_WGS":
        return str(gpu_data[2])
    else:
        return "1"

def get_lms(c, gpu_data):
    conf = c[0]
    if conf == "MAX_LOC":
        return str(int(gpu_data[1]) - 128)
    else:
        return "1"

def my_exec(exe):
    print "running command: " + " ".join(exe)
    p_obj = subprocess.Popen(exe, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ret_code = p_obj.wait()
    sout, serr = p_obj.communicate()
    if (ret_code != 0):
        PRINT("Error running " + " ".join(cmd))
        exit(ret_code)
        
    return sout

def get_part_group_and_time(s):
    ret = re.findall("kernel ran with a total of \d+ workgroups", s)
    if (len(ret) != 1):
        print "error output: "
        print s
        return None,None
    assert(len(ret) == 1)
    occ = ret[0]
    ret = re.findall("\d+", occ)
    assert(len(ret) == 1)
    occ = ret[0]

    return occ,0.0
    
def run_spin(cmd, wgs, lms):
    exe = [cmd, "1000", wgs, lms, "1", "0"]
    ret = []
    for i in range(int(ITERATIONS)):
        output = my_exec(exe)
        occ,time = get_part_group_and_time(output)
        while occ is None:
            occ,time = get_part_group_and_time(output)
        ret.append((occ,time))
        print "found " + occ + " workgroups"
    return ret;

def run_ticket(cmd, wgs, lms):
    exe = [cmd, "1000", wgs, lms, "1", "1"]
    ret = []
    for i in range(int(ITERATIONS)):
        output = my_exec(exe)
        occ,time = get_part_group_and_time(output)
        ret.append((occ,time))
        print "found " + occ + " workgroups"
    return ret;

def pp_config(c):
    s1 = ""
    s2 = ""
    if c[0] == "MAX_LOC":
        s1 = "max local memory"
    else:
        s1 = "min local memory"
    if c[1] == "MAX_WGS":
        s2 = "max workgroup size"
    else:
        s2 = "min workgroup size"
    s3 = "---"
    return "\n".join([s1,s2,s3])                

def run_configs(gpu_data):
    all_configs = [(x,y) for x in LOCAL_CONFIG for y in WGS_CONFIG]
    cmd = os.path.join(EXE_PATH,"occupancy_test")
    data_list = []
    for c in all_configs:
        data = {}
        data["config"] = pp_config(c)
        wgs = get_wgs(c, gpu_data)
        lms = get_lms(c, gpu_data)
        real_occupancy = find_occupancy(cmd ,wgs, lms, 0, 512)
        data["occupancy"] = real_occupancy
        data["spin_lock"] = run_spin(cmd, wgs, lms)
        data["ticket"] = run_ticket(cmd, wgs, lms)
        data_list.append(data)
    return data_list

def mk_header(gpu_data):
    s1 = "chip: " + gpu_data[0]
    s2 = "vendor: " + gpu_data[4]
    s3 = "local memory:" + gpu_data[1]
    s4 = "max workgroup size:" + gpu_data[2]
    s5 = "number of multiprocessors:" + gpu_data[3]
    s6 = "-----------------------------------"
    return "\n".join([s1,s2,s3,s4,s5,s6])            

def avg(l):
    return reduce(lambda x, y: x + y, l) / float(len(l))

def std_dev(l):    
    orig_avg = avg(l)
    new_l = [(x - orig_avg)**2 for x in l]
    var = avg(new_l)

    return math.sqrt(var)

def int_l(l):
    return [int(x) for x in l]

def pp_data(d):

    s1 = d["config"]
    s2 = "true occupancy: " + str(d["occupancy"])

    spin_lock_occ = [int(x[0]) for x in d["spin_lock"]]
    s3 = "spinlock occupancy list: " + str(int_l(spin_lock_occ))
    s4 = "spinlock average occupancy: " + str(avg(spin_lock_occ))
    s5 = "spinlock occupancy standard deviation: " + str(std_dev(spin_lock_occ))

    ticket_occ = [int(x[0]) for x in d["ticket"]]
    s6 = "ticket occupancy list: " + str(int_l(ticket_occ))
    s7 = "ticket occupancy average: " + str(avg(ticket_occ))
    s8 = "ticket occupancy standard deviation: " + str(std_dev(ticket_occ))

    s9 = "-----------------------------------"
    return "\n".join([s1,s2,s3,s4,s5,s6,s7,s8,s9])

def print_to_file(gpu_data, data):
    s1 = mk_header(gpu_data)
    str_list = [s1]
    for d in data:
        str_list.append(pp_data(d))
        
    to_write =  "\n".join(str_list)
    fname = gpu_data[0].replace(" ", "_") + ".txt"
    fname = fname.replace("\r", "")
    fname = fname.replace("(", "_")
    fname = fname.replace(")", "_")
    f = open(fname,"w")
    f.write(to_write)
    f.close()
        
def main():

    global EXE_PATH
    global ITERATIONS
    
    if len(sys.argv) != 3:
        print "Please provide the follwing arguments:"
        print "path_to_executables iterations"
        return 1

    EXE_PATH = sys.argv[1]
    ITERATIONS = sys.argv[2]
    gpu_data = get_gpu_info()
    data = run_configs(gpu_data)
    print_to_file(gpu_data,data)
    

if __name__ == '__main__':
    sys.exit(main())

