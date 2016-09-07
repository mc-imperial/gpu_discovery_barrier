# Script to run timing tests for the discovery protocol. 
# args are: 'path to the executables' and 'number of iterations'

import sys
import os
import re
import subprocess

EXE_PATH = ""
ITERATIONS = 0
INCREMENT = 8

def get_part_group_and_time(s):

    sys.stdout.flush()
    ret = re.findall("kernel ran with a total of \d+ workgroups", s)
    if len(ret) != 1:
        return None,None
    occ = ret[0]
    ret = re.findall("\d+", occ)
    assert(len(ret) == 1)
    occ = ret[0]

    ret = re.findall("kernel time: \d+.\d+", s)
    assert(len(ret) == 1)
    time = ret[0]
    ret = re.findall("\d+.\d+", time)
    assert(len(ret) == 1)
    time = ret[0]

    return occ,time

def my_exec(exe):
    print "running command: " + " ".join(exe)
    p_obj = subprocess.Popen(exe, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ret_code = p_obj.wait()
    sout, serr = p_obj.communicate()
    if (ret_code != 0):
        print "Error running " + " ".join(exe)
        exit(ret_code)

    print serr
    return sout

def run_device_query():
    exe = os.path.join(EXE_PATH,"device_query")
    p_obj = subprocess.Popen(exe, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ret_code = p_obj.wait()
    sout, serr = p_obj.communicate()
    print sout
    return sout

def run_ticket_max_occ(cmd, wgs, lms):
    exe = [cmd, "1000", wgs, lms, "1"]
    occ_ret = 0
    for i in range(10):
        output = my_exec(exe)
        occ,time = get_part_group_and_time(output)
        occ_ret = max(occ_ret,int(occ))
    return str(occ_ret)

def avg(l):
    return reduce(lambda x, y: x + y, l) / float(len(l))


def avg_run_ticket_time(cmd, wgs, lms):
    exe = [cmd, "1000", wgs, lms, "1"]
    ret = []
    ret_occ = []
    for i in range(int(ITERATIONS)):
        output = my_exec(exe)
        occ,time = get_part_group_and_time(output)
        ret.append(float(time))
        ret_occ.append(float(occ))
        print "found " + occ + " workgroups"
        print "found " + time + " time"
    return avg(ret),avg(ret_occ)

def avg_run_spin_time(cmd, wgs, lms):
    exe = [cmd, "1000", wgs, lms, "0"]
    ret = []
    ret_occ = []
    for i in range(int(ITERATIONS)):
        success = 0
        while success == 0:
            output = my_exec(exe)
            occ,time = get_part_group_and_time(output)
            if occ == None:
                    print "failed a run, re-trying"
                    success = 0
            else:
                success = 1
                ret.append(float(time))
                ret_occ.append(float(occ))
                print "found " + occ + " workgroups"
                print "found " + time + " time"
    return avg(ret),avg(ret_occ)

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

def get_max_wgs(gpu_data):
    return int(gpu_data[2])

def clamp(x):
    if x == 0:
        return 1
    return x

def run_timing(gpu_data):
    wgs = 0
    cmd = os.path.join(EXE_PATH,"time_prot")
    ret = []
    for x in range((get_max_wgs(gpu_data)/INCREMENT)+1):
        wgs = clamp(x * INCREMENT)
        true_occ_est = run_ticket_max_occ(cmd, str(wgs), str(1))
        time_ticket,occ_ticket = avg_run_ticket_time(cmd, str(wgs), str(1))
        time_spin,occ_spin = avg_run_spin_time(cmd, str(wgs), str(1))
        ret.append((str(true_occ_est),str(time_ticket),str(time_spin), str(occ_ticket),str(occ_spin)))
    return ret    

def mk_header(gpu_data):
    return "true_occ ticket_avg_time spin_avg_time ticket_avg_occ spin_avg_occ"

def print_to_file(gpu_data,data):
    fname = gpu_data[0].replace(" ", "_") + "_timing.txt"
    fname = fname.replace("\r", "")
    fname = fname.replace("(", "_")
    fname = fname.replace(")", "_")

    s1 = mk_header(gpu_data)
    str_list = [s1]
    for d in data:        
        line = " ".join(d)
        str_list.append(line)

    to_write = "\n".join(str_list)
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
    data = run_timing(gpu_data)
    print_to_file(gpu_data,data)

if __name__ == '__main__':
    sys.exit(main())
