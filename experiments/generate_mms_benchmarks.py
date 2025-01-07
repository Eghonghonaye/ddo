

from os import listdir
import re
import sys


def write_operations(file,job_id,op_id):
    with open(file, "a") as myfile:
        myfile.write(f"v: o: op({job_id},{op_id});\n")

"""// operations

v: o: op(1,1);
v: o: op(1,2);
v: o: op(2,1);
v: o: op(2,2);"""

def write_machines(file,machine_id):
    with open(file, "a") as myfile:
        myfile.write(f"v: m: m({machine_id});\n")

"""// machines
v: m: m(1);
v: m: m(2);"""

def write_assignments(file,job_id,op_id,machine_id):
    with open(file, "a") as myfile:
        myfile.write(f"c: assign(op({job_id},{op_id}),m({machine_id}));\n")

"""// assignments
c: assign(op(1,1),m(1));
c: assign(op(1,2),m(2));
c: assign(op(2,1),m(1));
c: assign(op(2,2),m(2));"""

def write_constraints():
    pass
"""// constraints"""

def write_processing(file,job_id,op_id,processing):
    with open(file, "a") as myfile:
        myfile.write(f"c: processing(op({job_id},{op_id}),{processing});\n")

"""c: processing(op(1,1),10);
c: processing(op(1,2),15);
c: processing(op(2,1),20);
c: processing(op(2,2),25);
"""

def write_setup():
    pass

"""c: setup(op(1,1),op(1,2),10);
c: setup(op(2,1),op(2,2),10);"""

def write_deadline():
    pass

"""
c: deadline(op(1,2),30);"""

def write_release(file,job_id,op_id,release):
    with open(file, "a") as myfile:
        myfile.write(f"c: release(op({job_id},{op_id}),{release});\n")
"""c: release(op(1,1),0);
c: release(op(1,2),0);
c: release(op(2,1),0);
c: release(op(2,2),0);"""

def write_precedence(file,job_id1,op_id1,job_id2,op_id2):
    with open(file, "a") as myfile:
        myfile.write(f"c: precedence(op({job_id1},{op_id1}),op({job_id2},{op_id2}));\n")
"""c: precedence(op(1,1),op(1,2));
c: precedence(op(2,1),op(2,2));"""

def write_no_repeat(file,job_id,op_id):
    with open(file, "a") as myfile:
        myfile.write(f"c: norepeat(op({job_id},{op_id}));\n")
"""
c: norepeat(op(1,1));
c: norepeat(op(1,2));
c: norepeat(op(2,1));
c: norepeat(op(2,2));"""



def read_demirkol(file):
    outfile = "/home/eaeigbe/Documents/PhD/ddo/resources/mms/" + file.split("/")[-1]

    num_lines = len(open(file).readlines())
    machines = []
    processing = []
    with open(file, "r") as file:

        NB_JOBS, NB_MACHINES = [int(v) for v in file.readline().split()]


        JOBS = [[int(v) for v in file.readline().split()] for i in range(NB_JOBS)]
        print(JOBS)
        print("\n\n\n")
        # Build list of machines. MACHINES[j][s] = id of the machine for the operation s of the job j
        machines = [[JOBS[j][2 * s] for s in range(NB_MACHINES)] for j in range(NB_JOBS)]
        # Build list of durations. DURATION[j][s] = duration of the operation s of the job j
        processing = [[JOBS[j][2 * s + 1] for s in range(NB_MACHINES)] for j in range(NB_JOBS)]

    #write
    for machine in range(NB_MACHINES):
        write_machines(outfile,machine)

    for job_id in range(len(machines)):
        for op_id in range(len(machines[job_id])):
            write_operations(outfile,job_id,op_id)
            write_assignments(outfile,job_id,op_id,machines[job_id][op_id])
            write_processing(outfile,job_id,op_id,processing[job_id][op_id])
            write_release(outfile,job_id,op_id,0)
            write_no_repeat(outfile,job_id,op_id)
            for prec_op_id in range(0,op_id):
                write_precedence(outfile,job_id,prec_op_id,job_id,op_id,)

def main():
    path = "/home/eaeigbe/Documents/PhD/DemirkolBenchmarksJobShop"
    filenames = [ path+"/"+filename for filename in listdir(path) if filename.endswith( "txt" ) ]

    for file in filenames:
        print(file)
        read_demirkol(file)


if __name__ == "__main__":
    main()
