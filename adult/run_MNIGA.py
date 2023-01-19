import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor

import os


def task(gpuID, lam, T, exp, fair_method, mode):
    thread_name = threading.current_thread().getName()
    print(f'task_lam-{lam}_T-{T}_exp-{exp}：doing something')
#     sh = f"CUDA_VISIBLE_DEVICES={gpuID} python main_prune_alexnet.py --fair_method MNIGA --mode eo --lam {lam} --exp {exp} --epochs 20 --T {T}"
    sh = f"CUDA_VISIBLE_DEVICES={gpuID} python main.py --fair_method {fair_method} --mode {mode} --lam {lam} --lam2 0.0 --exp {exp} --epochs 20 --T {T} --sl 1"
    os.system(sh)
    print(f'task_lam-{lam}_T-{T}_exp-{exp}：done')
    

def main():
#   
    lams = [0.0]

    fair_methods = ["van"]
    pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix='Thread')
    counter = 0
    mode = "eo"
    for fair_method in fair_methods:
        if fair_method == "MNIGA":
            Ts = [0]
        else:
            Ts = [0]
        for exp in range(10):
            for lam in lams:
                for T in Ts:
    #                 print(lam)
                    gpuID = 2
                    pool.submit(task, gpuID, lam, T, exp, fair_method, mode)


        
#         sh = f"CUDA_VISIBLE_DEVICES={gpus[j%6]} nohup python repair/repair_compas.py --attr g\&r --p0 $[i] --p1 $[j] --acc_lb 0.5 --percent 0.3 --weight_threshold 0.2 2>&1 > compas_a\&r.log &"
#         os.system(sh)
        
if __name__ == '__main__':
    main()
    
# CUDA_VISIBLE_DEVICES=0 python main_prune.py --fair_method GapReg --mode eop --lam 0.01 --lam2 0.0 --exp 0 --epochs 10 --T 2 --sl 1