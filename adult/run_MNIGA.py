import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor

import os


def task(gpuID, lam, T, exp, fair_method, mode):
    thread_name = threading.current_thread().getName()
    print(f'task_lam-{lam}_T-{T}_exp-{exp}：doing something')

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

        
if __name__ == '__main__':
    main()
