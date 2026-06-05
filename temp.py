import multiprocessing
import numpy as np
import random
from multiprocessing import Pool
from tqdm import tqdm


def sto()->bool:
    x = random.random()*2 - 1
    y = random.random()*2 - 1

    return x*x + y*y < 1




def get_frac(n:int)->float:
    res:list[bool] = []
    for _ in range(n):
        res.append(sto())
    return res.count(True)/len(res)

def worker(queue):

    n = 1000
    frac = get_frac(n)
    queue.put((n,frac))




if __name__ == "__main__":


    print(list([]))

    print('pool-closed')
