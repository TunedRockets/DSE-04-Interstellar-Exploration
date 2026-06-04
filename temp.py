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


    with Pool() as pool:

        r = pool.apply_async(sto)


        res = pool.imap_unordered(get_frac, tqdm([10000 for _ in range(100000)], desc='finding pi'),chunksize=100)
        sum = 0
        count = 0
        for r in res:
            sum += r
            count += 1
        avg = sum/count
        print(avg*4)


    print('pool-closed')
