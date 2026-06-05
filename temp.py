import multiprocessing
import numpy as np
import random
from multiprocessing import Pool
from tqdm import tqdm




if __name__ == "__main__":


    for x in range(100000000):
        print(f"Progress {x}", end="\r")
        
    print('\n')
    print('pool-closed')
