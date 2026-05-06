import pandas as pd
import numpy as np

pois = np.array([
    [1,2],
    [5,3],
    [5,6],
    [8,11],
    [1,3],
])



pois = pois[pois[:,0] < pois[:,1]]
print(pois)