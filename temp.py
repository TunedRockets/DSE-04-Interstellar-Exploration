import pandas as pd
import numpy as np
from src.orbit import Orbit
from src.get_ISO import get_ISO
from src.utilities import vector_elazr, SGP_SUN
import matplotlib.pyplot as plt
from Trajectories.Rendezvous_dV_requirements import get_data


df = get_data()
df = df[df['rdvz_total'] < 20]

points = []
for idx, row in df.iterrows():

    ISO = Orbit(
        row['parameter'],
        row['e'],
        row['i'],
        row['RAAN'],
        row['arg_p'],
        row['t_p'],
        SGP_SUN
    )


    p = vector_elazr(ISO.hyperbolic_origin(True))
    points.append([p[1],p[0]]) # swap azimuth and elevation
points = np.array(points)

plt.scatter(points[:,0],points[:,1])
plt.xlabel("Right Ascention")
plt.ylabel("Declination")
plt.show()
