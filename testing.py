from src.get_ISO import get_ISO

from src.orbit import Orbit
from src.utilities import AU, SGP_SUN
import matplotlib.pyplot as plt
import numpy as np


obs = get_ISO()
for ob in obs:
    print(ob)
print(len(obs))

ax = plt.figure().add_subplot(projection='3d')
ax.scatter(0,0,0,color='red', lw=3)

LIMIT = 3*AU
# plot:
for ob in obs:
    lim = ob.crosses_altitude(LIMIT)
    if lim is None: continue

    locus = ob.point_locus(-lim,ob.time_to_theta(0))
    point = ob.time_to_rv(0)[0]
    if np.linalg.norm(point) > LIMIT: continue


    ax.plot(locus[:,0],locus[:,1],locus[:,2])
    ax.scatter(point[0],point[1],point[2])

Earth = Orbit(AU,0.05,0,0,0,0,SGP_SUN)
locus = Earth.point_locus()
ax.plot(locus[:,0],locus[:,1],locus[:,2], color="blue")

plt.axis('scaled')
plt.show()
