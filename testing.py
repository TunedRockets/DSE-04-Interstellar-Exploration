from src.get_ISO import get_ISO

from src.orbit import Orbit, plot_orbit
from src.utilities import AU, SGP_SUN
from src.examples import Earth, Mars, Jupiter
import matplotlib.pyplot as plt
import numpy as np


obs = get_ISO()
# for ob in obs:
#     print(ob)
print(len(obs))

ax = plt.figure().add_subplot(projection='3d')
ax.scatter(0,0,0,color='red', lw=3)

LIMIT = 6*AU
# plot:
for ob in obs[:50]:
    lim = ob.crosses_altitude(LIMIT)
    if lim is None: continue
    if np.linalg.norm(ob.time_to_rv(0)[0]) > LIMIT: continue

    plot_orbit(ax,ob, max_alt=LIMIT)

plot_orbit(ax,Earth, color='Blue')
plot_orbit(ax,Mars,color='Red')
plot_orbit(ax,Jupiter,color='darkorange')

plt.axis('scaled')
plt.show()
