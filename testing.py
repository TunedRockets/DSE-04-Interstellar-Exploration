from src.get_ISO import get_ISO

from src.orbit import Orbit, plot_orbit, trajectory_optimizer
from src.utilities import AU, SGP_SUN, YEAR, DAY
from src.examples import Earth, Mars, Jupiter, Omuamua, to_epoch
import matplotlib.pyplot as plt
import numpy as np
import datetime



ISOs = get_ISO(1)
ax = plt.figure().add_subplot(projection='3d')
print(len(ISOs))
for iso in ISOs:
    print(f"{iso.polar_equation(iso.time_to_theta(YEAR))/AU:.2f}\t\t {iso.periapsis/AU:.2f}")
    plot_orbit(ax,iso, max_alt=5*AU, time=YEAR)
    

plot_orbit(ax, Earth,color="blue")
ax.scatter(0,0,0, color="red", lw=3)
plt.axis('scaled')
plt.show()