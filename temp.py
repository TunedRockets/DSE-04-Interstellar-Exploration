import pandas as pd
import numpy as np
from src.orbit import Orbit, orbit_from_keplerian
from src.get_ISO import get_ISO
from src.utilities import vector_elazr, SGP_SUN, time_2_true, true_2_time, mean_2_true, true_2_mean, EQ_RAD_EARTH, SGP_EARTH
import matplotlib.pyplot as plt
from Trajectories.Rendezvous_dV_requirements import get_data
import math as m
import numpy as np
from src.test_orbits import *

t = Borisov.time_after_periapsis_to_theta(9*31*DAY)
print(Borisov.polar_equation(t)/AU)