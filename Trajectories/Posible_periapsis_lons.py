import jkat as jk
import numpy as np

from Trajectories.Rendezvous_dV_requirements import MAX_MISSION_TIME
from src.helio_optim import *
from Rendezvous_dV_requirements import get_parking

AU = jk.AU
DAY = jk.DAY
YEAR = jk.YEAR
MAX_MISSION_TIME = 10

def job(ISOtuple:tuple[jkat.Orbit, float, str], longp_num:int, dv_budget:tuple[float, float, float])->dict:

    dv_inc, dv_oberth, dv_rendezvous = dv_budget

    np.seterr(all="ignore")
    ISO, detect_t, g_type = ISOtuple

    detect_r = ISO.r(ISO.f(detect_t))/AU
    longps = np.linspace(-m.pi, m.pi, longp_num)
    for longp in longps:
        name = f"{m.degrees(longp):3.1f}"
        try:
            possible = check_if_possible(dv_inc, dv_oberth, dv_rendezvous, get_parking(longp), ISO, ISO.tp + MAX_MISSION_TIME*YEAR)

        except (ArithmeticError, ValueError, AssertionError): pass
