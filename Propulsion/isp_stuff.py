
# Cold gas


from ReactoPy.CycloPy import Helium, BAR, Xenon, Nitrogen
import math


g0 = 9.81
t0 = 273.15

pe = 0
p0 = 10*BAR

def get_isp(pe, p0, t0, gas):
    gamma = gas.gamma
    rs = gas.R
    pressure_ratio = pe / p0
    exponent = (gamma - 1) / gamma

    velocity_sq = ((2 * gamma) / (gamma - 1)) * rs * t0 * (1 - (pressure_ratio ** exponent))
    ve = math.sqrt(velocity_sq)

    # Calculate Isp (seconds)
    isp = ve / g0
    return isp

print("Helium Isp")
print(get_isp(pe, p0, t0, Helium))

print()

print("Xenon Isp")
print(get_isp(pe, p0, t0, Xenon))

print()

print("Nitrogen Isp")
print(get_isp(pe, p0, t0, Nitrogen))

