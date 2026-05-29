''' 
N2-like estimator for mass and sizing
very quick but basis for more advanced steps.


'''
import math as m



# ==================

def dv2m(dV:float, isp:float, mpl:float, l:float)->float:
    '''dv, specific impulse, payload (& engine) mass, 
    structural mass fraction to total mass'''
    ve = 9.80665 * isp
    e = m.exp(dV/ve)
    mf = mpl*(e-1)/(1+l-l*e) # fuel mass
    m0 = mpl + mf*(1+l) # wet mass
    
    return m0




# ===== config ======

# Antenna:
M_antenna = 50 # [kg] antenna mass
P_antenna = 200 # [w] antenna avg power draw
A_antenna = (2.2**2)*m.pi # [m^2] antenna exposed area


# payload:
M_pl = 100 # [kg] payload and bus mass
P_pl = 190 # [w] payload power draw
A_pl = 2*2 # [m^2] payload (bus) exposed area

M_pl += 200 # [kg]


# propulsion:
Isp_ion = 4150 # [s]
dV_ion = 17 # [km/s] Ion required dv
Am_ion = 0.01 # [m^2/kg] area per mass for ion system (replace with lambda?)
l_ion = 0.05 # [-] ion tank mass fraction
Me_ion = 100 # [kg] ion engine mass
P_ion = 7400 # [w] power per ion engine
F_ion = 0.237 # [N] thrust per ion engine

T_max = 86_000 * 365 # one year

a_min = dV_ion/T_max # [m/s^2] minimum acceleration


Isp_boost = 330 # [s]
dV_boost = 4 # [km/s] boost required dv
Am_boost = 0.01 # [m^2/kg] area per mass for prop system (replace with a lambda?)
l_boost = 0.09 # [-] boost stage mass fraction
Me_boost = 100 # [kg] ion engine mass

# heat shield:
rho_heat = 1750 # [kg/m^3] heat shield density
t_heat = 0.11 # [m] heat shield thickness


# reactor:
Psp_nuke = 134 # [w/kg] reactor power density


# ==================


# === variables ===

M_ion = 100 # [kg] ion propulsion mass (incl-tanks)
A_heat = 50 # [m^2] heat shield area
M_heat = A_heat*rho_heat*t_heat # [kg] heat shield mass
M_nuke = 50 # [kg] reactor mass
M_boost = 1000 # [kg] boost propulsion mass (incl-tanks)
N_ion = 2 # [-] number of ion engines



# loop:
for _ in range(10000):
    # note: dry mass does not include variable tank mass

    print("\n====== New iteration =====\n")

    


    # nuke:
    P_nuke = N_ion*P_ion + P_antenna + P_pl
    M_nuke = P_nuke / Psp_nuke
    print(f"reactor electric power is : {P_nuke:6.1f} W. with mass of {M_nuke:5.1f} kg")

    # get cruse mass:
    M_cruise_dry = M_antenna + M_pl + M_nuke + Me_ion*N_ion
    # get ion stage mass:
    M_cruise_wet = dv2m(dV_ion,Isp_ion,M_cruise_dry,l_ion)

    M_ion = M_cruise_wet - M_cruise_dry + Me_ion*N_ion
    print(f"total ion stage mass: {M_cruise_wet:5.1f} kg, of which {M_ion:5.1f} kg is propulstion")

    # get boost mass:
    M_boost_dry = M_cruise_wet + Me_boost + M_heat
    # get boost wet mass:
    M_boost_wet = dv2m(dV_boost,Isp_boost,M_boost_dry,l_boost)

    M_boost = M_boost_wet - M_boost_dry + Me_boost

    print(f"total boost stage mass: {M_boost_wet:5.1f} kg, of which {M_boost:5.1f} kg is propulstion")


    # get surface area:
    A_heat = A_antenna + A_pl + Am_boost*M_boost + Am_ion*M_ion
    # heat mass:
    M_heat = A_heat*rho_heat*t_heat

    print(f"Heat shield area is: {A_heat}")

    #ion engines:
    N_ion = m.ceil(M_cruise_dry*a_min/F_ion)
    print(f"Ion engine number: {N_ion}")

