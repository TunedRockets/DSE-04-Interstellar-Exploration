''' 
Figure out the massof stuff via N2 convergence.

Stealing bits and pieces from the other code

'''
import math as m
from Power.powerinsizeout import reactor

# ==== consts =====

static_mass = 50+100+200
'''[kg] mass of scientific payload, antenna, bus and oter non-varying things'''
static_power_draw = 1600
'''[w] static power draw of non-propulsion equipment'''
static_area = (2.2**2)*m.pi + 2*2
'''[m^2] static exposed area of bus, antenna, etc.'''

# ion system: (http://large.stanford.edu/courses/2025/ph240/tuckey1/docs/nasa-nov17.pdf)
Isp_ion = 4220
'''[s] ion drive isp'''
dV_inclination = 3000
'''[m/s] dv for the inclination change maneuver'''
dV_rdvz = 17_000
'''[m/s] dv for the rendezvous'''
dV_ion = dV_rdvz + dV_inclination
'''[m/s] total dv required by ion system'''
Me_ion = 15 + 36 # NEXT thruster mass
'''[kg] ion engine mass'''
P_ion = 7400
'''[w] power per ion engine'''
F_ion = 0.235
'''[N] thrust per ion engine'''
T_max_inclination = 86_000*365*1.31
'''max time spent on inclination burn'''

T_max_inclination = 86_000*300  # changed from Andres estimate to more pessimistic value


a_min_ion = dV_inclination/T_max_inclination
'''[m/s^2] minimum acceleration of the ion engines'''
l_ion = 0.05
'''[-] ion tank mass fraction'''

# boost system:
Isp_boost = 330
'''[s] boost drive isp'''
dV_boost = 4_000
'''[m/s] total dv required by boost system'''
Me_boost = 100
'''[kg] boost engine mass'''
l_boost = 0.05
'''[-] boost tank mass fraction'''

# heat shield:
rho_heat = 152 # reverse engineers from parker solar probe numbers
'''[kg/m^3] heat shield density'''
t_heat = 0.11
'''[m] heat shield thickness'''
A_heat_margin = 1.1
'''Heat shield area margin (for overhang, etc.)'''


# reactor:
Psp_nuke = 134
'''[w/kg] reactor power density'''

# Stefan–Boltzmann constant
sigma = 5.670374419e-8  # W/m^2/K^4

# Inputs
T_cold = 1298.0679247865448  # K
areal_density = 15           # kg/m^2
emissivity = 0.9

# Power areal density (W/m^2)
q = emissivity * sigma * T_cold**4

# Specific power (W/kg)
rad_specific_power = q / areal_density


def dv2mf(dV:float, isp:float, m1:float, l:float)->float:
    '''dv [in km/s], specific impulse, non-tank-mass, 
    structural mass fraction to fuel mass'''
    ve = 9.80665 * isp
    e = m.exp(dV/ve)
    mf = m1*(e-1)/(1+l-l*e) # fuel mass
    return mf

@staticmethod
class Hestia():
    '''this is the design to configure, as a class,
    each variable has a method to set itself, which is run through
    every iteration. once iterations have converged it will terminate'''

    def __init__(self) -> None:
        
        # the varying variables 
        self.Mass_ion = 51
        '''the ion engines and tanks (not fuel)'''
        self.Mass_ion_fuel = 200
        '''fuel mass of xenon'''
        self.Area_heatshield = 21
        '''area of the heat shield'''
        self.Mass_boost = 106
        '''the boost stage, engines and tanks (not heat shield, or fuel)'''
        self.Mass_boost_fuel = 200
        '''fuel mass of MON/MMH or w/e we're using'''
        self.Mass_power_truss = 58
        '''mass of nuke, truss, and radiators'''
        self.Power_provided = 0
        '''power provided by the nuke truss'''
        self.Number_ions = 1
        '''Number of ion engines'''

    def __repr__(self) -> str:
        return (
            '--- Hestia configuration: ---\n'
            f'payload mass: {self.upper_stage_pl_mass:6.1f} kg\n'
            f'ion dry mass: {self.upper_stage_dry_mass:6.1f} kg\n'
            f'ion wet mass: {self.upper_stage_wet_mass:6.1f} kg\n'
            '---\n'
            f'boost sys mass: {self.lower_stage_pl_mass:6.1f} kg\n'
            f'boost dry mass: {self.lower_stage_dry_mass:6.1f} kg\n'
            f'boost wet mass: {self.lower_stage_wet_mass:6.1f} kg\n'
            '---\n'
            f'{self.Number_ions} ion engines\n'
            f'inclination burn time: {self.inclination_burn_time/86_000:3.2f} days\n'
            f'rendezvous burn time: {self.rdvz_burn_time/86_000:3.2f} days\n'
            f'{self.Power_provided:6.1f} W used from reactor with mass {self.Mass_power_truss:6.1f} kg\n'
            '---\n'
            f'total heat shield area of {self.Area_heatshield:3.3f} m^2, with mass {self.Mass_heatshield:6.1f} kg'
        )


    def _converge(self, max_iter:int=1000):
        '''run the convergence'''

        for _ in range(max_iter):

            if self._iterate():
                print('\n!!! conversion finished !!!\n\n\n\n')
                print(self)
                return
        else:
            raise TimeoutError("Did not converge in time")

    def _iterate(self)->bool:
        '''runs through all iteration methods'''

        var_dict = self.__dict__.copy()

        print("\n====== New iteration =====\n")
        mydir = dir(self)
        myfuncs = [fn for fn in mydir if callable(getattr(self, fn))]
        myfuncs = [fn for fn in myfuncs if fn.startswith('size')]
        for fn in myfuncs: # all the sizing funcs
            fn_call = getattr(self,fn)
            fn_call() # call function

        # check for convergence
        converged = True
        for key, value in var_dict.items():

            if abs(value - self.__dict__[key]) > 1e-8:
                converged = False
        return converged

    # property methods are fine as long as there are no side-effects

    @property
    def upper_stage_pl_mass(self):
        '''mass of upper stage w/o propulsion system'''
        return static_mass + self.Mass_power_truss
    
    @property
    def upper_stage_dry_mass(self):
        '''mass of upper stage w/o propulsion system'''
        return self.upper_stage_pl_mass + self.Mass_ion

    @property
    def upper_stage_wet_mass(self):
        '''mass of upper stage w/o propulsion system'''
        return self.upper_stage_dry_mass + self.Mass_ion_fuel

    @property
    def lower_stage_pl_mass(self):
        '''mass of lower stage w/o prop system (upper stage and heat shield)'''
        return self.upper_stage_wet_mass + self.Mass_heatshield
    
    @property
    def lower_stage_dry_mass(self):
        '''mass of upper stage w/o propulsion system'''
        return self.lower_stage_pl_mass + self.Mass_boost

    @property
    def lower_stage_wet_mass(self):
        '''mass of upper stage w/o propulsion system'''
        return self.lower_stage_dry_mass + self.Mass_boost_fuel

    @property
    def Mass_heatshield(self):
        '''mass of heat shield'''
        return self.Area_heatshield * t_heat * rho_heat

    @property
    def inclination_burn_time(self):
        '''pessemistic estimate of burn time'''
        return dV_inclination/(self.Number_ions*F_ion/self.lower_stage_wet_mass)
    
    @property
    def rdvz_burn_time(self):
        '''pessemistic estimate of burn time'''
        return dV_rdvz/(self.Number_ions*F_ion/self.upper_stage_wet_mass)


    def size_ion_system(self):
        '''size the ion system and figure out number of engines and power draw'''

        # get no. engines and their mass:
        F_need = self.lower_stage_wet_mass*a_min_ion
        self.Number_ions = m.ceil(F_need/F_ion)
        # set new ion mass:
        self.Mass_ion = (l_ion*self.Mass_ion_fuel) + self.Number_ions*Me_ion


        m_rdzv = dv2mf(dV_rdvz, Isp_ion, self.upper_stage_pl_mass+ self.Number_ions*Me_ion, l_ion)

        m_plane = dv2mf(dV_inclination, Isp_ion, self.lower_stage_dry_mass + ((1+l_ion) * m_rdzv) + self.Number_ions * Me_ion, l_ion)

        mf = m_plane + m_rdzv
        self.Mass_ion_fuel = mf

        print(f"ion engine number: {self.Number_ions}, xenon: {self.Mass_ion_fuel:5.1f} kg")

    def size_boost_system(self):
        '''size boost fuel tank and rest'''

        m1 = self.lower_stage_pl_mass + Me_boost
        mf = dv2mf(dV_boost, Isp_boost, m1, l_boost)
        self.Mass_boost_fuel = mf

        print(f'boost fuel: {self.Mass_boost_fuel:5.1f} kg, total wet mass: {self.lower_stage_wet_mass:5.1f} kg')

    def size_power_system(self):
        '''uses only simple power density, include better system later'''

        Preq = static_power_draw + self.Number_ions*P_ion # needed power

        reactor_mass, reactor_fuel_mass, thermal_power = reactor(Preq)

        reactor_mass += reactor_fuel_mass
        self.Mass_power_truss = reactor_mass
        self.Power_provided = Preq

        # Radiator
        disipated_power = thermal_power - Preq

        radiator_mass = disipated_power/rad_specific_power

        self.Mass_power_truss += radiator_mass

        print(f'reactor truss weight: {self.Mass_power_truss:5.1f} kg, generating: {Preq:5.1f} W')
        print(f'thermal power: {thermal_power:5.1f} W')
        print(f'radiator mass: {radiator_mass:5.1f} kg')
        print(f'radiator area: {radiator_mass/areal_density:5.1f} m2')
    def size_heat_shield(self):
        '''uses very simple model, include better system later'''

        A = static_area

        # cylinder mass to area:
        # V = pi*r*r*h
        # A = 2*r*h
        # V = m/rho
        # ==>
        # A = 2*h*sqrt(V/(pi*h))
        # ==> 
        A_fn = lambda mass,h,rho: 2*m.sqrt(mass*h/(m.pi*rho))

        # VERY APPROXIAMTE VALUES!!! CHANGE

        A += A_fn(self.Mass_power_truss,9,7000)
        # power truss approximated as cylinder half ariane6 fairing
        # with density of steel (average of reactor + truss)

        A += A_fn(self.Mass_boost_fuel,3, 1000) # 3 m cyliner of fuel
        A += A_fn(self.Mass_ion_fuel,2,1500) # 2 m cyliner of xenon

        A *= A_heat_margin # margin

        self.Area_heatshield = A
        self.Mass_heatshield

        print(f'heat shield area is: {A:3.2f} m^2 with a mass of {self.Mass_heatshield:6.1f} kg')



if __name__ == "__main__":
    SC = Hestia()

    SC._converge()