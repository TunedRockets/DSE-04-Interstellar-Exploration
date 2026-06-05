import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import pandas as pd


class Reactor:
    # === Constants ===
    AVOGADRO        = 6.02214076e23          # 1/mol https://www.nist.gov/si-redefinition/meet-constants
    J_PER_EV        = 1.602176634e-19        # J/eV
    FISSION_MEV     = 169.1e6               # eV (energy per U-235 fission) https://web.archive.org/web/20190505175631/http://www.kayelaby.npl.co.uk/atomic_and_nuclear_physics/4_7/4_7_1.html
    M_U235          = 0.235                  # kg/mol
    STEFAN_BOLTZMANN = 5.670374419e-8        # W/ m^2 / K^4

    def __init__(
        self,
        coolant_inlet_temp,  # K  helium inlet temperature
        core_temp,          # K  peak coolant outlet temperature
        heat,               # W  required thermal power output
        enrichment=0.95,    # fraction of U235 in fuel (0–1) 
        burnup=0.15,        # fraction of fuel burned at EoM
        lifetime_s = 315360000 * 2, # operational period before refuel (s); default = 20 year
        operating_pressure = 2e6, # Pa helium pressure
        power_density=11e6,  # W/m^3 volumetric power density Prometheus
        uranium_pebble_fraction = 0.95, # UO2 pellets
        allowable_stress = 146e6, # from ASME Table 1B NO6617
        helium_data_path = "data/helium_prop.csv"
    ):
        self.core_temp         = core_temp
        self.heat_out          = heat
        self.enrichment        = enrichment
        self.burnup            = burnup
        self.lifetime_s        = lifetime_s
        self.coolant_inlet_temp = coolant_inlet_temp
        self.power_density     = power_density
        self.operating_pressure = operating_pressure
        self.uranium_pebble_fraction = uranium_pebble_fraction
        self.allowable_stress = allowable_stress

        # Outputs (filled by sizing methods)
        self.control_rods      = 0.0   # fraction of core volume
        self.fuel_kg           = 0.0   # kg of total fuel (pebbles)
        self.u235_kg           = 0.0   # kg of fissile U235
        self.coolant_flow      = 0.0   # kg/s helium mass flow rate
        self.fuel_specific_energy = 0.0  # J/kg of U235
        self.core_geometry = {
                "cylinder_radius_m": 0.0, # m Core radius
                "cylinder_height_m": 0.0, # m Core Height m
                "cylinder_volume_m3": 0.0 # m^3 Core volume
            }
        self.reflector_thickness = 0.0, # Thickness of reflector blades
        self.barrel_thickness = 0.0 # kg of barrel
        self.core_barrel_gap = 0.0 # gas gap between core (reflector) and barrel



        self._cp_interp, self._rho_interp = self._build_interpolators(helium_data_path)

    # ------------------------------------------------------------------
    def _u235_specific_energy(self):
        """Energy released per kg of U-235 fully fissioned (J/kg)."""
        e_per_fission = self.FISSION_MEV * self.J_PER_EV            # J per fission event
        e_per_mol     = e_per_fission * self.AVOGADRO               # J per mol U235
        return e_per_mol / self.M_U235                              # J per kg U235
    

    def _build_interpolators(self, path):
        """
        Load Arp table data and build 2D (T, P) interpolators for Cp and rho.

        Expected CSV format (columns):
            temperature_K, pressure_MPa, cp_J_per_gK, rho_kg_per_m3

        The grid must be regular: every combination of the unique T and P
        values must have a row (i.e. a full grid, not scattered points).
        """

        df = pd.read_csv(path)

        # Extract the unique axis values (must be sorted)
        T_vals = np.sort(df["temperature_K"].unique())     # shape (n_T,)
        P_vals = np.sort(df["pressure_MPa"].unique())      # shape (n_P,)

        # Pivot into 2D arrays shaped (n_T, n_P)
        cp_grid  = (df.pivot(index="temperature_K", columns="pressure_MPa",
                             values="cp_J_per_gK")
                      .loc[T_vals, P_vals].values)

        rho_grid = (df.pivot(index="temperature_K", columns="pressure_MPa",
                             values="rho_kg_per_m3")
                      .loc[T_vals, P_vals].values)

        cp_interp  = RegularGridInterpolator(
            (T_vals, P_vals), cp_grid,
            method="linear",
            bounds_error=True,
        )
        rho_interp = RegularGridInterpolator(
            (T_vals, P_vals), rho_grid,
            method="linear",
            bounds_error=True,
        )

        return cp_interp, rho_interp


    # ------------------------------------------------------------------
    def size_fuel(self):
        """
        Determine fissile and total fuel mass.

        Energy balance over the core lifetime:
            thermal_energy_total = heat_out x lifetime_s
            u235_fissioned = thermal_energy_total / (specific_energy x efficiency)
            u235_loaded    = u235_fissioned / burnup
            total_fuel     = u235_loaded / enrichment

        A thermal efficiency of ~1 is used here because heat_out is already
        the *thermal* power; electrical efficiency is a separate conversion step.
        """
        self.fuel_specific_energy = self._u235_specific_energy()

        # kg of U-235 that must be present at BOL to sustain the required power
        # over the full lifetime at the stated burnup fraction
        total_thermal_energy    = self.heat_out * self.lifetime_s      # J
        u235_fissioned          = total_thermal_energy / self.fuel_specific_energy
        self.u235_kg            = u235_fissioned / self.burnup        # loaded at beginning-of-life
        self.u_kg               = self.u235_kg / self.enrichment      # total heavy metal amoun
        uo2_kg                  = self.u_kg * (235*self.enrichment+238*(1-self.enrichment)+16+16)/(235*self.enrichment+238*(1-self.enrichment))

        uo2_density             = 10963 # kg/m3 https://www.sciencedirect.com/science/article/pii/S0022311599002731#aep-section-id20
        uo2_vol                 = uo2_kg / uo2_density
        
        # uMo10_kg                = self.u_kg * (235*self.enrichment+238*(1-self.enrichment)+95.95*10)/(235*self.enrichment+238*(1-self.enrichment))

        # uo2_ratio_cermet        = 0.6
        # tungsten_vol            = uo2_vol / 0.6 * 0.4
        # rho_tungsten            = 19300 # kg/m3 https://physics.nist.gov/cgi-bin/Star/compos.pl?mode=text&matno=074
        # tungsten_kg             = tungsten_vol * rho_tungsten 
        
        self.fuel_kg            = uo2_kg /self.uranium_pebble_fraction     # total fuel mass 
        # self.fuel_kg            = uo2_kg + tungsten_kg
        # self.fuel_kg            = uMo10_kg

        return {
            "u235_specific_energy_MJ_per_kg": self.fuel_specific_energy / 1e6,
            "u235_fissioned_kg":              u235_fissioned,
            "u235_loaded_kg":                 self.u235_kg,
            "total_fuel_kg":                  self.fuel_kg,
        }

    # ------------------------------------------------------------------
    def size_core(self):
        """
        Determine core volume from volumetric power density,
        then derive a simple cylindrical geometry (H ≈ D for neutron economy).
        """
        self.core_geometry["cylinder_volume_m3"] = self.heat_out / self.power_density       # m3

        # uo2_kg                  = self.u_kg * (235+16+16)/235

        # uo2_density             = 10963 # kg/m3 https://www.sciencedirect.com/science/article/pii/S0022311599002731#aep-section-id20
        # uo2_vol                 = uo2_kg / uo2_density

        # uo2_ratio_cermet        = 0.6
        # cermet_vol              = uo2_vol / 0.6 * 0.4
        # geometry_channel_frac   = 0.2113 # from sheets
        
        # self.core_geometry["cylinder_volume_m3"] = cermet_vol /(1-geometry_channel_frac)

        # Optimal H/D ≈ 1 for a bare cylinder minimises neutron leakage
        self.core_geometry["cylinder_radius_m"]  = (self.core_geometry["cylinder_volume_m3"] / (2 * np.pi)) ** (1/3)
        self.core_geometry["cylinder_height_m"]  = 2 * self.core_geometry["cylinder_radius_m"]

        return {
            "core_volume_m3": self.core_geometry["cylinder_volume_m3"],
            "cylinder_radius_m": self.core_geometry["cylinder_radius_m"],
            "cylinder_height_m": self.core_geometry["cylinder_height_m"],
        }

    # ------------------------------------------------------------------
    def size_coolant(self):
        """
        He mass-flow rate from Q = mdot Cp delta_T.
        ΔT is the rise from inlet to the required core outlet temperature.

        cp_helium : J/g/K from https://nvlpubs.nist.gov/nistpubs/Legacy/TN/nbstechnicalnote1334.pdf
        rho_helium: kg/m^3 from same source

        """
        delta_T           = self.core_temp - self.coolant_inlet_temp
        if delta_T <= 0:
            raise ValueError("core_temp must exceed coolant_inlet_temp")

        T_bulk = (self.core_temp + self.coolant_inlet_temp)/2
        P_to_MPa     = self.operating_pressure / 1e6   # convert Pa --> MPa to match table

        query     = np.array([[T_bulk, P_to_MPa]])     # shape (1, 2) as required
        cp_helium  = float(self._cp_interp(query).item())*1000
        rho_helium = float(self._rho_interp(query).item())

        self.coolant_flow = self.heat_out / (cp_helium * delta_T)

        # Volumetric flow at operating density
        vol_flow = self.coolant_flow / rho_helium

        return {
            "delta_T_K":            delta_T,
            "mass_flow_kg_per_s":   self.coolant_flow,
            "volume_flow_m3_per_s": vol_flow,
        }

    # ------------------------------------------------------------------
    def size_control_rods(self, rod_volume_fraction=0.30):
        """
        Reserve a fraction of core volume for control/shutdown rods.
        0.30 is a typical conservative design margin for HTGRs.
        """
        self.control_rods = rod_volume_fraction
        control_rod_density = 2484 # kg/m3 from sheets
        rod_volume        = self.core_geometry["cylinder_volume_m3"] * rod_volume_fraction

        return {
            "control_rod_fraction": self.control_rods,
            "rod_volume_m3":        rod_volume,
            "active_fuel_volume_m3": self.core_geometry["cylinder_volume_m3"] * (1 - rod_volume_fraction),
        }

    # ----------------------------------------------------------------------------------------------------------
    def size_reflector(self):
        """
        Graphite reflector surrounding the cylindrical core.
        """
        r_core  = self.core_geometry["cylinder_radius_m"]
        h_core  = self.core_geometry["cylinder_height_m"]

        thickness_ratio = 0.8 # from sheet

        V_core  = np.pi * r_core**2 * h_core
        thickness = r_core * thickness_ratio
        r_refl  = r_core + thickness
        h_refl  = h_core + 2 * thickness          # top and bottom too
        V_total = np.pi * r_refl**2 * h_refl

        V_reflector  = V_total - V_core
        # rho_graphite = 1700.0    # kg/m3  (nuclear grade graphite)
        # rho_reflector = 1530        # kg/m3 http://large.stanford.edu/courses/2016/ph241/tew2/docs/3310868.pdf
        rho_reflector = 3010 # kg/m3 BeO https://physics.nist.gov/cgi-bin/Star/compos.pl?mode=text&matno=116
        m_reflector  = V_reflector * rho_reflector

        self.reflector_thickness = thickness
        return {"reflector_thickness_m": thickness, "reflector_mass_kg": m_reflector}

    # -----------------------------------------------------------------------------------------------------------------------
    def size_core_barrel(self):
        """
        This is the thing that apparently holds the core together.
        Coolant first flows through the gap between the pressure vessel and the barrel, and then it goes through the gap between the barrel and the core.
        """
        r_core = self.core_geometry["cylinder_radius_m"]
        h_core = self.core_geometry["cylinder_height_m"]
        thickness_ratio = 0.04
        thickness = r_core * thickness_ratio

        gas_thickness_ratio = 0.1
        gas_gap = r_core * gas_thickness_ratio

        r_not_barrel  = r_core + self.reflector_thickness + gas_gap
        h_not_barrel  = h_core + 2 *( self.reflector_thickness + gas_gap )
        V_not_barrel = np.pi * r_not_barrel**2 * h_not_barrel

        r_barl  = r_not_barrel + thickness
        h_barl  = h_not_barrel + 2 * thickness          # top and bottom too
        V_total = np.pi * r_barl**2 * h_barl

        V_barl = V_total - V_not_barrel
        rho_barrel = 7800  # kg/m3 http://large.stanford.edu/courses/2016/ph241/tew2/docs/3310868.pdf
        m_barrel = rho_barrel * V_barl

        self.core_barrel_gap = gas_gap
        self.barrel_thickness = thickness
        return {"barrel_thickness_m": thickness, "barrel_mass_kg": m_barrel}

    # ---------------------------------------------------------------------------------------------------------
    def size_pressure_vessel(self):
        """
        Cylindrical vessel with hemispherical end caps.
        """
        # Vessel must be larger than core — add clearance for reflector and also barrel
        gas_thickness_ratio = 0.17
        gas_gap = self.core_geometry["cylinder_radius_m"] * gas_thickness_ratio

        r_inner = self.core_geometry["cylinder_radius_m"] + self.reflector_thickness + self.core_barrel_gap + self.barrel_thickness + gas_gap

        P       = self.operating_pressure
        S       = self.allowable_stress      # Pa, from ASME tables at T_vessel
        E_weld  = 0.85

        # ASME thin-wall thickness
        t = (P * r_inner) / (S * E_weld - 0.6 * P) # https://www.engineersedge.com/pressure,045vessel/thin_wall_pressure_vessels_13909.htm
        # min_t = 0.1 * self.core_geometry["cylinder_radius_m"] # using ratio from sheet
        t = max(t, 0.001)

        r_outer = r_inner + t
        h_cyl   = self.core_geometry["cylinder_height_m"] + 2*(self.reflector_thickness+self.core_barrel_gap+self.barrel_thickness+gas_gap)

        # Volume of steel: cylinder shell + 2 hemispherical caps
        V_cyl_shell = np.pi * (r_outer**2 - r_inner**2) * h_cyl
        V_caps      = (4/3) * np.pi * (r_outer**3 - r_inner**3)   # 2 hemispheres = 1 sphere
        V_vessel     = V_cyl_shell + V_caps

        # rho_steel   =  7675   # kg/m3 Using 2.25Cr:1Mo steel for now, using density from the HTR Modul 200 from http://large.stanford.edu/courses/2016/ph241/tew2/docs/3310868.pdf
        rho_steel   =  8360 # kg/m3 Inconel-617 https://www.aerospacemetals.com/wp-content/uploads/2023/08/Special-Metals-INCONEL%C2%AE-Alloy-617.pdf
        m_vessel    = V_vessel * rho_steel
        print("Outer Diameter:", 2*r_outer)
        print("Height:", h_cyl+r_outer*2)

        return {"wall_thickness_m": t, "vessel_mass_kg": m_vessel}

    # ------------------------------------------------------------------
    def size_all(self, print_true=True):
        """Run the full sizing chain and print a summary."""
        fuel    = self.size_fuel()
        core    = self.size_core()
        coolant = self.size_coolant()
        rods    = self.size_control_rods()
        reflector = self.size_reflector()
        barrel = self.size_core_barrel()
        vessel = self.size_pressure_vessel()
        total_mass = fuel["total_fuel_kg"] + reflector["reflector_mass_kg"] + barrel["barrel_mass_kg"] + vessel["vessel_mass_kg"]
        total_mass *= 1.1 # Margin for the stuff I missed rn

        if print_true:
            print("=" * 55)
            print(f"  HTGR Sizing Summary")
            print("=" * 55)
            print(f"  Thermal power         : {self.heat_out/1e6:.1f} MW(th)")
            print(f"  Core outlet temp      : {self.core_temp} K  ({self.core_temp-273.15:.0f} degrees C)")
            print()
            print(f"  -- Fuel --")
            print(f"  U-235 specific energy : {fuel['u235_specific_energy_MJ_per_kg']:.2e} MJ/kg")
            print(f"  U-235 loaded (BOL)    : {fuel['u235_loaded_kg']:.1f} kg")
            print(f"  Total fuel mass    : {fuel['total_fuel_kg']:.1f} kg")
            print()
            print(f"  -- Core geometry --")
            print(f"  Core volume           : {core['core_volume_m3']:.2f} m^3")
            print(f"  Cylinder radius       : {core['cylinder_radius_m']:.2f} m")
            print(f"  Cylinder height       : {core['cylinder_height_m']:.2f} m")
            print()
            print(f"  -- Helium coolant --")
            print(f"  Inlet --> outlet Delta_T     : {coolant['delta_T_K']:.0f} K")
            print(f"  He mass flow          : {coolant['mass_flow_kg_per_s']:.4f} kg/s")
            print(f"  He volumetric flow    : {coolant['volume_flow_m3_per_s']:.4f} m^3/s")
            print()
            print(f"  -- Control rods --")
            print(f"  Rod volume fraction   : {rods['control_rod_fraction']*100:.0f}%")
            print(f"  Active fuel volume    : {rods['active_fuel_volume_m3']:.2f} m^3")
            print()
            print(f"  -- Reflectors --")
            print(f"  Reflector thickness   : {reflector['reflector_thickness_m']:.2f} m")
            print(f"  Reflector mass    : {reflector['reflector_mass_kg']:.2f} kg")
            print()
            print(f"  -- Core barrel --")
            print(f"  Barrel thickness   : {barrel['barrel_thickness_m']:.4f} m")
            print(f"  Barrel mass    : {barrel['barrel_mass_kg']:.2f} kg")
            print()
            print(f"  -- Pressure vessel --")
            print(f"  Pressure vessel thickness   : {vessel['wall_thickness_m']:.4f} m")
            print(f"  Vessel mass    : {vessel['vessel_mass_kg']:.2f} kg")
            print()
            print(f"  -- Total Mass (0.1 Margin) --")
            print(f"  Total mass    : {total_mass:.2f} kg")
            print()
            print("=" * 55)


def uranium_frac_vs_fuel_mass():
    ufrac = np.arange(0.04,0.8,0.1)
    fuelmass = []
    for val in ufrac:
        reactor_iter = Reactor(1050, 1273, 190000, uranium_pebble_fraction=val)
        fuelmass.append(reactor_iter.size_fuel()["total_fuel_kg"])
    plt.plot(ufrac, fuelmass)
    plt.show()


def main():
    reactorquestionmark = Reactor(1050, 1273, 190000)
    reactorquestionmark.size_all()
    haleu = Reactor(1050, 1273, 190000, enrichment=0.2, power_density=5.8e6)
    haleu.size_all()


if __name__ == "__main__":
    main()