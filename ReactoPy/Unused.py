
def geometric_buckling(R, H, extrapolation_distance=0.0):
    """
    Geometric buckling for a finite cylinder.
    B^2 = (2.405/R_eff)^2 + (π/H_eff)^2
    
    The extrapolation distance (d ≈ 0.7104 λtr) accounts for
    the fact that neutrons leak a little beyond the physical boundary.
    For a rough check, set to 0.
    """
    R_eff = R + extrapolation_distance
    H_eff = H + 2 * extrapolation_distance
    return (2.405 / R_eff)**2 + (np.pi / H_eff)**2

def eta_reproduction(enrichment):
    """
    start of life eta
    eta = nu * N_235 * sigma_235_f / (N_235 * sigma_235_f + N_235 * sigma_235_c + N_238 * sigma_238_c)
    
    or

    eta = nu * enrichment * sigma_235_f / ( enrichment * sigma_235_a + (1-enrichment)* sigma_238_c)

    Uses tabulated nuclear data for U-235/U-238 at thermal energies.
    All cross sections in barns, at 293K (temperature correction needed
    for hot operation — see below).
    """
    # Nuclear data (2200 m/s, 0.025 eV) from ENDF/B-VIII.0
    # Source: https://www.nndc.bnl.gov/sigma/ << this isn't real, mr claude, but ENDF and nuclear-power.com gives similar results
    sigma_f_U235   = 585.1   # barns — fission
    sigma_a_U235   = 680.9   # barns — total absorption (fission + capture)
    sigma_a_U238   = 2.717   # barns — mostly capture
    # sigma_a_W      = 18.4    # barns — tungsten (cermet matrix)
    # sigma_a_O      = 0.00019 # barns — oxygen (negligible)
    
    nu             = 2.437   # neutrons per fission (U-235, thermal)
    
    # η: neutrons produced per absorption in fuel
    # For a mixture of U-235 and U-238:
    n_U235 = enrichment
    n_U238 = 1 - enrichment
    
    sigma_a_fuel = n_U235 * sigma_a_U235 + n_U238 * sigma_a_U238
    eta = nu * (n_U235 * sigma_f_U235) / sigma_a_fuel
    return eta

def f_thermal_ut(enrichment, v_core, v_poison):
    """
    f = n. of neutrons absorbed in fuel / n. of neutrons absorbed in core as whole
    for a homogeneous core
    f = S_a_U / (S_a_U + S_a_M + S_a_P + S_a_CR + S_a_B + S_a_BR + S_a_O)
    simplified for the case
    Fuel, and control rods
    control rods out of B4C
    but also add volumetric weighting (so basically like the heterogeneous case but ignoring self-shielding effects)
    f = S_a_U * V_core / (S_a_U * V_core + S_a_CR * V_controlrods)
    """
    sigma_a_U235 = 680.9
    sigma_a_U238 = 2.717

    sigma_a_fuel = enrichment * sigma_a_U235 + (1-enrichment) * sigma_a_U238
    S_a_U = sigma_a_fuel * 1e-24 * enrichment_atomic_number_density(rho_compound=RHO_UO2/1000, M_element=237.9, M_compound=269.9, enrichment=enrichment, M_iso=235)
    
    sigma_a_CR = 200 # B-10 capture
    S_a_CR = sigma_a_CR * 1e-24 * atomic_number_density(rho=RHO_B4C/1000, M=10.81*4+12.011) * 4
    f = S_a_U * v_core / (S_a_U * v_core + S_a_CR * v_poison)

    return f

def p_resonance_escape(diameter, enrichment, fuel_volume, moderator_volume):
    """
    https://www.nuclear-power.com/nuclear-power/reactor-physics/nuclear-fission-chain-reaction/resonance-escape-probability/
    https://www.nuclear-power.com/glossary/neutron-moderatoraverage-logarithmic-energy-decrement/
    p = exp(-N_F * V_F / (xi * S_s_M * V_M) * I_eff)
    where
    I_eff = 4.45 + 26.6*sqrt(4/(rho * D)) for UO2, rho and D(iameter) in g/cm3 and cm, valid for D > 0.2cm
    and taking moderator as the oxygen and the berrylium
    """

    rho = RHO_UO2 
    D = diameter # cm
    # D = 10000000000
    I_eff = 4.45 + 26.6*np.sqrt(4/(rho/1000 * D))

    N_f = enrichment_atomic_number_density(rho_compound=RHO_UO2/1000, M_element=237.9, M_compound=269.9, enrichment=enrichment, M_iso=235)
    V_f = fuel_volume
    # specifically for BeO and UO2
    V_m = moderator_volume # not really a moderator, but we will consider scattering from beryllium oxide, so just reflector volume
    s_oxy = 4 # barns, scattering
    s_beo = 9.9 # barns, scattering https://wwwndc.jaea.go.jp/cgi-bin/list451.cgi?lib=J40SC&iso=BeO
    S_s_M_O = s_oxy*1e-24 * enrichment_atomic_number_density(RHO_UO2/1000, 16*2, 267.7, 1, 16) # Scattering of oxygen in UO2
    S_s_M_BeO = s_beo*1e-24 * atomic_number_density(RHO_BEO/1000, 9.0122+16) # scattering of Beryllium Oxide
    S_s_M_V_m_eff = S_s_M_BeO * V_m +  S_s_M_O * V_f
    xi = 2/(267.7+2/3) # UO2
    print("N_f:", N_f )
    print("V_f:", V_f )
    print("xi:", xi)
    print("S_s_M:", S_s_M_V_m_eff)
    print("I_eff:", I_eff)
    # I_eff = 1.45e-22 
    print(-N_f * V_f / (xi * S_s_M_V_m_eff) * I_eff)
    p = np.exp(-N_f * V_f / (xi * S_s_M_V_m_eff) * I_eff)

    return p


def fast_fission():
    fast_thermal_neutrons = 2.42+0.0162 # (fast) neutrons produced by thermal fission
    all_fast_neutrons = 2.42+0.0162+2.63+0.0165 # neutrons produced by all fission
    epsilon = all_fast_neutrons/ fast_thermal_neutrons 

    return epsilon

def k_inf_thermal(enrichment, v_core, v_poison, diameter, fuel_volume, moderator_volume, print_true=False):
    """
    Rough k∞ estimate for a thermal reactor.
    factors:
    eta = reproduction
    f = thermal utilization
    p = resonance escape
    epsilon = fast fission
    """ 

    eta = eta_reproduction(enrichment)
    
    f = f_thermal_ut(enrichment, v_core, v_poison)
    
    p = p_resonance_escape(diameter, enrichment, fuel_volume, moderator_volume) 
    # p = 0.8 # override

    epsilon = fast_fission()
    epsilon = 1.03 # override
    if print_true:
        print("eta:", eta)
        print("f:", f)
        print("p:", p)
        print("epsilon:", epsilon)

    
    return eta * f * p * epsilon

def M2_cm2():
    """
    https://www.nuclear-power.com/nuclear-power/reactor-physics/neutron-diffusion-theory/migration-length-migration-area/
    https://canteach.candu.org/Content%20Library/20050906.pdf
    """
    M2 = 1.54**2 
    # M2 = 29**2 + 10**2 #beryllium
    return M2


def check_criticality(R, H, M2_cm2, enrichment, v_core, v_poison, fuel_volume, moderator_volume, print_true=False):
    """
    R, H in metres. M2 in cm².
    Returns keff and whether critical.
    """
    # convert to cm
    R = R*100
    H = H*100

    B2 = geometric_buckling(R, H, extrapolation_distance())
    k_inf = k_inf_thermal(enrichment, v_core, v_poison, 2*R, fuel_volume, moderator_volume, print_true)
    keff = k_inf / (1 + M2_cm2 * B2)
    if print_true:
        print("M2_cm2:", M2_cm2)
        print("B2:", B2)
        print("k_inf:", k_inf)
        print("k_eff:", keff)
    return keff, keff >= 1.0

def atomic_number_density(rho, M):
    """
    N = rho * N_A / M

    for mixed compounds

    N_mix = rho_mix * N_A / M_mix
    """

    return rho * AVOGADRO / M

def enrichment_atomic_number_density(rho_compound, M_element, M_compound, enrichment, M_iso):
    """
    for enrichment

    N_iso = rho_iso * N_A / M_iso = enrichment * rho_element * N_A / M_iso
    
    and for compounds eg UO2,

    rho_element = rho_UO2 * M_U/M_UO2 
    """

    rho_element = rho_compound* M_element / M_compound
    N_iso = enrichment * rho_element * AVOGADRO / M_iso

    return N_iso


def extrapolation_distance(compound="UO2"):
    """
    The extrapolation distance (d ≈ 0.7104 λtr) uses the free-mean path which is related to the diffusion coefficient
    https://www.nuclear-power.com/nuclear-power/reactor-physics/neutron-diffusion-theory/vacuum-boundary-condition-extrapolated-length/
    https://www.nuclear-power.com/nuclear-power/reactor-physics/neutron-diffusion-theory/diffusion-coefficient/
    https://www.nuclear-power.com/nuclear-power/reactor-physics/nuclear-engineering-fundamentals/neutron-nuclear-reactions/microscopic-cross-section/
    https://www.nuclear-power.com/nuclear-power/reactor-physics/nuclear-engineering-fundamentals/neutron-nuclear-reactions/macroscopic-cross-section/
    https://www.nuclear-power.com/nuclear-power/reactor-physics/nuclear-engineering-fundamentals/neutron-nuclear-reactions/atomic-number-density/
    """

    if compound=="UO2":
        s = 10 # barn (1e-24 cm2) Microscopic scattering cross section
        M = 235
        S_s = s*1e-24 * enrichment_atomic_number_density(rho_compound=10.5, M_element=237.9, M_compound=269.9, enrichment=0.95, M_iso=M) # 1/cm Macroscopic cross section
        mubar = 2/(3*M)
        diff = 1/(3*S_s*(1-mubar))
        lambda_tr = 3 * diff
    return lambda_tr

def moderator_ratio(M, sigma_s, sigma_a):
    """
    xi * sigma_s / sigma_a - should be >> 1 for good moderator.
    """
    xi = 2/(M+2/3)
    return xi * sigma_s/ sigma_a


if __name__ == "__main__":
    print(moderator_ratio(95.95,5.71,2.48))