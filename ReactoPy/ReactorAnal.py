import openmc
import openmc.data
import numpy as np
import matplotlib.pyplot as plt
import ReactoPy.ReactorSize

# Then point OpenMC at it
openmc.config['cross_sections'] = 'data/ENDF_folder/endfb-viii.1-hdf5/cross_sections.xml'

# --- Settings ---
settings = openmc.Settings()
settings.run_mode    = 'eigenvalue'
settings.batches     = 150
settings.inactive    = 30       # discard first 30 batches for source convergence
settings.particles   = 10000  # per batch — increase to 50k for tighter uncertainty
settings.photon_transport = True
settings.temperature = {
    "method" : "interpolation"
}


def make_control_drum(x0, y0, r_drum, rotation_angle_deg,
                      reflector, poison,
                      bot_plane, top_plane,
                      absorber_arc_deg=120.0):
    """
    Single control drum at position (x0, y0) in the reflector.
    
    x0, y0          : drum centre coordinates (cm)
    r_drum          : drum radius (cm)
    rotation_angle  : angle of drum rotation (deg) — 0 = absorber faces core,
                      180 = absorber faces away from core (full reactivity)
    beo, b4c        : openmc.Material objects
    z_bot, z_top    : axial extent (cm)
    absorber_arc_deg: arc of B4C absorber (120° typical)
    """
    # Drum bounding cylinder (offset from origin)
    drum_cyl = openmc.ZCylinder(x0=x0, y0=y0, r=r_drum)

    # Axial bounds (shared with reflector)
    axial = +bot_plane & -top_plane

    # B4C absorber sector — defined by two half-planes through drum centre
    # The normal to the dividing plane is perpendicular to the arc bisector
    half_arc  = np.radians(absorber_arc_deg / 2)
    rot       = np.radians(rotation_angle_deg)

    # Bisector of absorber arc points in direction (cos(rot), sin(rot))
    # The two bounding planes are rotated ±half_arc from that direction
    # A plane through (x0,y0) with normal (nx, ny) divides space as:
    #   nx*(x - x0) + ny*(y - y0) = 0
    # Use openmc.Plane for an arbitrarily oriented plane

    def sector_plane(angle):
        """Half-plane through drum centre at given angle."""
        nx =  np.sin(angle)   # normal points 90° from the plane direction
        ny = -np.cos(angle)
        # Plane equation: nx*x + ny*y = nx*x0 + ny*y0
        return openmc.Plane(a=nx, b=ny, c=0,
                            d=nx * x0 + ny * y0)

    plane1 = sector_plane(rot - half_arc)
    plane2 = sector_plane(rot + half_arc)

    # B4C region: inside drum AND between the two planes (the absorber arc)
    # The absorber faces in the direction of rot, so we want the half-space
    # on the rot side of both planes
    absorber_region = (-drum_cyl & axial
                       & +plane1    # adjust sign if arc comes out wrong
                       & -plane2)

    # BeO region: rest of drum
    reflector_region = -drum_cyl & axial & ~(+plane1 & -plane2)

    absorber_cell = openmc.Cell(fill=poison, region=absorber_region,
                                name=f'drum_b4c_{x0:.0f}_{y0:.0f}')
    reflector_cell      = openmc.Cell(fill=reflector, region=reflector_region,
                                name=f'drum_beo_{x0:.0f}_{y0:.0f}')

    return drum_cyl, [absorber_cell, reflector_cell]

def make_mono_drum(x0, y0, r_drum,
                      drum_mat,
                      bot_plane, top_plane):

    # Drum bounding cylinder (offset from origin)
    drum_cyl = openmc.ZCylinder(x0=x0, y0=y0, r=r_drum)

    # Axial bounds (shared with reflector)
    axial = +bot_plane & -top_plane

    drum_region = (-drum_cyl & axial)

    drum_cell = openmc.Cell(fill=drum_mat, region=drum_region, name=f"drum_{x0:.0f}_{y0:.0f}")
    return drum_cyl, drum_cell

def make_all_drums(n_drums, r_drum_centre, r_drum,
                   rotation_angle_deg,
                   reflector, poison, bot_plane, top_plane, mono=False):
    """
    Place n_drums evenly spaced around a circle of radius r_drum_centre.
    Returns list of drum cylinders (for carving out of reflector)
    and list of all drum cells.
    """
    drum_cyls  = []
    drum_cells = []

    for i in range(n_drums):
        position_angle = 2 * np.pi * i / n_drums   # evenly spaced
        x0 = r_drum_centre * np.cos(position_angle)
        y0 = r_drum_centre * np.sin(position_angle)

        if not mono:
            local_rotation_deg = np.degrees(position_angle) + rotation_angle_deg

            cyl, cells = make_control_drum(
                x0, y0, r_drum, local_rotation_deg,
                reflector, poison, bot_plane, top_plane
            )
            drum_cells.extend(cells)

        else:
            if i < 0:
                cyl, cell = make_mono_drum(
                    x0, y0, r_drum, poison, bot_plane, top_plane
                )
            else:
                cyl,cell = make_mono_drum(
                    x0, y0, r_drum, reflector, bot_plane, top_plane
                )
            drum_cells.append(cell)

        drum_cyls.append(cyl)

    return drum_cyls, drum_cells

# 1273
high_temp = 1273
# 862
low_temp = 862

reactor = ReactoPy.ReactorSize.Reactor(low_temp, high_temp, 23800/0.3973, fuel_type="U10MO", operating_pressure=4.025e6, uranium_pebble_fraction=0.95, power_density="int")
reactor.size_all()

rho_smear = reactor.fuel_kg*1000 / (reactor.core_geometry["cylinder_volume_m3"]*1e6)
# Manual override of fuel mass
man_fuel = 55
rho_smear = man_fuel*1000 / (reactor.core_geometry["cylinder_volume_m3"]*1e6)

# --- Materials ---
# # UO2 fuel
# fuel = openmc.Material(name='UO2_fuel')
# fuel.add_nuclide('U235', 0.95, 'wo')
# fuel.add_nuclide('U238', 0.05, 'wo')
# fuel.add_element('O',    0.118, 'wo')   # UO2: 238/(238+32) ≈ 0.882 U by mass
# fuel.set_density('g/cm3', rho_smear)
# fuel.temperature = 1273   # K — Doppler broadening at operating temp

# U-10M fuel
fuel = openmc.Material(name='U-10M_fuel')
fuel.add_nuclide('U235', 0.95 * 0.9, 'wo')
fuel.add_nuclide('U238', 0.05* 0.9, 'wo')
fuel.add_element('Mo',    0.10, 'wo')
fuel.set_density('g/cm3', rho_smear)
fuel.temperature = high_temp   # K — Doppler broadening at operating temp

# BeO reflector
beo = openmc.Material(name='BeO_reflector')
beo.add_element('Be', 0.36, 'wo')   # 9/(9+16)
beo.add_element('O',  0.64, 'wo')
beo.set_density('g/cm3', 3.01)
beo.temperature = low_temp  # K — cooler than core

# B4C poison
b4c = openmc.Material(name="B4C_absorber")
b4c.add_element("B", 0.782, "wo")
b4c.add_element("C", 0.218, "wo")
b4c.set_density("g/cm3", 2.52)
b4c.temperature = high_temp 

# Be reflector
be = openmc.Material(name='Be_reflector')
be.add_element('Be', 1, 'wo')
be.set_density('g/cm3', 1.848)
be.temperature = low_temp  # K — cooler than core

# Tungsten
wsten = openmc.Material(name="W_shield")
wsten.add_element("W", 1, 'wo')
wsten.set_density("g/cm3", 19.254)
wsten.temperature = 300

reflector = beo

poison = b4c

shield = wsten

shield_reflector = be

materials = openmc.Materials([fuel, reflector, poison, shield, shield_reflector])


# --- Geometry (pull dimensions from your Reactor instance) ---
R_core  = reactor.core_geometry["cylinder_radius_m"] *100   # cm
H_core  = reactor.core_geometry["cylinder_height_m"]  * 100  # cm
t_refl = reactor.reflector_geometry["reflector_thickness_m"] * 100 # cm
R_refl  = R_core + t_refl # cm — reflector outer radius

r_drum        = t_refl * 0.45    # drum fits inside reflector with small clearance
r_drum_centre = R_core + r_drum  # drum centre sits just inside reflector midpoint
n_drums       = 8                # typical for compact space reactors


core_cyl    = openmc.ZCylinder(r=R_core)
refl_cyl    = openmc.ZCylinder(r=R_refl, boundary_type='vacuum')
core_top    = openmc.ZPlane(z0= H_core/2)
core_bot    = openmc.ZPlane(z0=-H_core/2)
refl_top    = openmc.ZPlane(z0= H_core/2 + t_refl, boundary_type='vacuum')
refl_bot    = openmc.ZPlane(z0=-H_core/2 - t_refl)



drum_cyls, drum_cells = make_all_drums(
    n_drums, r_drum_centre, r_drum,
    rotation_angle_deg=0,   # start fully withdrawn (BeO facing core)
    reflector=beo, poison=b4c,
    bot_plane=refl_bot, top_plane=refl_top, mono=False
)



core_region = -core_cyl & +core_bot & -core_top

# cut out control rods

# refl_region = (+core_cyl & -refl_cyl & +core_bot & -core_top)
refl_region = (-refl_cyl & +refl_bot & -refl_top) & ~core_region

for d_cyl in drum_cyls:
    refl_region = refl_region & ~(-d_cyl)   # exclude each drum footprint
    core_region = core_region & ~(-d_cyl)




# Shield geometry parameters
t_shield_cm = 0.0334*100
t_shield_cm = 30
separation_cm = 1000
z_shield_top = -(H_core/2 + t_refl)          # bottom of reflector
z_shield_bot = z_shield_top - t_shield_cm     # bottom of shield material
z_spacecraft = z_shield_bot - separation_cm   # spacecraft bus location

shield_top_plane = openmc.ZPlane(z0=z_shield_top)
shield_mid_plane1 = openmc.ZPlane(z0=z_shield_top-t_shield_cm/3)
shield_mid_plane2 = openmc.ZPlane(z0=z_shield_bot+t_shield_cm/4)
shield_bot_plane = openmc.ZPlane(z0=z_shield_bot)

# Shield Cell
shield_cell_be = openmc.Cell(
    fill=shield_reflector,
    region=-shield_top_plane & +shield_mid_plane1 & -refl_cyl,
    name="shield_be_part"
)
shield_cell_b4c = openmc.Cell(
    fill=poison,
    region=-shield_mid_plane1 & +shield_mid_plane2 & -refl_cyl,
    name="shield_b4c_part"
)
shield_cell_w = openmc.Cell(
    fill=shield,
    region=-shield_mid_plane2 & +shield_bot_plane & -refl_cyl,
    name="shield_tungsten_part"
)

far_vacuum  = openmc.ZPlane(z0=z_spacecraft - 50, boundary_type="vacuum")

# Void Cell
void_cell = openmc.Cell(
    fill=None,
    region=-refl_cyl & +far_vacuum & -shield_bot_plane,
    name="boom_region"
) 

# Tally planes: axial slices from bottom of reflector to spacecraft

# Mesh 1: thin pencil along z-axis, many axial bins
axial_mesh = openmc.CylindricalMesh([0, 5.0],
                                    np.linspace(z_spacecraft,   # just above the shield
                                                z_shield_top -5,   # spacecraft plane
                                                60))
axial_mesh.r_grid   = [0, 5.0]                              # cm — thin central pencil
axial_mesh.phi_grid = [0, 2*np.pi]

# Mesh 2: radial disc at spacecraft plane, thin axial slice
R_spacecraft_bus = 2**0.5 * 100
radial_disc_mesh = openmc.CylindricalMesh(np.linspace(0, R_spacecraft_bus, 40),
                                            [z_spacecraft - 1, z_spacecraft +1]) # thin slice at spacecraft
radial_disc_mesh.phi_grid = [0, 2*np.pi]



# All cells

core_cell = openmc.Cell(fill=fuel,      region=core_region, name='core')
refl_cell = openmc.Cell(fill=reflector, region=refl_region, name='reflector')


geometry  = openmc.Geometry([core_cell, refl_cell, shield_cell_be, shield_cell_b4c, shield_cell_w, void_cell]+drum_cells)
# geometry  = openmc.Geometry(drum_cells)
# geometry  = openmc.Geometry([core_cell, refl_cell])
geometry  = openmc.Geometry([core_cell])

# Initial fission source — uniform in core cylinder
bounds = [-R_core, -R_core, -H_core/2, R_core, R_core, H_core/2]
settings.source = openmc.IndependentSource(
    space=openmc.stats.Box(bounds[:3], bounds[3:], only_fissionable=True)
)


# --- Tally for neutron flux
# ASTM E722 silicon damage energy function — piecewise linear response
# Source: ASTM E722-14, Table 1
# Units: relative displacement cross section (barn equivalent) vs energy (MeV)
# Simplified tabulation — use full table from ASTM for production work
astm_e722_energies = np.array([
    1e-10, 1.5e-7, 1e-6, 1e-5, 1e-4, 1e-3,
    1e-2,  1e-1,   1.0,  2.0,   5.0,  10.0,  20.0
]) * 1e6  # convert MeV to eV for OpenMC energy filters

astm_e722_response = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0533, 1.0, 1.778, 2.111, 2.222, 2.222
])
# Response normalised to 1.0 at 1 MeV — so result is in 1-MeV equivalents


# --- Neutron 1-MeV equiv, axial ---
axial_filter       = openmc.MeshFilter(axial_mesh)
energy_filter      = openmc.EnergyFilter(astm_e722_energies)

n_axial_tally = openmc.Tally(name='neutron_axial')
n_axial_tally.filters = [axial_filter, energy_filter]
n_axial_tally.scores  = ['flux']

# --- Gamma dose, axial ---
g_axial_tally = openmc.Tally(name='gamma_axial')
g_axial_tally.filters = [axial_filter]
g_axial_tally.scores  = ['heating']

# --- Neutron, radial disc at spacecraft ---
disc_filter = openmc.MeshFilter(radial_disc_mesh)

n_disc_tally = openmc.Tally(name='neutron_disc')
n_disc_tally.filters = [disc_filter, energy_filter]
n_disc_tally.scores  = ['flux']

# --- Gamma, radial disc ---
g_disc_tally = openmc.Tally(name='gamma_disc')
g_disc_tally.filters = [disc_filter]
g_disc_tally.scores  = ['heating']

tallies = openmc.Tallies([
    n_axial_tally,
    g_axial_tally,
    n_disc_tally,
    g_disc_tally,
])




# --- Model Construction ---

model = openmc.Model(geometry, materials, settings, tallies=tallies)
model.export_to_xml()

# Quick visual check — does each drum show a black B4C wedge
# oriented correctly relative to the core?
model.plot(
    basis='xy',
    origin=(0, 0, 0),
    width=(2*R_refl*1.1, 2*R_refl*1.1),
    pixels=(1000, 1000),
    color_by='cell',
)


plt.show()

# --- Run ---
sp_path = model.run()

with openmc.StatePoint(sp_path) as sp:
    keff = sp.keff
    print(f"keff = {keff.nominal_value:.5f} ± {keff.std_dev:.5f}")

    # Axial tallies
    nt_ax = sp.get_tally(name='neutron_axial')
    gt_ax = sp.get_tally(name='gamma_axial')
    nt_disc = sp.get_tally(name='neutron_disc')
    gt_disc = sp.get_tally(name='gamma_disc')

    n_ax_flux = nt_ax.get_values(scores=['flux']).reshape(
        len(axial_mesh.z_grid)-1, len(astm_e722_energies)-1
    )
    g_ax_heat = gt_ax.get_values(scores=['heating']).flatten()

    n_disc_flux = nt_disc.get_values(scores=['flux']).reshape(
        len(radial_disc_mesh.r_grid)-1, len(astm_e722_energies)-1
    )
    g_disc_heat = gt_disc.get_values(scores=['heating']).flatten()


# Fission source strength
E_per_fission  = 202.0e6 * 1.602e-19   # J
fission_rate   = reactor.heat_out / E_per_fission   # fissions/s
nu             = 2.60    # neutrons per fast fission
source_strength = fission_rate * nu   # neutrons/s

# Interpolate ASTM response at energy bin midpoints
# Apply ASTM E722 response for 1-MeV equiv
e_mids = np.sqrt(astm_e722_energies[:-1] * astm_e722_energies[1:])
response_at_mids = np.interp(e_mids, astm_e722_energies, astm_e722_response)

# 1-MeV equivalent fluence per source neutron at each radial position
# To convert to absolute fluence: multiply by total source strength (n/s)
# source_strength = reactor.heat_out / energy_per_fission  (fissions/s * neutrons/fission)
n_ax_1MeV   = (n_ax_flux   * response_at_mids).sum(axis=1) * source_strength
n_disc_1MeV = (n_disc_flux * response_at_mids).sum(axis=1) * source_strength

# Gamma dose rate
# 1 rad = 100 erg/g = 6.242e7 eV/g
# rho_Si = 2.33 g/cm³
# Volume of each mesh cell needed

eV_per_rad_per_g = 6.242e7      # eV/g
rho_Si           = 2.33         # g/cm³ (silicon)

# heating score is in eV per source particle per cm³ of mesh cell
# dose_rate = heating * source_strength / (eV_per_rad_per_g * rho_Si)
eV_per_rad = eV_per_rad_per_g * rho_Si   # eV per rad per cm³ of silicon (eV/g * g/cm³)
g_ax_dose   = g_ax_heat   * source_strength / eV_per_rad   # rad/s, axial
g_disc_dose = g_disc_heat * source_strength / eV_per_rad   # rad/s, disc

# Convert to krad over mission:
mission_s    = 3.1536e7   # 1 year in seconds
g_ax_dose_krad    = g_ax_dose * mission_s / 1000
g_disc_dose_krad    = g_disc_dose * mission_s / 1000

n_ax_1MeV = n_ax_1MeV * mission_s
n_disc_1MeV = n_disc_1MeV * mission_s

# z-coordinates of axial bin midpoints
z_mids = 0.5 * (axial_mesh.z_grid[:-1] + axial_mesh.z_grid[1:])
r_mids = 0.5 * (radial_disc_mesh.r_grid[:-1] + radial_disc_mesh.r_grid[1:])

# ---- Plot 1: axial decay from reactor bottom to spacecraft ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.semilogy(-z_mids, n_ax_1MeV)   # flip sign so distance increases rightward
ax1.axvline(-z_shield_top, color='gray', linestyle='--', label='Reactor bottom')
ax1.axvline(-z_shield_bot, color='k',    linestyle='--', label='Shield bottom')
ax1.axvline(-z_spacecraft, color='r',    linestyle='--', label='Spacecraft plane')
ax1.set_xlabel('Axial distance below reactor bottom (cm)')
ax1.set_ylabel('1-MeV equiv. neutron (n/cm²)')
ax1.set_title('Neutron fluence — axial centreline')
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.semilogy(-z_mids, g_ax_dose)
ax2.axvline(-z_shield_top, color='gray', linestyle='--', label='Reactor bottom')
ax2.axvline(-z_shield_bot, color='k',    linestyle='--', label='Shield bottom')
ax2.axvline(-z_spacecraft, color='r',    linestyle='--', label='Spacecraft plane')
ax2.set_xlabel('Axial distance below reactor bottom (cm)')
ax2.set_ylabel('Gamma dose (krad(Si))')
ax2.set_title('Gamma dose — axial centreline')
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('axial_dose.png', dpi=150)
plt.show()

# ---- Plot 2: radial profile at spacecraft plane ----
# This tells you how wide the shadow cone needs to be
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.semilogy(r_mids, n_disc_1MeV)
ax1.set_xlabel('Radius at spacecraft plane (cm)')
ax1.set_ylabel('1-MeV equiv. neutron flux (n/cm²/s)')
ax1.set_title(f'Neutron fluence — radial profile at z={z_spacecraft:.0f} cm')
ax1.grid(True, alpha=0.3)

ax2.semilogy(r_mids, g_disc_dose)
ax2.set_xlabel('Radius at spacecraft plane (cm)')
ax2.set_ylabel('Gamma dose rate (rad(Si)/s)')
ax2.set_title(f'Gamma dose rate — radial profile at z={z_spacecraft:.0f} cm')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('radial_dose_spacecraft.png', dpi=150)
plt.show()


# angles = np.linspace(0, 180, 10)   # 0 = fully inserted, 180 = fully withdrawn
# keffs  = []

# for angle in angles:
#     # rebuild geometry and model with new drum cells

#     drum_cyls, drum_cells = make_all_drums(
#         n_drums, r_drum_centre, r_drum,
#         rotation_angle_deg=angle,   # start fully withdrawn (BeO facing core)
#         reflector=beo, poison=b4c,
#         bot_plane=refl_bot, top_plane=refl_top
#     )
#     # All cells

#     geometry  = openmc.Geometry([core_cell, refl_cell]+drum_cells)


#     model = openmc.Model(geometry, materials, settings, tallies=tallies)
#     model.export_to_xml()

#     # # Quick visual check — does each drum show a black B4C wedge
#     # # oriented correctly relative to the core?
#     # model.plot(
#     #     basis='xy',
#     #     origin=(0, 0, 0),
#     #     width=(2*R_refl*1.1, 2*R_refl*1.1),
#     #     pixels=(1000, 1000),
#     #     color_by='material',
#     # )


#     # plt.show()

#     # # --- Run ---
#     sp_path = model.run()

#     with openmc.StatePoint(sp_path) as sp:
#         keff = sp.keff
#         print(f"keff = {keff.nominal_value:.5f} ± {keff.std_dev:.5f}")



#     keffs.append(keff.nominal_value)

# print(angles)
# print(keffs)
# plt.plot(angles, keffs)
# plt.show()


# print(max(keffs))
# print(min(keffs))

# # 300-600 : 0.968 - 1.038
# # 857-1273 : 0.969 - 1.0381
# # 1600-2000: 0.97 - 1.038