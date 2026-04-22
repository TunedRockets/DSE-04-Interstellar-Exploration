# Load standard modules
import numpy as np
from matplotlib import pyplot as plt

# Load tudatpy modules
from tudatpy.interface import spice
from tudatpy import dynamics
from tudatpy.dynamics import environment_setup, propagation_setup
from tudatpy.astro import element_conversion
from tudatpy import constants
from tudatpy.util import result2array
from tudatpy.astro.time_representation import DateTime

spice.load_standard_kernels()

# Create default body settings for "Earth"
bodies_to_create = ["Earth"]

# Create default body settings for bodies_to_create, with "Earth"/"J2000" as the global frame origin and orientation
global_frame_origin = "Earth"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
   bodies_to_create, global_frame_origin, global_frame_orientation)

# Create empty body settings for the satellite
body_settings.add_empty_settings("Delfi-C3")


# Create system of bodies
bodies = environment_setup.create_system_of_bodies(body_settings)


# Define bodies that are propagated
bodies_to_propagate = ["Delfi-C3"]

# Define central bodies of propagation
central_bodies = ["Earth"]

# Define accelerations acting on Delfi-C3
acceleration_settings_delfi_c3 = dict(
   Earth=[propagation_setup.acceleration.point_mass_gravity()]
)

acceleration_settings = {"Delfi-C3": acceleration_settings_delfi_c3}

# Create acceleration models
acceleration_models = propagation_setup.create_acceleration_models(
   bodies, acceleration_settings, bodies_to_propagate, central_bodies)


# Set initial conditions for the satellite that will be
# propagated in this simulation. The initial conditions are given in
# Keplerian elements and later on converted to Cartesian elements
earth_gravitational_parameter = bodies.get("Earth").gravitational_parameter

initial_state = element_conversion.keplerian_to_cartesian_elementwise(
   gravitational_parameter = earth_gravitational_parameter,
   semi_major_axis = 6.99276221e+06, # meters
   eccentricity = 4.03294322e-03, # unitless
   inclination = 1.71065169e+00, # radians
   argument_of_periapsis = 1.31226971e+00, # radians
   longitude_of_ascending_node = 3.82958313e-01, # radians
   true_anomaly = 3.07018490e+00, # radians
)

# Set simulation start and end epochs
simulation_start_epoch = DateTime(2020, 1, 1).to_epoch()
simulation_end_epoch   = DateTime(2020, 1, 2).to_epoch()

# Create numerical integrator settings
integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
   time_step=10.0, coefficient_set=propagation_setup.integrator.rk_4
)

propagator_type = propagation_setup.propagator.cowell

# Create termination settings
termination_settings = propagation_setup.propagator.time_termination(simulation_end_epoch)

# Define list of dependent variables to save
dependent_variables_to_save = [
   propagation_setup.dependent_variable.latitude("Delfi-C3", "Earth"),
   propagation_setup.dependent_variable.longitude("Delfi-C3", "Earth"),
]

# Create propagation settings
propagator_settings = propagation_setup.propagator.translational(
   central_bodies,
   acceleration_models,
   bodies_to_propagate,
   initial_state,
   simulation_start_epoch,
   integrator_settings,
   termination_settings,
   propagator=propagator_type,
   output_variables=dependent_variables_to_save
)

# Create simulation object and propagate the dynamics
dynamics_simulator = dynamics.simulator.create_dynamics_simulator(
   bodies, propagator_settings
)

# Extract the resulting state history and convert it to an ndarray
states = dynamics_simulator.propagation_results.state_history
states_array = result2array(states)

# Extract the resulting dependent variable history and convert it to an ndarray
dependent_variables = dynamics_simulator.propagation_results.dependent_variable_history
dependent_variables_array = result2array(dependent_variables)


print(
   f"""
Single Earth-Orbiting Satellite Example.
The initial position vector of Delfi-C3 is [km]: \n
{states[simulation_start_epoch][:3] / 1E3}
The initial velocity vector of Delfi-C3 is [km/s]: \n
{states[simulation_start_epoch][3:] / 1E3} \n
After {simulation_end_epoch - simulation_start_epoch} seconds the position vector of Delfi-C3 is [km]: \n
{states[simulation_end_epoch][:3] / 1E3}
And the velocity vector of Delfi-C3 is [km/s]: \n
{states[simulation_end_epoch][3:] / 1E3}
"""
)

# Define a 3D figure using pyplot
fig = plt.figure(figsize=(6,6), dpi=125)
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f'Delfi-C3 trajectory around Earth')

# Plot the positional state history
ax.plot(states_array[:, 1], states_array[:, 2], states_array[:, 3], label=bodies_to_propagate[0], linestyle='-.')
ax.scatter(0.0, 0.0, 0.0, label="Earth", marker='o', color='blue')

# Add the legend and labels, then show the plot
ax.legend()
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
plt.show()

fig, ax = plt.subplots(tight_layout=True)

latitude = dependent_variables_array[:, 1]
longitude = dependent_variables_array[:, 2]

# Extract 3 hours data subset
relative_time_hours = (dependent_variables_array[:, 0] - simulation_start_epoch) / 3600
hours_to_extract = 3
propagation_span_hours = (simulation_end_epoch - simulation_start_epoch) / 3600

subset = int(len(relative_time_hours) / propagation_span_hours * hours_to_extract)

latitude = np.rad2deg(latitude[:subset])
longitude = np.rad2deg(longitude[:subset])

# Plot ground track
ax.set_title("3 hour ground track of Delfi-C3")
ax.scatter(longitude, latitude, s=1)
ax.scatter(longitude[0], latitude[0], label="Start", color="green", marker="o")
ax.scatter(longitude[-1], latitude[-1], label="End", color="red", marker="x")

# Configure plot
ax.set_xlabel("Longitude [deg]")
ax.set_ylabel("Latitude [deg]")
ax.set_xlim([-180, 180])
ax.set_ylim([-90, 90])
ax.set_xticks(np.arange(-180, 181, step=45))
ax.set_yticks(np.arange(-90, 91, step=45))
ax.legend()
ax.grid(True)
plt.show()