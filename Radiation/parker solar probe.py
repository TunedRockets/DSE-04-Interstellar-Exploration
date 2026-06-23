
"""Example: load PSP FIELDS (magnetic field) and SWEAP SPC (solar probe cup)
and plot magnetic field components and particle density vs radial distance.

Requires: pyspedas, pytplot, matplotlib, numpy, xarray
"""

from pyspedas.projects.psp import fields, spc, epi
import numpy as np
import matplotlib.pyplot as plt
import os

# --- User parameters ---
trange = ['2024-12-10', '2024-12-28']  # time range to load (UTC)
out_dir = os.path.dirname(__file__)

# Load magnetic field (FIELDS) using notplot=True to get dict of xarray DataArrays
print('Loading FIELDS (magnetic field)')
fields_data = fields(trange=trange, datatype='mag_RTN', level='l2', no_update=True, notplot=True)

# Load SWEAP Solar Probe Cup (SPC) data (particle moments)
print('Loading SPC (SWEAP)')
spc_data = spc(trange=trange, datatype='l2i', level='l2', no_update=True, notplot=True)
# here no_updata can be set to True if you want to load from local cache instead of downloading new data
# this makes it faster for testing, but may not have the latest data if you haven't downloaded recently
# same for line 19

# Load ephemeris (spacecraft position)
print('Loading ephemeris')
epi_data = epi(trange=trange, no_update=True, notplot=True)

# Combine all loaded data into a flattened dict for easier access
print('\nLoaded variables:')
flat_data = {}
for src_dict in [fields_data or {}, spc_data or {}, epi_data or {}]:
    for key, val in src_dict.items():
        if isinstance(val, dict) and ('x' in val or 'y' in val or 'v' in val):
            # Extract nested structure: {'x': times, 'y': data, 'v': aux} -> store as 'data' dict
            flat_data[key] = val
            y_key = 'y' if 'y' in val else 'v'
            data_shape = val[y_key].shape if hasattr(val[y_key], 'shape') else 'N/A'
            print(f'  {key}: {data_shape}')
        else:
            flat_data[key] = val
            print(f'  {key}: (other type)')

# Helper: find variable by keyword
def find_var(keywords, data_dict):
    for key in data_dict.keys():
        if any(k.lower() in key.lower() for k in keywords):
            return key
    return None

# Find best candidates
b_var = find_var(['mag_rtn', 'mag_sc', 'mag'], flat_data)
den_var = find_var(['density', 'den', 'proton_density', 'np', 'current'], flat_data)
pos_var = find_var(['pos', 'sc_pos', 'position', 'r_xyz', 'sc_r'], flat_data)

print('\nSelected variables:')
print(' B-field var:', b_var)
print(' density var:', den_var)
print(' position var:', pos_var)

# Extract and plot B-field if available
if b_var is not None:
    b_dict = flat_data[b_var]
    times_b = b_dict.get('x', b_dict.get('times', None))
    data_b = b_dict.get('y', b_dict.get('v', None))
    
    if times_b is not None and data_b is not None:
        plt.figure(figsize=(12, 4))
        if data_b.ndim == 2 and data_b.shape[1] >= 3:
            plt.plot(times_b, data_b[:, 0], label='B1 (RTN)', alpha=0.8)
            plt.plot(times_b, data_b[:, 1], label='B2 (RTN)', alpha=0.8)
            plt.plot(times_b, data_b[:, 2], label='B3 (RTN)', alpha=0.8)
        else:
            plt.plot(times_b, data_b, label=b_var, alpha=0.8)
        plt.xlabel('Time')
        plt.ylabel('B (nT)')
        plt.title('Magnetic field components (PSP FIELDS)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = os.path.join(out_dir, 'psp_B_field.png')
        plt.savefig(fname, dpi=100)
        plt.close()
        print('Saved', fname)
    else:
        print('B-field data incomplete.')
else:
    print('No magnetic field variable found.')

# Extract density and radial distance if available
radial_distance = None
if pos_var is not None:
    pos_dict = flat_data[pos_var]
    pos_data = pos_dict.get('y', pos_dict.get('v', None))
    # pos_data expected shape (N, 3)
    if pos_data is not None and pos_data.ndim == 2 and pos_data.shape[1] >= 3:
        # compute radial distance
        radial_distance = np.linalg.norm(pos_data[:, :3], axis=1)
        # Try to convert to AU if units are in km (reasonable guess)
        # Typical radial distances: ~0.05 AU for PSP close to Sun
        if np.median(radial_distance) > 1:  # If in km
            AU = 149597870.7
            radial_distance = radial_distance / AU

# Plot SPC current vs time (as proxy for particle flux)
spc_current_var = find_var(['a_current', 'b_current', 'c_current', 'd_current'], flat_data)
if spc_current_var is not None:
    spc_dict = flat_data[spc_current_var]
    times_spc = spc_dict.get('x', spc_dict.get('times', None))
    spc_data = spc_dict.get('y', spc_dict.get('v', None))
    
    if times_spc is not None and spc_data is not None:
        plt.figure(figsize=(12, 4))
        if spc_data.ndim == 2:
            # Plot first and last channels as representatives
            plt.plot(times_spc, spc_data[:, 0], label='Channel 0', alpha=0.7)
            if spc_data.shape[1] > 1:
                plt.plot(times_spc, spc_data[:, -1], label=f'Channel {spc_data.shape[1]-1}', alpha=0.7)
        else:
            plt.plot(times_spc, spc_data, alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Current (A)')
        plt.title(f'SPC Current ({spc_current_var}) vs time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        fname = os.path.join(out_dir, 'psp_spc_current.png')
        plt.savefig(fname, dpi=100)
        #plt.close()
        print('Saved', fname)

# Plot particle count rate vs time
count_var = find_var(['countrate', 'count_rate', 'rate'], flat_data)
if count_var is not None:
    count_dict = flat_data[count_var]
    times_count = count_dict.get('x', count_dict.get('times', None))
    count_data = count_dict.get('y', count_dict.get('v', None))
    
    if times_count is not None and count_data is not None:
        plt.figure(figsize=(12, 4))
        if count_data.ndim == 2 and count_data.shape[1] > 1:
            # Plot first few channels
            for i in range(min(5, count_data.shape[1])):
                plt.plot(times_count, count_data[:, i], label=f'Channel {i}', alpha=0.7)
        else:
            plt.plot(times_count, count_data, alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Count rate (counts/s)')
        plt.title(f'Particle count rate ({count_var}) vs time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        fname = os.path.join(out_dir, 'psp_particle_count.png')
        plt.savefig(fname, dpi=100)
        #plt.close()
        print('Saved', fname)

# Density vs distance plot (if position data becomes available)
if den_var is not None and radial_distance is not None:
    den_dict = flat_data[den_var]
    den_data = den_dict.get('y', den_dict.get('v', None))
    if den_data is not None:
        # Align by taking minimum length
        n = min(len(radial_distance), len(den_data) if den_data.ndim == 1 else len(den_data))
        den_vals = den_data[:n] if den_data.ndim == 1 else den_data[:n, 0]
        
        plt.figure(figsize=(7, 5))
        plt.scatter(radial_distance[:n], np.abs(den_vals), s=8, alpha=0.6)
        plt.xscale('linear')
        plt.yscale('log')
        plt.xlabel('Radial distance (AU)')
        plt.ylabel('Measured quantity (SPC)')
        plt.title('SPC measurements vs radial distance')
        plt.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        fname = os.path.join(out_dir, 'psp_quantity_vs_distance.png')
        plt.savefig(fname, dpi=100)
        #plt.close()
        plt.show()
        print('Saved', fname)
    else:
        print('Density data incomplete.')
else:
    if radial_distance is None:
        print('(Position data not loaded in ephemeris; plots vs distance skipped.)')
    if den_var is None:
        print('(Density variable not selected.)')

print('Done.')


