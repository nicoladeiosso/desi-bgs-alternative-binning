import numpy as np
from astropy.io import fits
from astropy.table import Table
from cosmoprimo.fiducial import DESI  # DESI's fiducial cosmology

# --- Initialize DESI cosmology ---
cosmo = DESI()  # Default DESI fiducial cosmology

# --- Parameters ---
M_cut = -20.2  # Adjust as needed
input_dir = '/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v1.1/nonKP/'
output_dir = '/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/ANY_magcut_catalog/'

# --- Process each random file ---
for i in range(0, 18):  # Assuming files are numbered from 1 to 18
    input_filename = f'BGS_ANY_NGC_{i}_clustering.ran.fits'
    output_filename = f'BGS_ANY-20.2_NGC_{i}_clustering.ran.fits'
    
    input_path = input_dir + input_filename
    output_path = output_dir + output_filename
    
    print(f"\nProcessing file {i}/18: {input_filename}")
    
    # --- Load the data ---
    with fits.open(input_path) as hdul:
        data = Table(hdul[1].data)
    
    # --- Extract redshifts ---
    z = data['Z']  # Redshift array
    
    # --- Compute luminosity distance (in Mpc/h) ---
    d_L = cosmo.luminosity_distance(z)  # Returns array in Mpc/h (DESI convention)
    
    # --- Convert to parsecs (for distance modulus) ---
    d_L_pc = d_L * 1e6 / cosmo.h  # Remove h-dependence and convert to parsecs (Deprecated)
    
    # --- Compute distance modulus ---
    distance_modulus = 5 * np.log10(d_L) + 25.  # 5*log10(d_L/10 pc)
    
    # --- Compute absolute magnitude M_r ---
    flux_r = data['flux_r_dered']
    r_mag = 22.5 - 2.5 * np.log10(flux_r)  # Apparent magnitude
    M_r = r_mag - distance_modulus  # Absolute magnitude
    
    # --- Apply the cut ---
    selected = M_r < M_cut
    
    # --- Create new catalog ---
    new_catalog = Table()
    for colname in data.colnames:
        new_catalog[colname] = data[colname][selected]
    
    # Add absolute magnitude for reference
    new_catalog['M_r'] = M_r[selected]
    
    # --- Save the new catalog ---
    new_catalog.write(output_path, overwrite=True)
    
    print(f"Original catalog: {len(data)} galaxies")
    print(f"After M_r < {M_cut} cut: {len(new_catalog)} galaxies")
    print(f"Saved new catalog to: {output_path}")

print("\nAll files processed successfully!")