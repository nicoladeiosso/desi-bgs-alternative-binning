import os
import numpy as np
from astropy.io import fits
from pyrecon import IterativeFFTReconstruction, setup_logging
from cosmoprimo.fiducial import DESI
from astropy import units as u
from astropy.cosmology import z_at_value
from scipy.interpolate import interp1d

# Set up logging
setup_logging()

def redshift_to_comoving_distance(z):
    """Convert redshift to comoving distance (Mpc/h) using DESI cosmology"""
    cosmo = DESI()
    return cosmo.comoving_radial_distance(z)
    
def ra_dec_z_to_xyz(ra, dec, z):
    """
    Convert RA, DEC, redshift to comoving Cartesian coordinates (x, y, z) in Mpc/h.
    """
    cosmo = DESI()
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    Dc = cosmo.comoving_radial_distance(z)  # Comoving distance (Mpc/h)
    x = Dc * np.cos(dec_rad) * np.cos(ra_rad)
    y = Dc * np.cos(dec_rad) * np.sin(ra_rad)
    z_coord = Dc * np.sin(dec_rad)
    return x, y, z_coord

def xyz_to_ra_dec_z(x, y, z):
    """
    Convert comoving Cartesian coordinates (x, y, z) back to RA, DEC, redshift.
    Uses pre-computed interpolation for better performance.
    """
    cosmo = DESI()
    
    # Pre-compute redshift-distance relation for interpolation
    z_values = np.linspace(0, 2, 10000)
    r_values = cosmo.comoving_radial_distance(z_values)
    r_to_z = interp1d(r_values, z_values, kind='cubic', bounds_error=True)
    
    r = np.sqrt(x**2 + y**2 + z**2)
    dec = np.degrees(np.arcsin(z/r))
    ra = np.degrees(np.arctan2(y, x)) % 360  # Ensure RA is between 0-360
    
    # Convert comoving distance back to redshift using interpolation
    try:
        redshift = r_to_z(r)
    except ValueError as e:
        # Handle edge cases
        r = np.clip(r, r_values[0], r_values[-1])
        redshift = r_to_z(r)
    
    return ra, dec, redshift

def compute_recon(data_fn, rand_fns, region='GCcomb', output_dir='postrecon', 
                 nmesh=512, boxsize=4000):
    """
    Perform reconstruction and create new data and random catalogs.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading data from {data_fn}")
    with fits.open(data_fn) as hdul:
        data = hdul[1].data
    
    print(f"Read {len(data)} data objects from {data_fn}")
    
    # Read randoms (combine all files)
    print(f"Reading {len(rand_fns)} random files...")
    randoms = []
    for fn in rand_fns:
        with fits.open(fn) as hdul:
            rand = hdul[1].data
            randoms.append(rand)
    randoms = np.concatenate(randoms)
    print(f"Read {len(randoms)} random objects from {len(rand_fns)} files")

    # Convert to Cartesian coordinates
    # Convert to Cartesian coordinates
    data_x, data_y, data_z = ra_dec_z_to_xyz(data['RA'], data['DEC'], data['Z'])
    rand_x, rand_y, rand_z = ra_dec_z_to_xyz(randoms['RA'], randoms['DEC'], randoms['Z'])

    # Prepare positions and weights
    data_positions = [data_x, data_y, data_z]
    rand_positions = [rand_x, rand_y, rand_z]
    data_weights = data['WEIGHT']
    rand_weights = randoms['WEIGHT']
    
    # Initialize reconstruction
    recon = IterativeFFTReconstruction(
        f=0.68, 
        bias=1.5, 
        data_positions=data_positions, 
        data_weights=data_weights,
        randoms_positions=rand_positions, 
        randoms_weights=rand_weights, 
        cellsize=6, 
        boxsize=4000, 
        position_type='xyz',
        fft_plan='estimate'
    )
    
    # Get shifted positions
    data_positions_rec = recon.read_shifted_positions(data_positions, field='disp+rsd')
    rand_positions_rec = recon.read_shifted_positions(rand_positions, field='disp+rsd')
    
    # Convert back to RA, DEC, Z
    # Convert back to RA, DEC, Z
    data_ra_rec, data_dec_rec, data_z_rec = xyz_to_ra_dec_z(*data_positions_rec)
    rand_ra_rec, rand_dec_rec, rand_z_rec = xyz_to_ra_dec_z(*rand_positions_rec)
    
    # Create new catalogs
    new_data = data.copy()
    new_data['RA'] = data_ra_rec
    new_data['DEC'] = data_dec_rec
    new_data['Z'] = data_z_rec
    
    new_randoms = randoms.copy()
    new_randoms['RA'] = rand_ra_rec
    new_randoms['DEC'] = rand_dec_rec
    new_randoms['Z'] = rand_z_rec
    
    # Save new catalogs
    data_output_fn = os.path.join(output_dir, f"BGS_ANY-20.2_{region}_clustering.dat.fits")
    print(f"Saving reconstructed data to {data_output_fn}")
    fits.writeto(data_output_fn, new_data, overwrite=True)
    
    # Split randoms back into 18 files
    for i in range(18):
        rand_output_fn = os.path.join(output_dir, f"BGS_ANY-20.2_{region}_{i}_clustering.ran.fits")
        chunk = new_randoms[i * len(new_randoms) // 18 : (i + 1) * len(new_randoms) // 18]
        print(f"Saving reconstructed randoms to {rand_output_fn} (size: {len(chunk)})")
        fits.writeto(rand_output_fn, chunk, overwrite=True)
    
    return recon

if __name__ == "__main__":
    # Parameters
    output_dir = "postrecon"
    
    # For NGC region
    data_fn = "/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/ANY_magcut_catalog/BGS_ANY-20.2_NGC_clustering.dat.fits"
    rand_fns = [f"/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/ANY_magcut_catalog/BGS_ANY-20.2_NGC_{i}_clustering.ran.fits" for i in range(18)]
    compute_recon(data_fn, rand_fns, region='NGC', output_dir=output_dir)
    
    # For SGC region
    data_fn = "/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/ANY_magcut_catalog/BGS_ANY-20.2_SGC_clustering.dat.fits"
    rand_fns = [f"/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/ANY_magcut_catalog/BGS_ANY-20.2_SGC_{i}_clustering.ran.fits" for i in range(18)]
    compute_recon(data_fn, rand_fns, region='SGC', output_dir=output_dir)

    # For combined region (no NGC/SGC in filename)
    #data_fn = "/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/ANY_magcut_catalog/BGS_ANY-21.5_clustering.dat.fits"
    #rand_fns = [f"/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/ANY_magcut_catalog/BGS_ANY-21.5_{i}_clustering.ran.fits" for i in range(18)]
    #compute_recon(data_fn, rand_fns, region='', output_dir=output_dir)
