#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute power spectrum multipoles (ℓ=0,2,4) and window functions for BGS_ANY data using PyPower.
"""

import os
import numpy as np
import tempfile
from astropy.io import fits
from cosmoprimo.fiducial import DESI
from pypower import CatalogFFTPower, CatalogSmoothWindow, PowerSpectrumSmoothWindow,\
                    BaseMatrix, PowerSpectrumSmoothWindowMatrix, PowerSpectrumOddWideAngleMatrix,\
                    mpi, setup_logging
import logging

# Initialize logging
logger = logging.getLogger(__name__)

# Initialize DESI cosmology
cosmo = DESI()
h = 100*cosmo.h  # Hubble parameter (H0 = 100h km/s/Mpc)

def redshift_to_comoving_distance(z):
    """Convert redshift to comoving distance (Mpc/h) using DESI cosmology"""
    cosmo = DESI()
    return cosmo.comoving_radial_distance(z)

def ra_dec_z_to_xyz(ra, dec, z):
    """
    Convert RA, DEC, redshift to comoving Cartesian coordinates (x, y, z) in Mpc/h.
    
    Parameters
    ----------
    ra, dec : array_like
        Right Ascension and Declination (degrees).
    z : array_like
        Redshift.
    
    Returns
    -------
    x, y, z : array_like
        Comoving Cartesian coordinates (Mpc/h).
    """
    cosmo = DESI()
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    Dc = cosmo.comoving_radial_distance(z)  # Comoving distance (Mpc/h)
    x = Dc * np.cos(dec_rad) * np.cos(ra_rad)
    y = Dc * np.cos(dec_rad) * np.sin(ra_rad)
    z_coord = Dc * np.sin(dec_rad)  # Renamed to avoid conflict with redshift z
    return x, y, z_coord

def get_alpha(data, randoms):
    alpha = len(data) / len(randoms)
    logger.info(f'Estimated alpha is {alpha}')
    return alpha

def select_region(catalog, region):
    """Select either NGC or SGC region based on RA cuts"""
    mask = np.ones(len(catalog), dtype=bool)
    if region == 'NGC':
        mask &= (catalog['RA'] > 88) & (catalog['RA'] < 303)
    elif region == 'SGC':
        mask &= (catalog['RA'] < 88) | (catalog['RA'] > 303)
    return catalog[mask]

def concatenate_catalogs(list_data, list_randoms, region='GCcomb'):
    """
    Concatenate multiple data and random catalogs, applying region selection and weight normalization.
    
    Parameters
    ----------
    list_data : list of arrays
        List of data catalogs
    list_randoms : list of arrays
        List of random catalogs
    region : str
        Region identifier ('NGC', 'SGC', or 'GCcomb')
    
    Returns
    -------
    data, randoms : arrays
        Concatenated catalogs with proper weight normalization
    """
    # Apply region selection if needed
    if region in ['NGC', 'SGC']:
        list_data = [select_region(catalog, region) for catalog in list_data]
        list_randoms = [select_region(catalog, region) for catalog in list_randoms]
    
    # Calculate weight sums and normalization factors
    wsums_data = [np.sum(data['WEIGHT']) for data in list_data]
    wsums_randoms = [np.sum(randoms['WEIGHT']) for randoms in list_randoms]
    alpha = sum(wsums_data) / sum(wsums_randoms)
    alphas = [wsum_data / wsum_randoms / alpha for wsum_data, wsum_randoms in zip(wsums_data, wsums_randoms)]
    
    logger.info(f'Renormalizing randoms weights by {alphas} before concatenation.')
    
    # Apply normalization to randoms
    for randoms, alpha in zip(list_randoms, alphas):
        randoms['WEIGHT'] *= alpha
    
    # Concatenate catalogs
    data = np.concatenate(list_data)
    randoms = np.concatenate(list_randoms)
    
    return data, randoms

def compute_power_spectrum(data_fn, rand_fns, mag = 20.2, region='GCcomb', output_dir='pk_ANY', 
                          zmin=0.1, zmax=0.4, boxsize=4000., cellsize=6., 
                          kmin=0., kmax=0.3, dk=0.005, nran=18):
    """
    Compute power spectrum multipoles and window functions for BGS_ANY data.
    
    Parameters
    ----------
    data_fn : str or list
        Path to data FITS file(s) - single file for NGC/SGC, list for GCcomb
    rand_fns : list of str
        List of paths to random FITS files - single list for NGC/SGC, list of lists for GCcomb
    mag: float
        Magnitude cut
    region : str, optional
        Region identifier (NGC, SGC, or GCcomb)
    output_dir : str, optional
        Output directory for results
    zmin, zmax : float, optional
        Redshift range
    boxsize : float, optional
        Box size in Mpc/h
    cellsize : float, optional
        Cell size in Mpc/h
    kmin, kmax : float, optional
        k-range for power spectrum
    dk : float, optional
        k-bin width
    nran : int, optional
        Number of random files (for filename)
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    setup_logging()
    
    # Handle input files based on region
    if region == 'GCcomb':
        # For GCcomb, we expect lists of NGC and SGC files
        if not isinstance(data_fn, list) or len(data_fn) != 2:
            raise ValueError("For GCcomb, provide a list of 2 data files [NGC, SGC]")
        if not isinstance(rand_fns[0], list) or len(rand_fns) != 2:
            raise ValueError("For GCcomb, provide a list of 2 random file lists [[NGC_rands], [SGC_rands]]")
        
        # Load all data and randoms
        list_data = []
        list_randoms = []
        
        for i, reg in enumerate(['NGC', 'SGC']):
            # Load data
            with fits.open(data_fn[i]) as hdul:
                data = hdul[1].data
                data = data[(data['Z'] >= zmin) & (data['Z'] <= zmax)]
                list_data.append(data)
            
            # Load randoms
            reg_randoms = []
            for fn in rand_fns[i]:
                with fits.open(fn) as hdul:
                    rand = hdul[1].data
                    rand = rand[(rand['Z'] >= zmin) & (rand['Z'] <= zmax)]
                    reg_randoms.append(rand)
            list_randoms.append(np.concatenate(reg_randoms))
        
        # Concatenate with proper normalization
        data, randoms = concatenate_catalogs(list_data, list_randoms, region='GCcomb')
          
    elif region in ['NGC', 'SGC']:
    # Load data with read-only mode
        if isinstance(data_fn, list):  # Handle case where it's passed as a list
            data_fn = data_fn[0]
        with fits.open(data_fn, mode='readonly') as hdul:
            data = hdul[1].data
            data = select_region(data, region)
            data = data[(data['Z'] >= zmin) & (data['Z'] <= zmax)]
        
        # Load randoms with read-only mode
        randoms = []
        for fn in rand_fns:
            with fits.open(fn, mode='readonly') as hdul:
                rand = hdul[1].data
                rand = select_region(rand, region)
                rand = rand[(rand['Z'] >= zmin) & (rand['Z'] <= zmax)]
                randoms.append(rand)
        randoms = np.concatenate(randoms)
        
    else:
        raise ValueError(f"Unknown region: {region}")

    # Calculate alpha (data/randoms ratio)
    alpha = get_alpha(data, randoms)
    
    # Convert to Cartesian coordinates
    data_x, data_y, data_z = ra_dec_z_to_xyz(data['RA'], data['DEC'], data['Z'])
    rand_x, rand_y, rand_z = ra_dec_z_to_xyz(randoms['RA'], randoms['DEC'], randoms['Z'])

    # Prepare positions and weights
    data_positions = [data_x, data_y, data_z]
    rand_positions = [rand_x, rand_y, rand_z]
    data_weights = data['WEIGHT']
    rand_weights = randoms['WEIGHT']
    
    # Compute power spectrum
    logger.info("Computing power spectrum...")
    result = CatalogFFTPower(
        data_positions1=data_positions,
        randoms_positions1=rand_positions,
        data_weights1=data_weights,
        randoms_weights1=rand_weights,
        edges={'step': dk, 'min': kmin, 'max': kmax},
        boxsize=boxsize,
        cellsize=cellsize,
        los='firstpoint',
        position_type='xyz',
        resampler='tsc',
        interlacing=2,
        ells=(0, 2, 4),
        dtype='f8',
    )
    
    # Save power spectrum
    poles = result.poles
    fn = os.path.join(output_dir, f"pkpoles_BGS_ANY-{mag}_{region}_z{zmin}-{zmax}_default_FKP_lin_nran{nran}_cellsize{cellsize}_boxsize{boxsize}.npy")
    result.save(fn)
    
    # Save as text file
    txt_fn = os.path.join(output_dir, 
                         f"pkpoles_BGS_ANY-{mag}_{region}_z{zmin}-{zmax}_default_FKP_lin_nran{nran}_cellsize{cellsize}_boxsize{boxsize}_d{dk}.txt")
    result.poles.save_txt(txt_fn, header=[f'data_file = {data_fn}', f'randoms_files = {rand_fns}'], complex=True)
    logger.info(f"Saved power spectrum to {txt_fn}")

    # Compute window function
    logger.info("Computing window function...")
    window = CatalogSmoothWindow(
        randoms_positions1=rand_positions, 
        randoms_weights1=rand_weights,
        power_ref=result.poles, 
        edges={'step': dk, 'min': kmin, 'max': kmax},
        position_type='xyz', 
        dtype='f8'
    ).poles
    
    # Save window function
    w_fn = os.path.join(output_dir, f"wmatrix_smooth_BGS_ANY-{mag}_{region}_z{zmin}-{zmax}_default_FKP_lin_nran{nran}_cellsize{cellsize}_boxsize{boxsize}.npy")
    window.save(w_fn)
    logger.info(f"Saved window function to {w_fn}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute power spectrum multipoles and window functions for BGS_ANY data')
    parser.add_argument('--data', type=str, required=True, nargs='+', 
                       help='Path to data FITS file(s) - one for NGC/SGC, two for GCcomb')
    parser.add_argument('--rand', type=str, required=True, nargs='+', action='append',
                       help='Paths to random FITS files (use --rand once for NGC, again for SGC when doing GCcomb)')
    parser.add_argument('--mag', type=float, default=20.2, help='Absolute value of abs mag cut')
    parser.add_argument('--region', type=str, default='GCcomb', choices=['NGC', 'SGC', 'GCcomb'], 
                       help='Region identifier (NGC, SGC, or GCcomb)')
    parser.add_argument('--output-dir', type=str, default='pk_ANY/prerecon', help='Output directory for results')
    parser.add_argument('--zmin', type=float, default=0.1, help='Minimum redshift')
    parser.add_argument('--zmax', type=float, default=0.4, help='Maximum redshift')
    parser.add_argument('--boxsize', type=float, default=4000., help='Box size in Mpc/h')
    parser.add_argument('--cellsize', type=float, default=6, help='Cell size in Mpc/h')
    parser.add_argument('--kmin', type=float, default=0., help='Minimum k-value')
    parser.add_argument('--kmax', type=float, default=0.3, help='Maximum k-value')
    parser.add_argument('--dk', type=float, default=0.005, help='k-bin width')
    parser.add_argument('--nran', type=int, default=18, help='Number of random files (for filename)')
    
    args = parser.parse_args()
    
    # Handle input files based on region
# Handle input files based on region
    if args.region == 'GCcomb':
        if len(args.data) != 2:
            raise ValueError("For GCcomb, provide exactly 2 data files (NGC and SGC)")
        if len(args.rand) != 2:
            raise ValueError("For GCcomb, provide random files in two groups (NGC and SGC)")
    else:
        if len(args.data) != 1:
            raise ValueError(f"For {args.region}, provide exactly 1 data file")
        # For single region, use first (and only) randoms list
        args.rand = args.rand[0]
    
    
    compute_power_spectrum(
        data_fn=args.data,
        rand_fns=args.rand,
        region=args.region,
        mag=args.mag,
        output_dir=args.output_dir,
        zmin=args.zmin,
        zmax=args.zmax,
        boxsize=args.boxsize,
        cellsize=args.cellsize,
        kmin=args.kmin,
        kmax=args.kmax,
        dk=args.dk,
        nran=args.nran
    )

if __name__ == "__main__":
    main()
