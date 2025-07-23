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
from pypower import (CatalogFFTPower, CatalogSmoothWindow, PowerSpectrumSmoothWindow,
                    BaseMatrix, PowerSpectrumSmoothWindowMatrix, PowerSpectrumOddWideAngleMatrix,
                    mpi, setup_logging)
import logging
from pypower import mpi
from matplotlib import pyplot as plt

mpicomm = mpi.COMM_WORLD
mpirank = mpicomm.rank

# Initialize logging
logger = logging.getLogger(__name__)

# Initialize DESI cosmology
cosmo = DESI()
h = 100*cosmo.h  # Hubble parameter (H0 = 100h km/s/Mpc)

def redshift_to_comoving_distance(z):
    """Convert redshift to comoving distance (Mpc/h) using DESI cosmology"""
    cosmo = DESI()
    return cosmo.comoving_radial_distance(z)

from pycorr import TwoPointCorrelationFunction

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

from pycorr import TwoPointCorrelationFunction
import numpy as np

def compute_angular_weights_rd(cat1, cat2, rand1=None, rand2=None, nthreads=64, mpicomm=None):
    ra1, dec1 = cat1['RA'], cat1['DEC']
    ra2, dec2 = cat2['RA'], cat2['DEC']
    w1, w2 = cat1['WEIGHT'], cat2['WEIGHT']

    pos1 = [ra1, dec1]
    pos2 = [ra2, dec2]

    # Se non forniti, usa i randoms uguali ai dati (basta per D1D2 ecc.)
    if rand1 is None:
        rand1 = cat1
    if rand2 is None:
        rand2 = cat2

    rand_ra1, rand_dec1 = rand1['RA'], rand1['DEC']
    rand_ra2, rand_dec2 = rand2['RA'], rand2['DEC']
    rand_w1, rand_w2 = rand1['WEIGHT'], rand2['WEIGHT']

    rand_pos1 = [rand_ra1, rand_dec1]
    rand_pos2 = [rand_ra2, rand_dec2]

    edges=np.logspace(-4., 0.5, 41)

    D1D2 = TwoPointCorrelationFunction('theta', edges,
        data_positions1=pos1, data_weights1=w1,
        data_positions2=pos2, data_weights2=w2,
        randoms_positions1=rand_pos1, randoms_weights1=rand_w1,
        randoms_positions2=rand_pos2, randoms_weights2=rand_w2,
        estimator='weight', engine='corrfunc', position_type='rd',
        nthreads=nthreads, mpicomm=mpicomm)

    D1R2 = TwoPointCorrelationFunction('theta', edges,
        data_positions1=pos1, data_weights1=w1,
        data_positions2=rand_pos2, data_weights2=rand_w2,
        randoms_positions1=rand_pos1, randoms_weights1=rand_w1,
        randoms_positions2=rand_pos2, randoms_weights2=rand_w2,
        estimator='weight', engine='corrfunc', position_type='rd',
        nthreads=nthreads, mpicomm=mpicomm)

    R1R2 = TwoPointCorrelationFunction('theta', edges,
        data_positions1=rand_pos1, data_weights1=rand_w1,
        data_positions2=rand_pos2, data_weights2=rand_w2,
        randoms_positions1=rand_pos1, randoms_weights1=rand_w1,
        randoms_positions2=rand_pos2, randoms_weights2=rand_w2,
        estimator='weight', engine='corrfunc', position_type='rd',
        nthreads=nthreads, mpicomm=mpicomm)

    return {
        'D1D2_twopoint_weights': D1D2,
        'D1R2_twopoint_weights': D1R2,
        'R1R2_twopoint_weights': R1R2,
    }

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

def postprocess_power_spectrum(power_fn, rebinning_factors=(1, 5), output_power_nodirect=None, 
                              windows=None, output_wmatrix=None, output_wmatrix_nodirect=None):

    import sys
    from pypower import PowerSpectrumStatistics, PowerSpectrumSmoothWindow, PowerSpectrumOddWideAngleMatrix, PowerSpectrumSmoothWindowMatrix
    sys.path.insert(0, '/pscratch/sd/n/ndeiosso/desi-y3-kp/desi_y3_files/..')
    from desi_y3_files import load
    """
    Post-process power spectrum results including:
    - Saving rebinned versions
    - Computing window function matrices
    - Creating plots
    
    Parameters
    ----------
    power_fn : str
        Path to power spectrum file
    rebinning_factors : tuple, optional
        Factors to rebin the power spectrum by
    output_power_nodirect : str, optional
        Path to save power spectrum without direct counting terms
    windows : str or list, optional
        Path(s) to window function files
    output_wmatrix : str, optional
        Path to save window matrix
    output_wmatrix_nodirect : str, optional
        Path to save window matrix without direct counting terms
    """
    power = PowerSpectrumStatistics.load(power_fn)
    fn_nodirect = power_nodirect = None

    if output_power_nodirect is not None:
        power_nodirect = power.deepcopy()
        power_nodirect.power_direct_nonorm[...] = 0.
        for name in ['corr_direct_nonorm', 'sep_direct']: 
            setattr(power_nodirect, name, None)
        power_nodirect.save(output_power_nodirect)
        fn_nodirect = str(output_power_nodirect)
    
    if mpirank == 0:
        # Save rebinned versions and create plots
        for output, fn in zip([power, power_nodirect], [power_fn, fn_nodirect]):
            if output is None: 
                continue

            for factor in rebinning_factors:
                rebinned = output[:(output.shape[0] // factor) * factor:factor]
                db = rebinned.edges[0][1] - rebinned.edges[0][0]
                ff = fn.replace('.npy', '_d{:.3f}'.format(db))
                rebinned.save_txt(ff + '.txt')
                
                # Create plot
                fig, ax = plt.subplots(figsize=(10, 6))
                for ell in output.ells:
                    label = r'$\ell = {}$'.format(ell)
                    ax.plot(output.k, output.k * output(ell=ell), label=label)
                ax.set_xlabel(r'$k$ [$h\,\mathrm{Mpc}^{-1}$]')
                ax.set_ylabel(r'$k P_\ell(k)$ [$h^{-2} \mathrm{Mpc}^2$]')
                ax.legend()
                ax.grid(True)
                fig.savefig(ff + '.png', bbox_inches='tight', dpi=150)
                plt.close(fig)

    # Compute window function matrix if requested
    if output_wmatrix is not None or output_wmatrix_nodirect is not None:
        if windows is None:
            raise ValueError("Window functions must be provided to compute window matrix")
            
        if isinstance(windows, (tuple, list)):
            argsort = np.argsort([np.max(window.attrs['boxsize']) for window in windows])[::-1]
            windows = [windows[ii] for ii in argsort]
            window = windows[0].concatenate_x(*windows, frac_nyq=0.9)
        else:
            window = windows
            
        # Let us compute the wide-angle and window function matrix
        ellsin = (0, 2, 4)  # input (theory) multipoles
        wa_orders = 1  # wide-angle order
        sep = np.geomspace(1e-4, 1e5, 1024 * 16)  # configuration space separation for FFTlog
        kin_rebin = 2  # rebin input theory to save memory
        kin_lim = (0, 2e1)  # pre-cut input (theory) ks to save some memory
        
        # Input projections for window function matrix:
        # theory multipoles at wa_order = 0, and wide-angle terms at wa_order = 1
        projsin = tuple(ellsin) + tuple(PowerSpectrumOddWideAngleMatrix.propose_out(ellsin, wa_orders=wa_orders))
        
        for output, nodirect in zip([output_wmatrix, output_wmatrix_nodirect], [False, True]):
            if output is None: 
                continue
                
            if nodirect:
                current_window = window.deepcopy()
                current_window.power_direct_nonorm[...] = 0.
                for name in ['corr_direct_nonorm', 'sep_direct']: 
                    setattr(current_window, name, None)
            else:
                current_window = window
                
            # Window matrix
            wmatrix = PowerSpectrumSmoothWindowMatrix(
                power, projsin=projsin, window=current_window, 
                sep=sep, kin_rebin=kin_rebin, kin_lim=kin_lim
            )
            
            # We resum over theory odd-wide angle
            wmatrix.resum_input_odd_wide_angle()
            wmatrix.attrs.update(power.attrs)
            
            if mpirank == 0:
                wmatrix.save(output)

def compute_power_spectrum(data_fn, rand_fns, mag=20.2, region='GCcomb', output_dir='pk_ANY', 
                          zmin=0.1, zmax=0.4, boxsize=4000., cellsize=6., 
                          kmin=0., kmax=0.522, dk=0.005, nran=18,
                          compute_wmatrix=True):
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
    compute_wmatrix : bool, optional
        Whether to compute the window function matrix
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
    
    logger.info("Computing angular weights for each pair combination...")

    #wang = compute_angular_weights_rd(data, data, rand1=randoms, rand2=randoms, mpicomm=mpicomm)
    #w_dd = wang['D1D2_twopoint_weights']
    #w_dr = wang['D1R2_twopoint_weights']
    #w_rr = wang['R1R2_twopoint_weights']

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
        mpicomm=mpicomm,
        #D1D2_twopoint_weights=w_dd,
        #D1R2_twopoint_weights=w_dr,
        #R1D2_twopoint_weights=w_dr,  # Using same as D1R2
        #R1R2_twopoint_weights=w_rr
    ).poles
    
    # Save power spectrum
    if mpirank == 0:
        # Main power spectrum file
        pk_fn = os.path.join(output_dir, 
                           f"pkpoles_BGS_BRIGHT-{mag}_{region}_z{zmin}-{zmax}_default_FKP_lin_nran{nran}_cellsize{cellsize}_boxsize{boxsize}.npy")
        result.save(pk_fn)
        
        # Text version
        txt_fn = os.path.join(output_dir, 
                           f"pkpoles_BGS_BRIGHT-{mag}_{region}_z{zmin}-{zmax}_default_FKP_lin_nran{nran}_cellsize{cellsize}_boxsize{boxsize}_d{dk}.txt")
        result.save_txt(txt_fn, header=[f'data_file = {data_fn}', f'randoms_files = {rand_fns}'], complex=True)
        logger.info(f"Saved power spectrum to {txt_fn}")

    
    # Compute window functions at different box sizes
    windows = []
    if compute_wmatrix:
        logger.info("Computing window functions...")
        boxscales = [1., 5., 20.]  # Different box sizes to combine
        boxsizes = boxsize * np.array(boxscales)
        
        for iboxsize, current_boxsize in enumerate(boxsizes):
            window = CatalogSmoothWindow(
                randoms_positions1=rand_positions, 
                randoms_weights1=rand_weights,
                power_ref=result, 
                edges={'step': 2. * np.pi / np.max(boxsizes)},
                boxsize=current_boxsize,
                position_type='xyz', 
                dtype='f8',
                mpicomm=mpicomm
            ).poles
            
            if mpirank == 0:
                window_fn = os.path.join(output_dir,
                                       f"window_{iboxsize}_BGS_BRIGHT-{mag}_{region}_z{zmin}-{zmax}_default_FKP_lin_nran{nran}_cellsize{cellsize}_boxsize{current_boxsize}.npy")
                window.save(window_fn)
                windows.append(window)
        
        # Combine windows from different box sizes
        if mpirank == 0 and windows:
            combined_window = windows[0].concatenate_x(*windows[::-1], frac_nyq=0.9)
            window_fn = os.path.join(output_dir,
                                   f"window_combined_BGS_BRIGHT-{mag}_{region}_z{zmin}-{zmax}_default_FKP_lin_nran{nran}_cellsize{cellsize}_boxsize{boxsize}.npy")
            combined_window.save(window_fn)
            
            # Window matrix file names
            wmatrix_fn = os.path.join(output_dir,
                                    f"wmatrix_smooth_BGS_BRIGHT-{mag}_{region}_z{zmin}-{zmax}_default_FKP_lin_nran{nran}_cellsize{cellsize}_boxsize{boxsize}.npy")
    
            # Post-process to create window matrices
            postprocess_power_spectrum(
                power_fn=pk_fn,
                #output_power_nodirect=pk_nodirect_fn,
                windows=combined_window,
                output_wmatrix=wmatrix_fn,
                #output_wmatrix_nodirect=wmatrix_nodirect_fn
            )

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
    parser.add_argument('--kmax', type=float, default=0.522, help='Maximum k-value')
    parser.add_argument('--dk', type=float, default=0.001, help='k-bin width')
    parser.add_argument('--nran', type=int, default=18, help='Number of random files (for filename)')
    parser.add_argument('--no-wmatrix', action='store_false', dest='compute_wmatrix',
                       help='Skip computing the window function matrix')
    
    args = parser.parse_args()
    
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
        nran=args.nran,
        compute_wmatrix=args.compute_wmatrix
    )

if __name__ == "__main__":
    main()
