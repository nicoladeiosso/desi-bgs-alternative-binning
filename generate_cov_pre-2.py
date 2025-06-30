import faulthandler
faulthandler.enable()
import yaml
import argparse
import numpy as np
import thecov.covariance
import thecov.geometry
import thecov.base
import thecov.utils
from cosmoprimo.fiducial import DESI
from mockfactory import Catalog, utils
from pypower import CatalogFFTPower, PowerSpectrumStatistics
import os
import logging
import time

# Parsing command line arguments
parser = argparse.ArgumentParser(description='Generate Covariance Matrix')
parser.add_argument('--tracer', required=True, help='Tracer type')
parser.add_argument('--region', required=True, help='Region type')
parser.add_argument('--boxsize', required=True, help='Box size')
parser.add_argument('--zmin', type=float, required=True, help='Minimum redshift')
parser.add_argument('--zmax', type=float, required=True, help='Maximum redshift')
args = parser.parse_args()

tracer = args.tracer
region = args.region
boxsize = args.boxsize
zmin = args.zmin
zmax = args.zmax

cosmo = DESI()

# Setting up the logger
logger = logging.getLogger('GenerateCovariance')
logging.basicConfig(filename='logs/covariance_{}_{}_{}_{}_pre.log'.format(tracer, region, zmin, zmax), level=logging.DEBUG)

# Load configuration from YAML file
try:
    with open("config_covariance_pre.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
        logger.info('Successfully loaded configuration from config_covariance_pre.yaml')
except Exception as e:
    logger.error(f'Error loading configuration: {str(e)}')
    raise

# Adjust paths based on the tracer
#if tracer == 'LRG+ELG_LOPnotqso':
    #config['pk_dir'] = '/global/cfs/cdirs/desi/survey/catalogs/DA2/analysis/kibo-v1/LSScats/v1/BAO/blinded/desipipe/2pt/recon_sm15_IFFT_recsym_z0.8-1.1/pk/'
    #config['catalog_dir'] = '/global/cfs/cdirs/desi/survey/catalogs/DA2/analysis/kibo-v1/LSScats/v1/BAO/blinded/desipipe/2pt/recon_sm15_IFFT_recsym_z0.8-1.1/'
    #config['randoms_dir'] = '/global/cfs/cdirs/desi/survey/catalogs/DA2/analysis/loa-v1/LSScats/v1/BAO/blinded/desipipe/2pt/recon_sm15_IFFT_recsym_z0.8-1.1/'
    
#elif tracer == 'QSO':
    #config['pk_dir'] = '/global/cfs/cdirs/desi/survey/catalogs/DA2/analysis/kibo-v1/LSScats/v1/BAO/blinded/desipipe/2pt/recon_sm30_IFFT_recsym_z0.8-2.1/pk/'
    #config['catalog_dir'] = '/global/cfs/cdirs/desi/survey/catalogs/DA2/analysis/kibo-v1/LSScats/v1/BAO/blinded/desipipe/2pt/recon_sm30_IFFT_recsym_z0.8-2.1/'
    #config['randoms_dir'] = '/global/cfs/cdirs/desi/survey/catalogs/DA2/analysis/kibo-v1/LSScats/v1/BAO/blinded/desipipe/2pt/recon_sm30_IFFT_recsym_z0.8-2.1/'
#else:
#    raise ValueError(f"Unknown tracer type: {tracer}")

pk_dir = config['pk_dir']
catalog_dir = config['catalog_dir']
randoms_dir = config['randoms_dir']
kmin = config['kmin']
kmax = config['kmax']
kstep = config['kstep']
window_kernel = config['window_kernel'].format(tracer=tracer, region=region, zmin=zmin, zmax=zmax, boxsize=boxsize)
output_covariance = config['output_covariance'].format(tracer=tracer, region=region, zmin=zmin, zmax=zmax, boxsize=boxsize)

# ----------- DATA PROCESSING FUNCTIONS --------------
def get_alpha(data, randoms):
    alpha = len(data) / len(randoms)
    logger.info(f'Estimated alpha is {alpha}')
    return alpha

def select_region(catalog, region):
    mask = catalog.trues()
    if region == 'NGC':
        mask &= (catalog['RA'] > 88) & (catalog['RA'] < 303)
    if region == 'SGC':
        mask &= (catalog['RA'] < 88) | (catalog['RA'] > 303)
    return catalog[mask]

def concatenate(list_data, list_randoms, region):
    list_data = [select_region(catalog, region) for catalog in list_data]
    list_randoms = [select_region(catalog, region) for catalog in list_randoms]
    wsums_data = [data['WEIGHT'].csum() for data in list_data]
    wsums_randoms = [randoms['WEIGHT'].csum() for randoms in list_randoms]
    alpha = sum(wsums_data) / sum(wsums_randoms)
    alphas = [wsum_data / wsum_randoms / alpha for wsum_data, wsum_randoms in zip(wsums_data, wsums_randoms)]
    if list_data[0].mpicomm.rank == 0:
        print('Renormalizing randoms weights by {} before concatenation.'.format(alphas))
        logger.info(f'Renormalizing randoms weights by {alphas} before concatenation.')
    for randoms, alpha in zip(list_randoms, alphas):
        randoms['WEIGHT'] *= alpha
    return Catalog.concatenate(list_data), Catalog.concatenate(list_randoms)

data_fns = [os.path.join(catalog_dir, '{}_{}_clustering.dat.fits'.format(tracer, reg)) for reg in ['NGC', 'SGC']]
randoms_fns = [os.path.join(catalog_dir, '{}_{}_0_clustering.ran.fits'.format(tracer, reg)) for reg in ['NGC', 'SGC']]
data_NSGC = [Catalog.read(fn) for fn in data_fns]
randoms_NSGC = [Catalog.read(fn) for fn in randoms_fns]

if region == 'NGC':
    data, randoms = data_NSGC[0], randoms_NSGC[0]
elif region == 'SGC':
    data, randoms = data_NSGC[1], randoms_NSGC[1]
elif region == 'GCcomb':
    data, randoms = concatenate(data_NSGC, randoms_NSGC, region)
else:
    raise ValueError('Unknown region {}'.format(region))

start_time = time.time()

randoms['POSITION'] = utils.sky_to_cartesian(cosmo.comoving_radial_distance(randoms['Z']), randoms['RA'], randoms['DEC'], degree=True)
alpha = get_alpha(data=data, randoms=randoms)

# --------- COMPUTING COVARIANCE --------------
geometry = thecov.geometry.SurveyGeometry(randoms, alpha=alpha, nmesh=32)

covariance = thecov.covariance.GaussianCovariance(geometry)
covariance.set_kbins(kmin, kmax, kstep)

from pypower import PowerSpectrumMultipoles

#pk_filename = os.path.join(pk_dir, f'pkpoles_{tracer}_{region}_z{zmin}-{zmax}_default_FKP_lin_nran18_cellsize6_{boxsize}.npy')
#power = PowerSpectrumMultipoles.load(pk_filename)
pk = covariance.load_pypower_file(os.path.join(pk_dir, f'pkpoles_{tracer}_{region}_z{zmin}-{zmax}_default_FKP_lin_nran18_cellsize6_{boxsize}.npy'), set_shotnoise=True)
#covariance.load_pypower(power)
#covariance.alpha = alpha
#covariance.set_shotnoise(shotnoise=power.shotnoise)
#pk = pk[:400:5]
#shotnoise = pk.shotnoise
#P = pk(ell=[0, 2, 4], remove_shotnoise=True, complex=False)
#P0, P2, P4 = pk[0, :], pk[1, :], pk[2, :]

#for p0, p2, p4 in zip(P0, P2, P4):
#covariance.set_galaxy_pk_multipole(pk, 0)
#covariance.set_galaxy_pk_multipole(pk, 2)
#covariance.set_galaxy_pk_multipole(pk, 4)

#covariance.set_shotnoise(shotnoise)

#if os.path.exists(window_kernel):
#    logger.info(f'Loading window kernels from {window_kernel}...')
#    geometry.load_window_kernels(window_kernel)
#geometry.set_resume_file(window_kernel)
covariance.compute_covariance()

#if not os.path.exists(window_kernel):
#    geometry.get_window_kernels(window_kernel)
covariance.symmetrize()
covariance.savetxt(output_covariance)

end_time = time.time()
elapsed_time = (end_time - start_time) / 60.

if __name__ == "__main__":
    print(f"Script executed in {elapsed_time:.2f} minutes.")
    print(f"Results saved to {output_covariance}")
    logger.info(f"Script executed in {elapsed_time:.2f} minutes.")
    logger.info(f"Results saved to {output_covariance}")
