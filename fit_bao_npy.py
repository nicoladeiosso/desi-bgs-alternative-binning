import numpy as np
from desilike.theories.galaxy_clustering import BAOPowerSpectrumTemplate, DampedBAOWigglesTracerPowerSpectrumMultipoles
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike import setup_logging
from cosmoprimo.fiducial import DESI
from desilike.profilers import MinuitProfiler
from pypower import CatalogMesh, MeshFFTPower, CatalogFFTPower, PowerSpectrumStatistics, utils, setup_logging
import re

setup_logging()
cosmo = DESI()

def load_wmatrix(self, mode='poles'):
    """Load matrix."""
    from pypower import MeshFFTWindow, BaseMatrix
    toret = MeshFFTWindow.load(self)
    try:
        toret = getattr(toret, mode)
    except AttributeError:
        toret =  BaseMatrix.load(self)
    return toret

def load_cov(self):
    """Load covariance."""
    from desilike.observables import ObservableCovariance
    return ObservableCovariance.load(self)


def parse_real(complex_str):
    pattern = re.compile(r'([+-]?\d+\.\d+e[+-]?\d+)')
    match = pattern.match(complex_str)
    if match:
        return float(match.group(1))
    raise ValueError("Unknow format")

def read_power_spectrum_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.split()
            if not parts or not parts[0].isdigit():
                continue
            
            try:
                kmid = float(parts[1])
                kavg = float(parts[2])
                P0 = parse_real(parts[3])
                P2 = parse_real(parts[4])
                P4 = parse_real(parts[5])
                data.append((kmid, kavg, P0, P2, P4))
            except ValueError as e:
                print("Parsing error:", e)
                continue
    
    return np.array(data)

txt_path = '/global/cfs/cdirs/desi/survey/catalogs/DA2/analysis/loa-v1/LSScats/v1.1/BAO/unblinded/desipipe/2pt/pk/pkpoles_BGS_BRIGHT-21.35_GCcomb_z0.1-0.4_default_FKP_lin_nran18_cellsize6_boxsize4000_d0.001.txt'

data_array = read_power_spectrum_data(txt_path)

#k = data_array[:, 0]

k = poles_npy.kavg

fn_npy = "/global/cfs/cdirs/desi/survey/catalogs/DA2/analysis/loa-v1/LSScats/v1.1/BAO/unblinded/desipipe/2pt/pk/pkpoles_BGS_BRIGHT-21.35_GCcomb_z0.1-0.4_default_FKP_lin_nran18_cellsize6_boxsize4000.npy"
poles_npy = PowerSpectrumStatistics.load(fn_npy)
pk = poles_npy.get_power(divide_wnorm=True, remove_shotnoise=True)

cov = load_cov('/global/cfs/cdirs/desi/survey/catalogs/DA2/analysis/loa-v1/LSScats/v1.1/BAO/unblinded/desipipe/cov_2pt/thecov/v1.1/covariance_power_BGS_BRIGHT-21.35_GCcomb_z0.1-0.4_default_FKP_lin.npy')

# ℓ to use
ells_to_use = [0]
all_ells = [0, 2, 4]  #Cov matrix order
mask = (k >= 0.02) & (k <= 0.3)
k_selected = k[mask]
k_indices = np.where(mask)[0]
n_k_selected = len(k_indices)
#
#indices = []
#for i, ell in enumerate(all_ells):
#    if ell in ells_to_use:
#        indices.extend([i * n_k_total + idx for idx in k_indices])
#
#cov = cov[np.ix_(indices, indices)]
#print(cov)
#
wmatrix = load_wmatrix('/global/cfs/cdirs/desi/survey/catalogs/DA2/analysis/loa-v1/LSScats/v1.1/BAO/unblinded/desipipe/2pt/pk/wmatrix_smooth_BGS_BRIGHT-21.35_GCcomb_z0.1-0.4_default_FKP_lin_nran18_cellsize6_boxsize4000.npy')


z = 0.1
template = BAOPowerSpectrumTemplate(z=z, fiducial='DESI', apmode='qisoqap')
theory = DampedBAOWigglesTracerPowerSpectrumMultipoles(
    template=template,
    ells=ells_to_use,
    broadband='pcs'
)

print(theory.params.names())

observable = TracerPowerSpectrumMultipolesObservable(
    data=pk,
    covariance=cov,
    wmatrix=wmatrix,
    k=k_selected,
    kinlim=(0.001, 0.35),
    ells=ells_to_use,
    theory=theory
)

likelihood = ObservablesGaussianLikelihood(observables=[observable])

params = likelihood.runtime_info.pipeline.params
print("\nActive parameters:")
print(sorted(params.basenames()))
# Parameters to fix for 1D analysis
#params['qap'].update(value=1., fixed=True)

params['dbeta'].update(value=1., fixed=True)
params['qap'].update(value=1., fixed=True)
params['sigmapar'].update(fixed=False) 
params['sigmaper'].update(fixed=False)  

#for name in params.basenames():
#    if name.startswith('al2_'):
#        params[name].update(value=0., fixed=True)
    
for name in params.basenames():
    if name.startswith('al2_'): params[name].update(value=0., fixed=True)
    if name.startswith('al4_'): params[name].update(value=0., fixed=True)    
    if name.startswith('al0_'): params[name].update(prior={'dist': 'norm', 'loc': 0., 'scale': 1e4}, fixed=False)
        
params['b1'].update(prior={'limits': [0.2, 4.]})
params['qiso'].update(prior={'limits': [0.8, 1.2]})
params['sigmas'].update(prior={'dist': 'norm', 'loc': 2.0, 'scale': 2.0, 'limits': [0., 20.]}, fixed = False)

params['sigmapar'].update(prior={'dist': 'norm', 'loc': 10.0, 'scale': 2.0, 'limits': [0., 20.]}, fixed=False)
params['sigmaper'].update(prior={'dist': 'norm', 'loc': 6.5, 'scale': 1.0, 'limits': [0., 20.]}, fixed=False)

observable.init.theory = theory


marg = True
if marg:
    for param in likelihood.all_params.select(basename=['al*_*']):
        param.update(derived='.auto')
if likelihood.mpicomm.rank == 0:
    likelihood.log_info('Use analytic marginalization for {}.'.format(likelihood.all_params.names(solved=True)))

solved_params = likelihood.all_params.select(solved=True)
print("Marginalized Params(solved):")
print(sorted(p.basename for p in solved_params))

#Check
print("\n" + "="*50 + " CONFIG " + "="*50)
#
##Fit
profiler = MinuitProfiler(likelihood, seed=42)
print("\nFitting BAO starting...")
profiles = profiler.maximize(niterations=10)
#
#
print("\n" + "="*50 + " RESULTS " + "="*50)
print(profiles.to_stats(tablefmt='pretty'))
