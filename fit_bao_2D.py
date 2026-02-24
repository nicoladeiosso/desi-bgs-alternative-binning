import numpy as np
from desilike.theories.galaxy_clustering import BAOPowerSpectrumTemplate, DampedBAOWigglesTracerPowerSpectrumMultipoles
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike import setup_logging
from cosmoprimo.fiducial import DESI
from desilike.profilers import MinuitProfiler
import argparse
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
                kavg = float(parts[2])
                P0 = parse_real(parts[3])
                P2 = parse_real(parts[4])
                P4 = parse_real(parts[5])
                data.append((kavg, P0, P2, P4))
            except ValueError as e:
                print("Parsing error:", e)
                continue
    
    return np.array(data)

parser = argparse.ArgumentParser(description='Run BAO fit with selected mode.')
parser.add_argument('--mode', type=str, default='all', choices=['all', 'profile', 'sample'],
                    help="Execution mode: 'all', 'profile', or 'sample'")
args = parser.parse_args()
mode = args.mode

# ... (Imports and setup_logging) ...

# ... (Previous functions: load_wmatrix, parse_real, etc) ...

# --- REPLACE YOUR DATA LOADING SECTION WITH THIS ---

txt_path = '/global/cfs/cdirs/desi/users/ndeiosso/archive/LSS_old/scripts/prerecon_PIP/pk/pkpoles_BGS_ANY-21.35_GCcomb_0.1_0.4_FKP_lin5_P07000.0.txt'
data_array = read_power_spectrum_data(txt_path)

k = data_array[:, 0]
p0 = data_array[:, 1]
p2 = data_array[:, 2]
p4 = data_array[:, 3]

# 1. Define the mask for the data you want to FIT (56 bins)
mask_fit = (k >= 0.02) & (k <= 0.3)

# 2. Define the mask for the COVARIANCE (60 bins)
# We assume the covariance covers the same max_k but starts at 0.0
# (We relax the lower bound to find the missing 4 bins)
mask_cov = (k >= 0.0) & (k <= 0.3) 

# Verify we found the correct 60 bins for the covariance
if np.sum(mask_cov) != 60:
    print(f"WARNING: Predicted covariance bins: {np.sum(mask_cov)}. Expected 60. Adjust 'mask_cov' limits.")

# 3. Filter Data
k_selected = k[mask_fit]
p0_selected = p0[mask_fit]
p2_selected = p2[mask_fit]
p4_selected = p4[mask_fit]

print(f"Selected data bins per pole: {len(k_selected)}") # Should be 56

ell_to_include = [0, 2]
data_dict = {0: p0_selected, 2: p2_selected, 4: p4_selected}
data = np.concatenate([data_dict[ell] for ell in ell_to_include])

# 4. Load Full Covariance (180x180)
cov_full = np.loadtxt('/global/cfs/cdirs/desi/users/ndeiosso/archive/BGS_ANY_DR2/DR2/LSS/loa-v1/LSScats/v1.1/desipipe/cov_2pt/thecov/v1.1/prerecon_PIP/cov_gaussian_BGS_ANY-21.35_GCcomb_z0.1_0.4.txt')

# ... (Previous code: steps 1 to 4 remain the same) ...

# 5. Calculate Slicing Indices
k_cov_vals = k[mask_cov]   # The 60 k-values in the cov
k_fit_vals = k[mask_fit]   # The 56 k-values in the fit

# Find indices in the 60-element array that correspond to the 56-element array
keep_indices_per_pole = [i for i, k_val in enumerate(k_cov_vals) if k_val in k_fit_vals]

# Expand ONLY to the poles we want to include
final_indices = []
n_bins_cov = 60         # Size of one pole in original covariance
cov_order = [0, 2, 4]   # The order of poles in the covariance file

for ell in ell_to_include:
    if ell in cov_order:
        # Determine which block (0, 1, or 2) this pole belongs to
        block_idx = cov_order.index(ell) 
        offset = block_idx * n_bins_cov
        
        # Add indices for this specific pole
        final_indices.extend([idx + offset for idx in keep_indices_per_pole])

# 6. Apply Slice
cov = cov_full[np.ix_(final_indices, final_indices)]

print(f"Covariance sliced to: {cov.shape}") 
# If ell_to_include=[0,2], this will now be (112, 112).
# If ell_to_include=[0,2,4], this will be (168, 168).

# ... (Continue with wmatrix loading) ...

print(f"Covariance sliced to: {cov.shape}") # Should be (168, 168)

# ... (Continue with wmatrix loading using the REAL filename) ...
wmatrix = load_wmatrix('/global/cfs/cdirs/desi/users/ndeiosso/archive/LSS_old/scripts/prerecon_PIP/pk/wmatrix_smooth_BGS_ANY-21.35_GCcomb_0.1_0.4_FKP_lin_P07000.0.npy')

# ... (Rest of your script) ...

#wm = np.load('/global/cfs/cdirs/desi/users/ndeiosso/archive/LSS/scripts/postrecon/pk/wmatrix_smooth_BGS_BRIGHT-21.35_GCcomb_0.1_0.4_FKP_lin_P07000.0.npy', allow_pickle = True).item()

print("\nData:")
print(f"k lenght: {len(k_selected)}")
print(f"k values: {k_selected[:5]}...{k_selected[-5:]}")
print(f"P0: {p0_selected[:5]}...{p0_selected[-5:]}")
print(f"P2: {p2_selected[:5]}...{p2_selected[-5:]}")

z = 0.295 #wm['attrs']['zeff']
print(z)
template = BAOPowerSpectrumTemplate(z=z, fiducial='DESI', apmode='qisoqap')
theory = DampedBAOWigglesTracerPowerSpectrumMultipoles(
    template=template,
    ells=ell_to_include,
    broadband='pcs'
)

print(theory.params.names())

observable = TracerPowerSpectrumMultipolesObservable(
    data=data,
    covariance=cov,
    wmatrix=wmatrix,
    k=k_selected,
    kinlim=(0.001, 0.35),
    ells=ell_to_include,
    theory=theory
)

likelihood = ObservablesGaussianLikelihood(observables=[observable])

params = likelihood.runtime_info.pipeline.params
print("\nActive parameters:")
print(sorted(params.basenames()))
# Parameters to fix for 1D analysis
#params['qap'].update(value=1., fixed=True)

params['dbeta'].update(fixed=False)
#params['qap'].update(value=1., fixed=True)
params['sigmapar'].update(fixed=False) 
params['sigmaper'].update(fixed=False)  

#for name in params.basenames():
#    if name.startswith('al2_'):
#        params[name].update(value=0., fixed=True)
    
for name in params.basenames():
    if name.startswith('al0_'): params[name].update(prior={'dist': 'norm', 'loc': 0., 'scale': 1e4}, fixed=False)
    if name.startswith('al2_'): params[name].update(prior={'dist': 'norm', 'loc': 0., 'scale': 1e4}, fixed=False)
    if name.startswith('al4_'): params[name].update(value=0., fixed=True)    
    if name.startswith('bl*_'): params[name].update(value=0., fixed=True)
        
params['b1'].update(prior={'limits': [0.2, 4.]})
params['qiso'].update(prior={'limits': [0.8, 1.2]})
params['qap'].update(prior={'limits': [0.8, 1.2]})

params['dbeta'].update(prior={'limits': [0.7, 1.3]})

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
##
###Fit
if mode in ['all', 'profile']:
    profiler = MinuitProfiler(likelihood, seed=42)
    print("\nFitting BAO starting...")
    profiles = profiler.maximize(niterations=10)
    #
    #
    print("\n" + "="*50 + " RESULTS " + "="*50)
    print(profiles.to_stats(tablefmt='pretty'))

if mode in ['all', 'sample']:
    from desilike.samplers import EmceeSampler
    
    nchains = 8
    burnin = 0.5
    thin = 10
    
    for param in likelihood.all_params.select(basename=['al*_*', 'bl*_*']):
        if param.varied: param.update(derived='.prec')
            
    chain_files = [f'fit_output_prerecon/chain_bao_{i}.npy' for i in range(nchains)]
    chains = nchains
    save_fn = [f'fit_output_prerecon/chain_bao_{i}.npy' for i in range(nchains)]
    sampler = EmceeSampler(likelihood, chains=nchains, nwalkers=4 * len(likelihood.varied_params), seed=42, save_fn=save_fn)
    chains = sampler.run(min_iterations=200, max_iterations=100000, check={'max_eigen_gr': 0.005})
    
    from desilike.samples import Chain, plotting

    mpicomm = sampler.mpicomm
    choice = None

    if mpicomm.rank == 0:
        attrs = {'zeff': z}
        for chain_file in save_fn:
            chain = Chain.load(chain_file)
            chain.attrs.update(attrs)
            chain.save(chain_file)
        
        chain = Chain.concatenate([
            Chain.load(f).remove_burnin(burnin)[::thin] for f in save_fn
        ])
        
        choice = chain.choice(index='argmax', input=True)

    likelihood(**mpicomm.bcast(choice, root=0))

    if mpicomm.rank == 0:
        print(chain.to_stats(tablefmt='pretty'))
        plotting.plot_triangle(chain, fn='fit_output_prerecon/chain_bao_triangle.png')

        for tablefmt, ext in {'pretty': 'txt', 'latex_raw': 'tex'}.items():
            chain.to_stats(tablefmt=tablefmt, fn=f'fit_output_prerecon/chain_bao_stats.{ext}')




