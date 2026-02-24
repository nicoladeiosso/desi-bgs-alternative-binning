import os
import numpy as np
import re
import matplotlib.pyplot as plt
from glob import glob
from desilike.theories.galaxy_clustering import BAOPowerSpectrumTemplate, DampedBAOWigglesTracerPowerSpectrumMultipoles
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike.profilers import MinuitProfiler
from desilike import setup_logging
from cosmoprimo.fiducial import DESI

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
    raise ValueError(f"Unknow format: {complex_str}")

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
            except ValueError:
                continue
    return np.array(data)

def get_zeff_from_filename(filename):
    """Return effective redshift based on filename patterns."""
    # Normalize dash to underscore for checking
    fn = filename.replace('-', '_')
    
    if "21.35" in fn:
        if "0.1_0.4" in fn: return 0.295
        if "0.25_0.4" in fn: return 0.3338
        if "0.0_0.5" in fn: return 0.3803
        if "0.3_0.5" in fn: return 0.4246
    elif "20.2" in fn:
        if "0.1_0.4" in fn: return 0.3336
        if "0.1_0.25" in fn: return 0.1923
        if "0.0_0.5" in fn: return 0.4214
        if "0.0_0.3" in fn: return 0.2276
        
    # If no exact match, try to find a default or log warning
    print(f"WARNING: No exact zeff for {filename}, using mid-point.")
    # Fallback: parse numbers and return average
    info = extract_info_from_pk(filename)
    return (float(info['zmin']) + float(info['zmax'])) / 2.0

def extract_info_from_pk(pk_file):
    """
    Parse pk filename. 
    Handles: pkpoles_BGS_BRIGHT-20.2_GCcomb_z0.1-0.25_default...
    And: pkpoles_BGS_ANY-21.35_GCcomb_0.1_0.4_FKP...
    """
    basename = os.path.basename(pk_file)
    
    # 1. Extract tracer (everything between pkpoles_ and the first z or digit)
    # 2. Extract z-range (handles z0.1-0.4 or 0.1_0.4)
    # 3. Extract Ptag (usually at the end before .txt)
    
    # Updated regex:
    # (?P<tracer>.+?) matches the tracer name greedily until it hits a redshift pattern
    # [z_]?(?P<zmin>\d\.\d+)[-_](?P<zmax>\d\.\d+) handles z0.1-0.4, 0.1_0.4, etc.
    pattern = r'pkpoles_(?P<tracer>.+?)[_]z?(?P<zmin>\d\.\d+)[-_](?P<zmax>\d\.\d+)'
    
    m = re.search(pattern, basename)
    if not m:
        raise ValueError(f"Cannot parse pk filename: {basename}")
    
    info = m.groupdict()
    
    # Special handling for Ptag which varies in position
    if '_P' in basename:
        ptag_match = re.search(r'_(P\d+\.\d+)', basename)
        info['Ptag'] = ptag_match.group(1) if ptag_match else "P07000.0"
    else:
        # Default for the new files which seem to use d0.005 instead of Ptag
        info['Ptag'] = "P07000.0" 
        
    return info


def find_matching_files(pk_file, cov_files, wm_files):
    info = extract_info_from_pk(pk_file)
    tracer = info["tracer"]
    zmin, zmax = info["zmin"], info["zmax"]

    # Search for covariance: match tracer and the redshift numbers regardless of separator
    # This looks for "tracer" AND "zmin" AND "zmax" in the filename
    matching_covs = [
        c for c in cov_files 
        if tracer in c and zmin in c and zmax in c
    ]

    # Search for Window Matrix: similar logic
    # WM often has the specific tracer and redshift range
    matching_wm = [
        w for w in wm_files 
        if tracer in w and zmin in w and zmax in w
    ]

    return matching_covs, matching_wm


def get_cov_indices(k, kmin, kmax, n_k_total, ells_to_use, all_ells=[0,2,4]):
    mask = (k >= kmin) & (k <= kmax)
    k_indices = np.where(mask)[0]
    indices = []
    for i, ell in enumerate(all_ells):
        if ell in ells_to_use:
            indices.extend([i * n_k_total + idx for idx in k_indices])
    return indices, k_indices

def standardize_basename(pk_file):
    basename = os.path.basename(pk_file)

    # Rimuovi prefisso
    basename = basename.replace('pkpoles_', '')
    basename = re.sub(r'z(\d+\.\d+)-(\d+\.\d+)', r'z\1_\2', basename)

    basename = re.sub(r'(z\d+\.\d+_\d+\.\d+).*', r'\1', basename)

    return basename


    
def fit_pk_cov(pk_file, cov_file, wm_file, output_dir, is_postrecon=False, kmin=0.02, kmax=0.3, mode='all'):
    basename = os.path.splitext(os.path.basename(pk_file))[0]
    print(f"\n>>> Fit per:\n  PK  = {pk_file}\n  COV = {cov_file}")

    data_array = read_power_spectrum_data(pk_file)
    k = data_array[:, 0]
    
    # 1. Define masks
    mask_fit = (k >= kmin) & (k <= kmax)
    mask_cov = (k >= 0.0) & (k <= kmax) # The full range the covariance covers (usually 60 bins)

    # 2. Filter Data
    k_selected = k[mask_fit]
    p0_selected = data_array[:, 1][mask_fit]
    p2_selected = data_array[:, 2][mask_fit]
    p4_selected = data_array[:, 3][mask_fit]

    ell_to_include = [0, 2] # The ells you want to fit
    data_dict = {0: p0_selected, 2: p2_selected, 4: p4_selected}
    data = np.concatenate([data_dict[ell] for ell in ell_to_include])

    # 3. Handle Covariance Slicing
    cov_full = np.loadtxt(cov_file)
    
    k_cov_vals = k[mask_cov]   # The 60 k-values assumed to be in the original cov blocks
    k_fit_vals = k[mask_fit]   # The 56 k-values we want to keep
    
    # Identify which indices in a single block (e.g., 0-59) to keep (e.g., 4-59)
    keep_indices_per_pole = [i for i, k_val in enumerate(k_cov_vals) if k_val in k_fit_vals]
    
    n_bins_cov_block = len(k_cov_vals) # Should be 60
    cov_order = [0, 2, 4]              # The physical order of blocks in your .txt cov file
    
    final_indices = []
    for ell in ell_to_include:
        if ell in cov_order:
            block_idx = cov_order.index(ell)
            offset = block_idx * n_bins_cov_block
            final_indices.extend([idx + offset for idx in keep_indices_per_pole])

    cov = cov_full[np.ix_(final_indices, final_indices)]
    
    print(f"Data bins: {len(k_selected)} | Cov shape: {cov.shape}") 
    # For 2 poles, this should be (112, 112)

    # ... rest of your theory setup (wmatrix, template, etc.)
    wmatrix = load_wmatrix(wm_file)
    z = get_zeff_from_filename(pk_file)
    
    # (Rest of the script remains similar)
    
    template = BAOPowerSpectrumTemplate(z=z, fiducial='DESI', apmode='qisoqap')
    theory = DampedBAOWigglesTracerPowerSpectrumMultipoles(template=template, ells=ell_to_include, broadband='pcs')
    observable = TracerPowerSpectrumMultipolesObservable(data=data, covariance=cov, wmatrix=wmatrix, k=k_selected, kinlim=(0.001, 0.35), ells=ell_to_include, theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])

    # Parametri
    params = likelihood.runtime_info.pipeline.params
    params['dbeta'].update(fixed=False)
    #params['qap'].update(value=1., fixed=True)
    params['sigmapar'].update(fixed=False)
    params['sigmaper'].update(fixed=False)

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
       
    if is_postrecon:
        params['sigmapar'].update(prior={'dist': 'norm', 'loc': 8.0, 'scale': 2.0, 'limits': [0., 20.]}, fixed=False)
        params['sigmaper'].update(prior={'dist': 'norm', 'loc': 3.0, 'scale': 1.0, 'limits': [0., 20.]}, fixed=False)
    else:
        params['sigmapar'].update(prior={'dist': 'norm', 'loc': 10.0, 'scale': 2.0, 'limits': [0., 20.]}, fixed=False)
        params['sigmaper'].update(prior={'dist': 'norm', 'loc': 6.5, 'scale': 1.0, 'limits': [0., 20.]}, fixed=False)
    
    observable.init.theory = theory

    marg = True
    if marg:
        for param in likelihood.all_params.select(basename=['al*_*']):
            param.update(derived='.auto')
    if likelihood.mpicomm.rank == 0:
        likelihood.log_info('Use analytic marginalization for {}.'.format(likelihood.all_params.names(solved=True)))

    if mode in ['all', 'profile']:
        profiler = MinuitProfiler(likelihood, seed=42)
        profiles = profiler.maximize(niterations=50)
    
        # Salva tabella
        result_path = os.path.join(output_dir, f'{basename}_fit_results.txt')
        with open(result_path, 'w') as f:
            f.write(profiles.to_stats(tablefmt='pretty'))
    
        result_tex_path = os.path.join(output_dir, f'{basename}_fit_results.tex')
        with open(result_tex_path, 'w') as f:
            f.write(profiles.to_stats(tablefmt='latex'))
    
        print(f"Salvati: {result_path}, {result_tex_path}")
    #model_prediction = observable(**params.to_dict())
    ## Salva plot
    #plt.figure()
    #plt.plot(k_selected, p0_selected, 'o', label='Data P0')
    #plt.plot(k_selected, model_prediction[0], '-', label='Model P0')
    #plt.xlabel('k [h/Mpc]')
    #plt.ylabel('P(k)')
    #plt.title(basename)
    #plt.legend()
    #plt.savefig(os.path.join(output_dir, f'{basename}_fit_plot.png'))
    #plt.close()

        print(f"Finito fit per {basename}")

    if mode in ['all', 'sample']:
        from desilike.samplers import EmceeSampler
        nchains = 8
        burnin = 0.5
        thin = 10
        for param in likelihood.all_params.select(basename=['al*_*', 'bl*_*']):
            if param.varied: param.update(derived='.prec')
                
        chain_files = [os.path.join(output_dir, f'chain_{basename}_{i}.npy') for i in range(nchains)]
        chains = nchains
        save_fn = [os.path.join(output_dir, f'chain_{basename}_{i}.npy') for i in range(nchains)]
        sampler = EmceeSampler(likelihood, chains=nchains, nwalkers=4 * len(likelihood.varied_params), seed=42, save_fn=save_fn)
        chains = sampler.run(min_iterations=200, max_iterations=100000, check={'max_eigen_gr': 0.005})
        from desilike.samples import Chain
        chain = Chain.concatenate([
            Chain.load(f).remove_burnin(0.5)[::10] for f in save_fn
        ])
        #print(chain.to_stats(tablefmt='pretty'))
    
        result_path = os.path.join(output_dir, f'chain_{basename}.txt')
        with open(result_path, 'w') as f:
            f.write(chain.to_stats(tablefmt='pretty'))
    
        result_tex_path = os.path.join(output_dir, f'chain_{basename}.tex')
        with open(result_tex_path, 'w') as f:
            f.write(chain.to_stats(tablefmt='latex'))
    
        print(f"Salvati: {result_path}, {result_tex_path}")
        
        from desilike.samples import plotting
        plotting.plot_triangle(chain, fn=os.path.join(output_dir, f'chain_{basename}_triangle.png'))
    
        print(f"Finito sampling per {basename}")

#def run_all_fits(pk_dir, cov_dir, output_dir, is_postrecon=False):
#    os.makedirs(output_dir, exist_ok=True)
#
#    pk_files = sorted(glob(os.path.join(pk_dir, 'pkpoles_BGS_BRIGHT-20.2_*.txt')))
#    cov_files = sorted(glob(os.path.join(cov_dir, 'cov_gaussian_*.txt')))
#
#    for pk in pk_files:
#        basename = standardize_basename(pk)
#        matching_covs = [c for c in cov_files if basename in c]
#        if not matching_covs:
#            print(f"Cov non trovata per {basename}")
#            continue
#        try:
#            fit_pk_cov(pk, matching_covs[0], output_dir, is_postrecon=is_postrecon)
#        except Exception as e:
#            print(f"Errore su {basename}: {e}")
#
## Esegui per prerecon
#run_all_fits(
#    pk_dir='/global/cfs/cdirs/desi/survey/catalogs/DA2/analysis/loa-v1/LSScats/v1.1/BAO/unblinded/desipipe/2pt/pk',
#    cov_dir='/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/DR2/LSS/loa-v1/LSScats/v1.1/desipipe/cov_2pt/thecov/v1.1/prerecon/Uend',
#    output_dir='fit_output_prerecon',
#    is_postrecon=False
#)
#
## Esegui per postrecon
#run_all_fits(
#    pk_dir='/global/cfs/cdirs/desi/survey/catalogs/DA2/analysis/loa-v1/LSScats/v1.1/BAO/unblinded/desipipe/2pt/recon_sm15_IFFT_recsym_z0.8-1.1/pk',
#    cov_dir='/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/DR2/LSS/loa-v1/LSScats/v1.1/desipipe/cov_2pt/thecov/v1.1/postrecon/Uend',
#    output_dir='fit_output_postrecon',
#    is_postrecon=True
#)

def run_all_fits(pk_cov_map, output_dir, is_postrecon=False):
    os.makedirs(output_dir, exist_ok=True)

    for pk_glob, (cov_dir, wm_dir) in pk_cov_map.items():
        print(f"\n>>> Analizzo pattern:\nPK  = {pk_glob}\nCOV = {cov_dir}\nWM  = {wm_dir}")
        
        pk_files = sorted(glob(pk_glob))
        print(f"Trovati {len(pk_files)} file pk.")
        
        pk_files = [f for f in pk_files if 'SYS1_FKP' not in f]
        print(f"File pk dopo filtro SYS1_FKP: {len(pk_files)}")
    
        cov_files = sorted(glob(os.path.join(cov_dir, 'cov_gaussian_*.txt')))
        print(f"Trovati {len(cov_files)} file cov.")

        wm_files = sorted(glob(os.path.join(wm_dir, 'wmatrix_smooth_*.npy')))
        print(f"Trovati {len(wm_files)} file wm.")
    
        for pk in pk_files:
            basename = standardize_basename(pk)
            #print(f"\nProcessing: {basename}")

            #matching_covs = [c for c in cov_files if basename in c]
            #if not matching_covs:
            #    print(f"⚠️ Cov non trovata per {basename}")
            #    continue

            #wm_basename = re.sub(r'_z(\d\.\d+)_(\d\.\d+)', r'_z\1-\2', basename)
            #matching_wm = [c for c in wm_files if wm_basename in c]
            #if not matching_wm:
            #    print(f"⚠️ Window matrix non trovata per {basename}")
            #    continue
            print(f"\nProcessing: {pk_files}")
            matching_covs, matching_wm = find_matching_files(pk, cov_files, wm_files)

            if not matching_covs:
                print(f"⚠️ Cov non trovata per {pk_files}")
                continue
            if not matching_wm:
                print(f"⚠️ Window matrix non trovata per {pk_files}")
                continue
                
            try:
                fit_pk_cov(pk, matching_covs[0], matching_wm[0], output_dir, is_postrecon=is_postrecon, mode='profile')
            except Exception as e:
                print(f"❌ Errore su {basename}: {e}")



pk_cov_map_prerecon = {
    '/global/cfs/cdirs/desi/users/ndeiosso/archive/LSS_old/scripts/prerecon_PIP/pk/pkpoles_BGS_*-20.7_*lin5_P07000.0.txt':
    ('/global/cfs/cdirs/desi/users/ndeiosso/archive/BGS_ANY_DR2/DR2/LSS/loa-v1/LSScats/v1.1/desipipe/cov_2pt/thecov/v1.1/prerecon_PIP',
    '/global/cfs/cdirs/desi/users/ndeiosso/archive/LSS_old/scripts/prerecon_PIP/pk'),
#    '/global/cfs/cdirs/desi/users/ndeiosso/archive/LSS_old/scripts/prerecon/pk/pkpoles_BGS_BRIGHT-21.35_*lin5_P07000.0.txt':
#        ('/global/cfs/cdirs/desi/users/ndeiosso/archive/BGS_ANY_DR2/DR2/LSS/loa-v1/LSScats/v1.1/desipipe/cov_2pt/thecov/v1.1/prerecon/prova_ashley',
#        '/global/cfs/cdirs/desi/users/ndeiosso/archive/LSS_old/scripts/prerecon/2pt/pk') 
}

run_all_fits(pk_cov_map_prerecon, 'fit_output_prerecon_2D', is_postrecon=False)
#

pk_cov_map_postrecon = {
    '/global/cfs/cdirs/desi/users/ndeiosso/archive/LSS_old/scripts/postrecon_PIP/pk/pkpoles_BGS_*-20.7_*lin5_P07000.0.txt':
        ('/global/cfs/cdirs/desi/users/ndeiosso/archive/BGS_ANY_DR2/DR2/LSS/loa-v1/LSScats/v1.1/desipipe/cov_2pt/thecov/v1.1/postrecon_PIP',
        '/global/cfs/cdirs/desi/users/ndeiosso/archive/LSS_old/scripts/postrecon_PIP/pk'),
#    '/global/cfs/cdirs/desi/users/ndeiosso/archive/LSS_old/scripts/postrecon/pk/pkpoles_BGS_BRIGHT-21.35_*lin5_P07000.0.txt':
#        ('/global/cfs/cdirs/desi/users/ndeiosso/archive/BGS_ANY_DR2/DR2/LSS/loa-v1/LSScats/v1.1/desipipe/cov_2pt/thecov/v1.1/postrecon_2',
#        '/global/cfs/cdirs/desi/users/ndeiosso/archive/LSS_old/scripts/postrecon/pk')
}


run_all_fits(pk_cov_map_postrecon, 'fit_output_postrecon_2D', is_postrecon=True)
#
#