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
                kmid = float(parts[1])
                P0 = parse_real(parts[3])
                P2 = parse_real(parts[4])
                P4 = parse_real(parts[5])
                data.append((kmid, P0, P2, P4))
            except ValueError:
                continue
    return np.array(data)

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

    # Rimuovi tutto quello che viene dopo il redshift, inclusi suffix tecnici
    # esempio: BGS_ANY_GCcomb_z0.1-0.4_default_FKP...txt → BGS_ANY_GCcomb_z0.1_0.4
    # Prima fai la sostituzione z0.X-Y → z0.X_Y
    basename = re.sub(r'z(\d+\.\d+)-(\d+\.\d+)', r'z\1_\2', basename)

    # Poi tronchi tutto dopo .txt o underscore dopo lo z-range
    basename = re.sub(r'(z\d+\.\d+_\d+\.\d+).*', r'\1', basename)

    return basename


    
def fit_pk_cov(pk_file, cov_file, output_dir, is_postrecon=False, kmin=0.02, kmax=0.3):
    basename = os.path.splitext(os.path.basename(pk_file))[0]
    print(f"\n>>> Fit per:\n  PK  = {pk_file}\n  COV = {cov_file}")
    # Lettura dati
    data_array = read_power_spectrum_data(pk_file)
    k, p0, p2 = data_array[:, 0], data_array[:, 1], data_array[:, 2]

    mask = (k >= 0.02) & (k <= 0.3)
    k_selected = k[mask]
    p0_selected = p0[mask]
    p2_selected = p2[mask]
    
    ell_to_include = [0]  #Poles to include
    data_dict = {
        0: p0_selected,
        2: p2_selected
    }

    data = np.concatenate([data_dict[ell] for ell in ell_to_include])

    cov = np.loadtxt(cov_file)
    n_k_total = len(k)  # Esempio: 60 punti totali
    
    # ℓ to use
    ells_to_use = [0]
    all_ells = [0, 2, 4]  #Cov matrix order
    mask = (k >= 0.02) & (k <= 0.3)
    k_indices = np.where(mask)[0]
    n_k_selected = len(k_indices)
    
    indices = []
    for i, ell in enumerate(all_ells):
        if ell in ells_to_use:
            indices.extend([i * n_k_total + idx for idx in k_indices])
    
    cov = cov[np.ix_(indices, indices)]

    # Setup modello
    z = 0.1  # hardcoded, da generalizzare in futuro
    template = BAOPowerSpectrumTemplate(z=z, fiducial='DESI', apmode='qisoqap')
    theory = DampedBAOWigglesTracerPowerSpectrumMultipoles(template=template, ells=ell_to_include, broadband='pcs')
    observable = TracerPowerSpectrumMultipolesObservable(data=data, covariance=cov, k=k_selected, ells=ell_to_include, theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])

    print(theory.params.names())
    
    # Parametri
    params = likelihood.runtime_info.pipeline.params
    params['qap'].update(value=1., fixed=True)
    params['dbeta'].update(value=1., fixed=True)
    params['sigmapar'].update(fixed=False)
    params['sigmaper'].update(fixed=False)

    for name in params.basenames():
        if name.startswith('al2_'): params[name].update(value=0., fixed=True)
        if name.startswith('al0_'): params[name].update(prior={'dist': 'norm', 'loc': 0., 'scale': 1e4})

    params['b1'].update(prior={'limits': [0.2, 4.]})
    params['qiso'].update(prior={'limits': [0.8, 1.2]})
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

    solved_params = likelihood.all_params.select(solved=True)
    print("Marginalized Params(solved):")
    print(sorted(p.basename for p in solved_params))
    
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

def run_all_fits(pk_dir, cov_dir, output_dir, is_postrecon=False):
    os.makedirs(output_dir, exist_ok=True)

    pk_files = sorted(glob(os.path.join(pk_dir, 'pkpoles_*.txt')))
    cov_files = sorted(glob(os.path.join(cov_dir, 'cov_gaussian_*.txt')))

    for pk in pk_files:
        basename = standardize_basename(pk)
        matching_covs = [c for c in cov_files if basename in c]
        if not matching_covs:
            print(f"Cov non trovata per {basename}")
            continue
        try:
            fit_pk_cov(pk, matching_covs[0], output_dir, is_postrecon=is_postrecon)
        except Exception as e:
            print(f"Errore su {basename}: {e}")

# Esegui per prerecon
run_all_fits(
    pk_dir='/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/DR2/LSS/loa-v1/LSScats/v1.1/desipipe/2pt/pk',
    cov_dir='/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/DR2/LSS/loa-v1/LSScats/v1.1/desipipe/cov_2pt/thecov/v1.1/prerecon',
    output_dir='fit_output_prerecon',
    is_postrecon=False
)

# Esegui per postrecon
run_all_fits(
    pk_dir='/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/DR2/LSS/loa-v1/LSScats/v1.1/desipipe/2pt/recon_sm15_IFFT_recsym/pk',
    cov_dir='/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/DR2/LSS/loa-v1/LSScats/v1.1/desipipe/cov_2pt/thecov/v1.1/postrecon',
    output_dir='fit_output_postrecon',
    is_postrecon=True
)


