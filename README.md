# desi-pimped-bgs
Repository for scripts, products and fits for the DR2 BGS analysis

Scipts probably needs some improvements, especially for the production of new catalogs. For now, these are made to produce only un cutted catalog at time (for magnitude cut and region) and then, separately, the other code recontruct catalogs for one cut at time but for all the three regions.
Also, nomenclature needs to be updated 

### Scripts and usage ###

# nz.py
Compute the comoving number density n(z) and shotnoise for each catalog (path need to be inserted in the script), debugged and cross-checked with mocks (Bernardita) and data.
Use: python nz.py

# nz_pimped.py
Same as nz.py but include Pk value for the evaluation of the shot noise contribution respect to the measurements
Use: python nz_pimped.py

# mag_cut.py and and mag_cut_rand.py
Need to be merged in a single script, it remains like that from the beginning. Cross checked respect to the original mag_cut catalog for the fiducial analysis. 
Use: so flags included as for the nz scripts, just launch them with python

# recon.py 
Reconstruction of the catalogs. It seems to work properly but maybe can be updated to be more flexible (paths and output need to be put in the script, no flags) 
Use: python recon.py

# pypower_xyz_b.py
Compute the power spectra and window function for a given set of data and random. About the nomencalture: xyz because is the version that uses xyz coordinates, converted form RA, DEC and Z. b because at the beginning it was made just for bright catas, but now the name refers to the new and proper version for the computation of the window function. Looking at the output, the pk produced are almost identical to the fiducial ones, with some minor differences at very large scales, but there is the problem related to the use of angular weigts, that maybe could help with the pk shape and than the covariance generation. Indeed, this in not so efficient in parallelization. 
For comparison, there is also the pypower_xyz.py with the original settings for window matrix computation (with no weights and old routind for window)
Use: there is a small difference between NGC/SGC and GCcomb. Example for BRIGHT cat:

python pypower_xyz_b.py \
  --data /global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v1.1/nonKP/BGS_BRIGHT-20.2_NGC_clustering.dat.fits \
  --rand $(for i in {0..17}; do echo "/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v1.1/nonKP/BGS_BRIGHT-20.2_NGC_${i}_clustering.ran.fits"; done) \
  --mag 20.2 --region NGC --zmin 0.1 --zmax 0.25

python pypower_xyz_b.py \
  --data /global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v1.1/nonKP/BGS_BRIGHT-20.2_NGC_clustering.dat.fits \
         /global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v1.1/nonKP/BGS_BRIGHT-20.2_SGC_clustering.dat.fits \
  --rand $(for i in {0..17}; do echo "/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v1.1/nonKP/BGS_BRIGHT-20.2_NGC_${i}_clustering.ran.fits"; done) \
  --rand $(for i in {0..17}; do echo "/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v1.1/nonKP/BGS_BRIGHT-20.2_SGC_${i}_clustering.ran.fits"; done) \
  --mag 20.2 --region GCcomb --zmin 0.1 --zmax 0.25

For the number of rand, you can also use the flag --nrand and define the path like --rand /global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v1.1/nonKP/BGS_BRIGHT-20.2_NGC_*_clustering.ran.fits
For the output dir, you can use the flag --output-dir

# pk_cleaner.py
Simple script for pk file cleaning from header and complex part, useful to have simple file to plot (for the plot, a command is implemented in pypower_xyz_b.py, as for the rebinning). Needed for the problems I'm having with notebooks.
Use: python3 pk_cleaner.py --tracer BGS_BRIGHT-20.2 --region GCcomb --boxsize 4000.0 --zmin 0.1 --zmax 0.25

# generate_cov_pre/post.py
Script for covariance generation. Again, the pre/post differentiation if because of laziness, and also for a matter of path definition in the validation task for the paper, where the structure was more complex. The main difference between the two is that they charge a different configuration file (the config_covariance.yaml) that define paths for pk, cats, kmax, klim, dk and output paths. The code is the same used for the fiducial analysis, so it works properly. 
Use: srun -c 256 python3 generate_cov_pre.py --tracer BGS_BRIGHT-20.2 --region GCcomb --boxsize boxsize4000.0 --zmin 0.1 --zmax 0.25

# fit_pimped_bao.py and fit_alll.py
BAO fit. One is for a particular case, the other one for all the files. It works properly, cross-checked with the fiducial analysis. The pimped one generates a txt and a tex file with the best fit values. All generate also latex tabs for every case (Post recon GCcomb, for example) with best fit results and chi2 for every magcut case, and plots with q_iso values.
Use: python fit_pimped.py

Catalog: prerecon are in ANY_magcut_catalog, postrecon are in postrecon
