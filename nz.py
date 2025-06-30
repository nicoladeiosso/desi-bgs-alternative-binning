import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from cosmoprimo.fiducial import DESI

# Initialize DESI cosmology
cosmo = DESI()
h = 100*cosmo.h  # Hubble parameter (H0 = 100h km/s/Mpc)

# --- File paths ---
cat1_path = "/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v1.1/nonKP/BGS_ANY_clustering.dat.fits"
cat2_path = "/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v1.1/BAO/blinded/BGS_BRIGHT-21.5_clustering.dat.fits"  
cat3_path = "/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v1.1/BAO/blinded/BGS_BRIGHT-21.35_clustering.dat.fits"
cat4_path = "/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v1.1/BAO/blinded/BGS_BRIGHT-20.2_clustering.dat.fits"  
cat5_path = "/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/ANY_magcut_catalog/BGS_ANY-21.35_clustering.dat.fits"
cat6_path = "/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/ANY_magcut_catalog/BGS_ANY-20.2_clustering.dat.fits"  
cat7_path = "/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/ANY_magcut_catalog/BGS_ANY-21.5_clustering.dat.fits"
cat11_path = "/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v1.1/nonKP/BGS_BRIGHT_clustering.dat.fits"
cat8_path = "/global/cfs/cdirs/desi/survey/catalogs/DA2/analysis/loa-v1/LSScats/v1.1/BAO/blinded/desipipe/2pt/recon_sm15_IFFT_recsym/BGS_BRIGHT-21.5_NGC_clustering.dat.fits"  
cat9_path = "/global/cfs/cdirs/desi/survey/catalogs/DA2/analysis/loa-v1/LSScats/v1.1/BAO/blinded/desipipe/2pt/recon_sm15_IFFT_recsym/BGS_BRIGHT-21.35_NGC_clustering.dat.fits"
cat10_path = "/global/cfs/cdirs/desi/survey/catalogs/DA2/analysis/loa-v1/LSScats/v1.1/BAO/blinded/desipipe/2pt/recon_sm15_IFFT_recsym/BGS_BRIGHT-20.2_NGC_clustering.dat.fits"  
#cat11_path = "/global/cfs/cdirs/desi/survey/catalogs/DA2/analysis/loa-v1/LSScats/v1.1/BAO/blinded/desipipe/2pt/recon_sm15_IFFT_recsym/BGS_BRIGHT_NGC_clustering.dat.fits"

# --- Function to compute comoving volume in a redshift shell ---
def compute_volume_shell(z_low, z_high, sky_area_deg2):
    """Compute comoving volume of a redshift shell in (Mpc/h)^3 using DESI cosmology."""
    sky_area_sterad = sky_area_deg2 * (np.pi/180)**2  # deg² → steradians
    f_sky = sky_area_sterad / (4 * np.pi)  # Fraction of full sky
    # print(sky_area_sterad)
    # Comoving distances (in Mpc/h) using DESI cosmology
    Dc_low = cosmo.comoving_radial_distance(z_low)  # Already in Mpc/h
    Dc_high = cosmo.comoving_radial_distance(z_high)
    
    # Volume of the shell in (Mpc/h)^3
    vol = (4/3) * np.pi * (Dc_high**3 - Dc_low**3)
    volume = (4/3) * np.pi * (Dc_high**3 - Dc_low**3) * f_sky

    #print(vol)
    #print(volume)
    
    return volume

# --- Function to compute n(z) in redshift bins ---
# --- Function to compute n(z) and shot noise in redshift bins ---
def compute_nz_and_shotnoise(catalog_path, z_bins, sky_area_deg2):
    """Compute number density and shot noise in redshift bins.
    Returns:
        nz: Number density [10^4 (h/Mpc)^3]
        P_shot: Shot noise [(Mpc/h)^3]
    """
    with fits.open(catalog_path) as hdul:
        data = Table(hdul[1].data)
    
    z = data['Z']
    weights = data['WEIGHT']  # Use WEIGHT_COMP/SYS if needed
    
    nz, P_shot = [], []
    for i in range(len(z_bins) - 1):
        z_low, z_high = z_bins[i], z_bins[i+1]
        mask = (z >= z_low) & (z < z_high)
        N = np.sum(weights[mask])
        V = compute_volume_shell(z_low, z_high, sky_area_deg2)
        
        # Number density [10^4 (h/Mpc)^3]
        nz_val = (N / V) * 1e4 if V > 0 else 0.0
        nz.append(nz_val)
        
        # Shot noise [(Mpc/h)^3] = 1/n_c (where n_c is in (h/Mpc)^3)
        P_shot_val = (1.0 / (N / V)) if (N > 0 and V > 0) else np.inf
        P_shot.append(P_shot_val)
    
    return np.array(nz), np.array(P_shot)

# --- Define redshift bins and sky area ---
z_min, z_max = 0.0001, 0.5  # Adjust for BGS
z_bins = np.linspace(z_min, z_max, 26)  # 10 bins (adjust as needed)
sky_area_deg2 = 12355*0.826  # DESI sky area (adjusted for completeness)

nz1, P_shot1 = compute_nz_and_shotnoise(cat1_path, z_bins, sky_area_deg2)
nz2, P_shot2 = compute_nz_and_shotnoise(cat2_path, z_bins, sky_area_deg2)
nz3, P_shot3 = compute_nz_and_shotnoise(cat3_path, z_bins, sky_area_deg2)
nz4, P_shot4 = compute_nz_and_shotnoise(cat4_path, z_bins, sky_area_deg2)
nz5, P_shot5 = compute_nz_and_shotnoise(cat5_path, z_bins, sky_area_deg2)
nz6, P_shot6 = compute_nz_and_shotnoise(cat6_path, z_bins, sky_area_deg2)
nz7, P_shot7 = compute_nz_and_shotnoise(cat7_path, z_bins, sky_area_deg2)
nz8, P_shot8 = compute_nz_and_shotnoise(cat8_path, z_bins, sky_area_deg2)
nz9, P_shot9 = compute_nz_and_shotnoise(cat9_path, z_bins, sky_area_deg2)
nz10, P_shot10 = compute_nz_and_shotnoise(cat10_path, z_bins, sky_area_deg2)
nz11, P_shot11 = compute_nz_and_shotnoise(cat11_path, z_bins, sky_area_deg2)

# --- Plot n(z) ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
z_centers = z_bins[:-1] + 0.5 * np.diff(z_bins)
plt.plot(z_centers, nz1, label='ANY')
plt.plot(z_centers, nz2, color = 'g', label='BRIGHT-21.5')
plt.plot(z_centers, nz3, color = 'y', label='BRIGHT-21.35')
plt.plot(z_centers, nz6, color = 'b', label='ANY-20.2')
plt.plot(z_centers, nz4, color = 'r', label='BRIGHT-20.2')
plt.plot(z_centers, nz7, color = 'c', label='ANY-21.5')
plt.plot(z_centers, nz5, color = 'c', label='ANY-21.35')
plt.plot(z_centers, nz11, label='BRIGHT')
#plt.plot(z_centers, nz8, label='recon_BRIGHT-21.5')
#plt.plot(z_centers, nz9, color = 'c', label='recon_BRIGHT-21.35')
#plt.plot(z_centers, nz10, color = 'grey', label='recon_BRIGHT-20.2')
#plt.plot(z_centers, nz11, color = 'purple', label='recon_BRIGHT')


plt.ylim(0, 150)
plt.xlabel('Redshift (z)')
plt.ylabel('$n(z)$ [$10^4$ (h/Mpc)$^3$]')
plt.legend()
plt.grid(True)

# --- Plot shot noise ---
plt.subplot(1, 2, 2)
plt.plot(z_centers, P_shot1, label='ANY')
plt.plot(z_centers, P_shot2, color = 'g', label='BRIGHT-21.5')
plt.plot(z_centers, P_shot3, color = 'y', label='BRIGHT-21.35')
plt.plot(z_centers, P_shot6, color = 'b', label='ANY-20.2')
plt.plot(z_centers, P_shot4, color = 'r', label='BRIGHT-20.2')
plt.plot(z_centers, P_shot7, color = 'c', label='ANY-21.5')
plt.plot(z_centers, P_shot11, label='BRIGHT')
plt.xlabel('Redshift (z)')
plt.ylabel('$P_{\\mathrm{shot}}(z)$ [(Mpc/h)$^3$]')
#plt.xscale('log')
plt.yscale('log')  # Shot noise can span large ranges
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('nz_and_shotnoise_new.png')
plt.show()