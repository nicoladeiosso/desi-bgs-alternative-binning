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
cat5_path = "/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v1.1/BAO/blinded/BGS_BRIGHT_clustering.dat.fits"

# --- Load power spectrum files ---
def load_pk(filepath):
    """Load power spectrum data and return k, P0, P2, P4"""
    data = np.loadtxt(filepath)
    k = data[:, 0]
    P0 = data[:, 1]
    P2 = data[:, 2] if data.shape[1] > 2 else None
    P4 = data[:, 3] if data.shape[1] > 3 else None
    return k, P0, P2, P4

# Load all Pk files
pk_files = {
    'postrecon_ANY_GComb': '/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/clean_pk/postrecon_ANY/pkpoles_BGS_ANY_GCcomb_z0.1-0.4_4000.0.txt',
    'postrecon_ANY_NGC': '/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/clean_pk/postrecon_ANY/pkpoles_BGS_ANY_NGC_z0.1-0.4_4000.0.txt',
    'postrecon_ANY_SGC': '/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/clean_pk/postrecon_ANY/pkpoles_BGS_ANY_SGC_z0.1-0.4_4000.0.txt',
    'prerecon_ANY_GComb': '/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/clean_pk/prerecon_ANY/pkpoles_BGS_ANY_GCcomb_z0.1-0.4_4000.0.txt',
    'prerecon_ANY_NGC': '/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/clean_pk/prerecon_ANY/pkpoles_BGS_ANY_NGC_z0.1-0.4_4000.0.txt',
    'prerecon_ANY_SGC': '/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/clean_pk/prerecon_ANY/pkpoles_BGS_ANY_SGC_z0.1-0.4_4000.0.txt',
    'postrecon_2135_GComb': '/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/clean_pk/postrecon_21.35/pkpoles_BGS_BRIGHT-21.35_GCcomb_z0.1-0.4_4000.txt',
    'postrecon_2135_NGC': '/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/clean_pk/postrecon_21.35/pkpoles_BGS_BRIGHT-21.35_NGC_z0.1-0.4_4000.txt',
    'postrecon_2135_SGC': '/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/clean_pk/postrecon_21.35/pkpoles_BGS_BRIGHT-21.35_SGC_z0.1-0.4_4000.txt',
    'prerecon_2135_GComb': '/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/clean_pk/prerecon_21.35/pkpoles_BGS_BRIGHT-21.35_GCcomb_z0.1-0.4_4000.txt',
    'prerecon_2135_NGC': '/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/clean_pk/prerecon_21.35/pkpoles_BGS_BRIGHT-21.35_NGC_z0.1-0.4_4000.txt',
    'prerecon_2135_SGC': '/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/clean_pk/prerecon_21.35/pkpoles_BGS_BRIGHT-21.35_SGC_z0.1-0.4_4000.txt',
    'postrecon_2135_GCcomb_0.25-0.4': '/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/clean_pk/postrecon_21.35/pkpoles_BGS_BRIGHT-21.35_GCcomb_z0.25-0.4_4000.txt',
    'postrecon_20.2_GCcomb_0.1-0.4': '/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/clean_pk/postrecon_20.2/pkpoles_BGS_BRIGHT-20.2_GCcomb_z0.1-0.4_4000.txt',
    'postrecon_20.2_GCcomb_0.1-0.25': '/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/clean_pk/postrecon_20.2/pkpoles_BGS_BRIGHT-20.2_GCcomb_z0.1-0.25_4000.txt',
    'prerecon_2135_GCcomb_0.25-0.4': '/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/clean_pk/prerecon_21.35/pkpoles_BGS_BRIGHT-21.35_GCcomb_z0.25-0.4_4000.txt',
    'prerecon_20.2_GCcomb_0.1-0.4': '/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/clean_pk/prerecon_20.2/pkpoles_BGS_BRIGHT-20.2_GCcomb_z0.1-0.4_4000.txt',
    'prerecon_20.2_GCcomb_0.1-0.25': '/pscratch/sd/n/ndeiosso/BGS_ANY_DR2/clean_pk/prerecon_20.2/pkpoles_BGS_BRIGHT-20.2_GCcomb_z0.1-0.25_4000.txt',    
}

pk_data = {}
for name, path in pk_files.items():
    pk_data[name] = load_pk(path)

# --- Function to compute comoving volume in a redshift shell ---
def compute_volume_shell(z_low, z_high, sky_area_deg2):
    """Compute comoving volume of a redshift shell in (Mpc/h)^3 using DESI cosmology."""
    sky_area_sterad = sky_area_deg2 * (np.pi/180)**2  # deg² → steradians
    f_sky = sky_area_sterad / (4 * np.pi)  # Fraction of full sky
    Dc_low = cosmo.comoving_radial_distance(z_low)  # Already in Mpc/h
    Dc_high = cosmo.comoving_radial_distance(z_high)
    volume = (4/3) * np.pi * (Dc_high**3 - Dc_low**3) * f_sky
    return volume

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
z_min, z_max = 0.1, 0.25  # Adjust for BGS
z_bins = np.linspace(z_min, z_max, 26)  # 10 bins (adjust as needed)
sky_area_deg2 = 12355  # DESI sky area (adjust!)

nz1, P_shot1 = compute_nz_and_shotnoise(cat1_path, z_bins, sky_area_deg2)
nz2, P_shot2 = compute_nz_and_shotnoise(cat2_path, z_bins, sky_area_deg2)
nz3, P_shot3 = compute_nz_and_shotnoise(cat3_path, z_bins, sky_area_deg2)
nz4, P_shot4 = compute_nz_and_shotnoise(cat4_path, z_bins, sky_area_deg2)
nz5, P_shot5 = compute_nz_and_shotnoise(cat5_path, z_bins, sky_area_deg2)

# --- Plot n(z) ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
z_centers = z_bins[:-1] + 0.5 * np.diff(z_bins)
#plt.plot(z_centers, nz1, label='ANY')
#plt.plot(z_centers, nz2, label='BRIGHT-21.5')
plt.plot(z_centers, nz3, label='BRIGHT-21.35')
plt.plot(z_centers, nz4, label='BRIGHT-20.2')
#plt.plot(z_centers, nz5, label='BRIGHT')
plt.xlabel('Redshift (z)')
plt.ylabel('$n(z)$ [$10^4$ (h/Mpc)$^3$]')
plt.legend()
plt.grid(True)

# --- Plot shot noise ---
plt.subplot(1, 2, 2)
#plt.plot(z_centers, P_shot1, label='ANY')
plt.plot(z_centers, P_shot3, label='BRIGHT-21.35')
plt.plot(z_centers, P_shot4, label='BRIGHT-20.2')
plt.xlabel('Redshift (z)')
plt.ylabel('$P_{\\mathrm{shot}}(z)$ [(Mpc/h)$^3$]')
plt.yscale('log')  # Shot noise can span large ranges
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('nz_and_shotnoise_z0.1-0.25z.png')
plt.show()

# --- New: Compare shot noise with P0 ---
plt.figure(figsize=(12, 6))

# Plot P0 for some representative samples
for name in ['prerecon_20.2_GCcomb_0.1-0.25', 'postrecon_20.2_GCcomb_0.1-0.25']:
    k, P0, _, _ = pk_data[name]
    label = name.replace('_GComb', '').replace('_', ' ')
    plt.plot(k, P0, label=label)

# Plot average shot noise for comparison
#avg_P_shot1 = np.mean(P_shot1[(z_centers >= 0.1) & (z_centers <= 0.4)])
#avg_P_shot3 = np.mean(P_shot3[(z_centers >= 0.1) & (z_centers <= 0.25)])
avg_P_shot4 = np.mean(P_shot4[(z_centers >= 0.1) & (z_centers <= 0.25)])
#plt.axhline(avg_P_shot1, color='k', linestyle='--', label='Avg shot noise (ANY)')
#plt.axhline(avg_P_shot3, color='r', linestyle='--', label='Avg shot noise (BRIGHT-21.35)')
plt.axhline(avg_P_shot4, color='b', linestyle='--', label='Avg shot noise (BRIGHT-20.2)')

plt.xlabel('k [h/Mpc]')
plt.ylabel('P0(k) [(Mpc/h)$^3$]')
plt.title('Power Spectrum vs Shot Noise')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('P0_vs_shotnoise_z1-0.25.png')
plt.show()

# Print some statistics
print("\nShot noise statistics:")
#print(f"ANY sample - Average shot noise: {avg_P_shot1:.2f} (Mpc/h)^3")
#print(f"BRIGHT-21.35 sample - Average shot noise: {avg_P_shot3:.2f} (Mpc/h)^3")

# Check if shot noise dominates
for name in ['postrecon_20.2_GCcomb_0.1-0.25']:
    k, P0, _, _ = pk_data[name]
    avg_P0 = np.mean(P0[(k >= 0.01) & (k <= 0.2)])  # Average over BAO scales
    if 'ANY' in name:
        ratio = avg_P_shot3 / avg_P0
    else:
        ratio = avg_P_shot4 / avg_P0
    print(f"\nFor {name}:")
    print(f"  Average P0 at BAO scales: {avg_P0:.2f} (Mpc/h)^3")
    print(f"  Shot noise to P0 ratio: {ratio:.2f}")
    if ratio > 0.5:
        print("  WARNING: Shot noise is significant (>50% of P0)")
    if ratio > 1:
        print("  WARNING: Shot noise dominates over P0!")