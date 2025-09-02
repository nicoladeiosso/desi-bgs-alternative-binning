import numpy as np
from astropy.io import fits

def compute_zeff(filename, zmin=0.0, zmax=1.0,
                 zcol="Z", wcol="WEIGHT_FKP"):
    """
    Compute effective redshift for a catalog in FITS format.

    Parameters
    ----------
    filename : str
        Path to FITS catalog.
    zmin, zmax : float
        Redshift range to select galaxies.
    zcol : str
        Column name for redshift.
    wcol : str
        Column name for weights (default: FKP).

    Returns
    -------
    zeff : float
        Effective redshift.
    ngal : int
        Number of galaxies used.
    """
    # Load table (assume data is in HDU 1)
    with fits.open(filename, memmap=True) as hdul:
        data = hdul[1].data

    z = np.array(data['Z'])
    w = np.array(data['WEIGHT_FKP'])*np.array(data['WEIGHT_SYS'])*np.array(data['WEIGHT_COMP'])

    # Apply redshift cut
    mask = (z >= zmin) & (z < zmax)
    z_sel, w_sel = z[mask], w[mask]

    # Compute zeff
    zeff = np.sum(w_sel**2 * z_sel) / np.sum(w_sel**2)
    return zeff, mask.sum()


if __name__ == "__main__":
    # Example usage
    catalog_file = "/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v1.1/nonKP/BGS_BRIGHT-21.35_clustering.dat.fits"
    zmin, zmax = 0.1, 0.4

    zeff, ngal = compute_zeff(catalog_file, zmin, zmax)
    print(f"Effective redshift in {zmin:.2f} < z < {zmax:.2f}: {zeff:.4f} (Ngal={ngal})")
