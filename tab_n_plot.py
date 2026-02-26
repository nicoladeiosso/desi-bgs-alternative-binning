import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

# --- Nuovo: parser generale per l'header del filename ---
_ZRANGE_RE = re.compile(
    r'^pkpoles_(?P<tracer>.+?)_'
    r'(?:(?:z(?P<zminA>\d+(?:\.\d+)?)-(?P<zmaxA>\d+(?:\.\d+)?))|(?P<zminB>\d+(?:\.\d+)?)_(?P<zmaxB>\d+(?:\.\d+)?))'
    r'_.*?_fit_results\.txt$'
)

def parse_header_from_filename(filename):
    """
    Supporta:
      pkpoles_BGS_BRIGHT-21.35_SGC_z0.25-0.4_..._fit_results.txt
      pkpoles_BGS_BRIGHT-21.35_SGC_0.0_0.5_..._fit_results.txt
    Ritorna: dataset, region, z_range_norm ('zmin-zmax'), zmin(float), zmax(float)
    """
    base = os.path.basename(filename)
    m = _ZRANGE_RE.match(base)
    if not m:
        raise ValueError(f"Filename non riconosciuto: {base}")

    tracer = m.group('tracer')
    zmin = m.group('zminA') or m.group('zminB')
    zmax = m.group('zmaxA') or m.group('zmaxB')

    # dataset e regione dall'ultima underscore del tracer
    tparts = tracer.split('_')
    region = tparts[-1] if len(tparts) >= 2 else "UNK"
    dataset = '_'.join(tparts[:-1]) if len(tparts) >= 2 else tracer

    z_range_norm = f"{zmin}-{zmax}"  # normalizzo sempre con '-'
    return dataset, region, z_range_norm, float(zmin), float(zmax)


def parse_files():
    files = glob.glob('pkpoles_*_fit_results.txt')
    data = []

    for filepath in files:
        try:
            dataset, region, z_range, zmin, zmax = parse_header_from_filename(filepath)
        except Exception as e:
            print(f"Skip file (pattern non riconosciuto): {filepath} -> {e}")
            continue

        with open(filepath, 'r') as f:
            content = f.read()

        params = {
            'dataset': dataset,
            'region': region,
            'z_range': z_range,  # forma normalizzata 'zmin-zmax'
            'zmin': zmin,
            'zmax': zmax,
        }

        # Estrai chi2 totale, dof, chi2 ridotto dalla prima riga (come prima)
        chi2_match = re.search(r'chi2\s*/\s*\((\d+)\s*-\s*(\d+)\)\s*=\s*([\d.]+)\s*/\s*(\d+)\s*=\s*([\d.]+)', content)
        if chi2_match:
            n_obs = int(chi2_match.group(1))
            n_par = int(chi2_match.group(2))
            chi2_total = float(chi2_match.group(3))
            dof = int(chi2_match.group(4))
            chi2_red = float(chi2_match.group(5))
        else:
            chi2_total = dof = chi2_red = None

        params['chi2'] = chi2_total
        params['dof'] = dof
        params['chi2_red'] = chi2_red

        # Parametri principali (come prima)
        patterns = {
            'qiso': r'qiso\s*\|\s*True\s*\|\s*([\d.-]+)\s*\|\s*([\d.-]+)',
            'qap': r'qap\s*\|\s*True\s*\|\s*([\d.-]+)\s*\|\s*([\d.-]+)',
            'b1': r'b1\s*\|\s*True\s*\|\s*([\d.-]+)\s*\|\s*([\d.-]+)',
            'sigmapar': r'sigmapar\s*\|\s*True\s*\|\s*([\d.-]+)\s*\|\s*([\d.-]+)',
            'sigmaper': r'sigmaper\s*\|\s*True\s*\|\s*([\d.-]+)\s*\|\s*([\d.-]+)',
        }

        for param, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                params[param] = float(match.group(1))
                params[f'{param}_err'] = float(match.group(2))
            else:
                params[param] = None
                params[f'{param}_err'] = None

        data.append(params)

    return pd.DataFrame(data)



def create_latex_tables(df):
    # Crea una tabella LaTeX per ogni regione unica
    regions = df['region'].unique()

    for region in regions:
        region_df = df[df['region'] == region].copy()
        
        # Ordina per dataset e z_range per leggibilità
        region_df.sort_values(by=['dataset', 'z_range'], inplace=True)
        
        latex_table = r"""\begin{table}[ht]
\centering
\caption{Parameters for region """ + region + r"""}
\begin{tabular}{llcccc}
\hline
Dataset & $z$-range & $q_{\mathrm{iso}}$ $q_{\mathrm{ap}}$ & $b_1$ & $\sigma_{\parallel,\perp}$ & $\chi^2_\nu$ \\
\hline
"""

# Nel ciclo sulle righe
        for _, row in region_df.iterrows():
            qiso_str = f"${row['qiso']:.4f} \\pm {row['qiso_err']:.4f}$" if pd.notnull(row['qiso']) else "---"
            qap_str = f"${row['qap']:.4f} \\pm {row['qap_err']:.4f}$" if pd.notnull(row['qap']) else "---"
            b1_str = f"${row['b1']:.3f} \\pm {row['b1_err']:.3f}$" if pd.notnull(row['b1']) else "---"
            sigmas_str = (
                f"$({row['sigmapar']:.1f} \\pm {row['sigmapar_err']:.1f}, "
                f"{row['sigmaper']:.1f} \\pm {row['sigmaper_err']:.1f})$"
                if pd.notnull(row['sigmapar']) and pd.notnull(row['sigmaper']) else "---"
            )
            chi2_str = f"${row['chi2_red']:.2f}$" if pd.notnull(row['chi2_red']) else "---"
            latex_table += f"{row['dataset']} & {row['z_range']} & {qiso_str} & {qap_str} & {b1_str} & {sigmas_str} & {chi2_str} \\\\\n"


        latex_table += r"""\hline
\end{tabular}
\label{tab:""" + region.lower() + r"""}
\end{table}
"""

        with open(f'table_{region}.tex', 'w') as f:
            f.write(latex_table)
        print(f"Tabella LaTeX per la regione {region} salvata come 'table_{region}.tex'")


def create_whisker_plots(df):
    regions = df['region'].unique()

    for region in regions:
        region_df = df[df['region'] == region].copy()

        if region_df.empty:
            continue

        # Ordina per dataset e z_range
        region_df.sort_values(by=['dataset', 'z_range'], inplace=True)

        labels = [f"{row['dataset']}\n{row['z_range']}" for _, row in region_df.iterrows()]
        qiso_values = region_df['qiso'].values
        qiso_errors = region_df['qiso_err'].values

        plt.figure(figsize=(12, 6))
        plt.errorbar(range(len(region_df)), qiso_values, yerr=qiso_errors,
                     fmt='o', capsize=5, markersize=6, linestyle='')

        plt.xticks(range(len(region_df)), labels, rotation=45, ha='right')
        plt.ylabel('$q_{\\mathrm{iso}}$', fontsize=13)
        plt.title(f'$q_{{\\mathrm{{iso}}}}$ values for region {region}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(f'whisker_plot_{region}.png', dpi=300)
        #plt.savefig(f'whisker_plot_{region}.pdf')
        plt.close()
        print(f"Plot per {region} salvato.")


def main():
    # Analizza i file e crea il dataframe
    df = parse_files()
    
    if df.empty:
        print("Nessun file valido trovato.")
        return
    
    # Crea le tabelle LaTeX
    create_latex_tables(df)
    
    # Crea i whisker plot
    create_whisker_plots(df)

if __name__ == '__main__':
    main()