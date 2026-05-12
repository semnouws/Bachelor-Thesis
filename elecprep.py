import pandas as pd
import numpy as np
import glob
import os
import re

# ── Load all files and apply aliases ─────────────────────────────────
data_folder = r"C:\Users\fleur\AppData\Roaming\JetBrains\PyCharm2023.3\scratches\Data\Uitslagen\*.csv"
file_paths = glob.glob(data_folder)

elec_aliases = {
    'Artikel 1':                        'BIJ1',
    'BVNL / Groep Van Haga':            'BVNL',
    'Christen Democratisch Appèl (CDA)':'CDA',
    'ChristenUnie':                     'CU',
    'Democraten 66 (D66)':              'D66',
    'GROENLINKS':                       'GL/PvdA',
    'GL':                               'GL/PvdA',
    'Fortuyn':                          'LPF',
    'LN':                               'Leefbaar Nederland',
    'Liberaal Democratische Partij (LibDem)': 'Liberaal Democratische Partij',
    'Libertarische Partij (LP)':        'LP',
    'Libertarische Partij':             'LP',
    'LP (Libertarische Partij)':        'LP',
    'LP (Libertaire Partij)':           'LP',
    'PVV (Partij voor de Vrijheid)':    'PVV',
    'Partij van de Arbeid (P.v.d.A.)':  'GL/PvdA',
    'Partij van de Toekomst':           'PvdT',
    'Partij van de Toekomst (PvdT)':    'PvdT',
    'PvdA':                             'GL/PvdA',
    'Partij voor de Dieren':            'PvdD',
    'SP (Socialistische Partij)':       'SP',
    'Staatkundig Gereformeerde Partij (SGP)': 'SGP',
    'Rooms Katholieke Partij Nederland':'RKPN',
    'Piratenpartij':                    'Piratenpartij - De Groenen',
    'De Groenen':                       'Piratenpartij - De Groenen',
    'Federatie Groenen':                'Piratenpartij - De Groenen',
    'Forum voor Democratie':            'FvD',
    'Nieuw Sociaal Contract':           'NSC',
    'Denk':                             'DENK',
    'VNL (VoorNederland)':              'VNL',
    'GROENLINKS / Partij van de Arbeid (PvdA)': 'GL/PvdA',
    'Christen Democratisch Appèl (CDA)':'CDA',
}

def extract_date(path):
    match = re.search(r'TK(\d{8})', os.path.basename(path))
    if match:
        date_str = match.group(1)
        return pd.to_datetime(date_str, format='%Y%m%d')
    return None

def extract_year(path):
    match = re.search(r'TK(\d{4})', os.path.basename(path))
    return int(match.group(1)) if match else 0

raw_frames = {}
for path in file_paths:
    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip()
    df["Partij"] = df["Partij"].replace(elec_aliases)
    raw_frames[path] = df

# ── Build global party index ──────────────────────────────────────────
all_parties = sorted(set(
    party
    for df in raw_frames.values()
    for party in df["Partij"].dropna().unique()
))

party_to_idx = {party: idx for idx, party in enumerate(all_parties)}

# ── Process each file ─────────────────────────────────────────────────
def process_elec(raw_frames, party_to_idx):
    results = {}
    n_parties = len(party_to_idx)

    for path, df in raw_frames.items():
        df = df.copy()
        df = df[["Partij", "AantalStemmen", "AantalZetels"]].copy()
        df = df[df["AantalStemmen"].notna() & (df["AantalStemmen"] > 0)]
        df = df.groupby("Partij", as_index=False).sum()

        total_stemmen = df["AantalStemmen"].sum()
        total_zetels  = df["AantalZetels"].sum()
        df["Percentage"]  = (df["AantalStemmen"] / total_stemmen * 100) if total_stemmen > 0 else 0
        df["ZetelsPct"]   = (df["AantalZetels"]  / total_zetels  * 100) if total_zetels  > 0 else 0

        election_date = extract_date(path)
        year          = extract_year(path)

        one_hot = np.zeros((len(df), n_parties), dtype=np.float32)
        for i, party in enumerate(df["Partij"]):
            if party in party_to_idx:
                one_hot[i, party_to_idx[party]] = 1.0

        stemmen    = df["AantalStemmen"].values.astype(np.float32).reshape(-1, 1)
        zetels     = df["AantalZetels"].values.astype(np.float32).reshape(-1, 1)
        pct        = df["Percentage"].values.astype(np.float32).reshape(-1, 1)
        zetels_pct = df["ZetelsPct"].values.astype(np.float32).reshape(-1, 1)

        matrix = np.concatenate([one_hot, stemmen, zetels, pct, zetels_pct], axis=1)

        results[path] = {
            "matrix":        matrix,
            "df":            df,
            "file":          os.path.basename(path),
            "date":          election_date,
            "year":          year,
        }

    return results

datasets = process_elec(raw_frames, party_to_idx)

# ── Sort chronologically and concatenate ─────────────────────────────
sorted_datasets = sorted(datasets.items(), key=lambda x: x[1]["date"] or pd.Timestamp.min)

all_matrices = np.concatenate([d["matrix"] for _, d in sorted_datasets], axis=0)

# row_dates: exact election date per row
row_dates = np.concatenate([
    np.array([d["date"].timestamp()] * d["matrix"].shape[0], dtype=np.float64)
    for _, d in sorted_datasets
])

if __name__ == "__main__":
    print("Party encoding guide")
    for party, idx in party_to_idx.items():
        print(f"  {idx:>3}  {party}")
    print("=" * 40)
    print(f"\nCombined matrix shape: {all_matrices.shape}")
    print(f"  → {len(party_to_idx)} party one-hot columns")
    print(f"     1 AantalStemmen column")
    print(f"     1 AantalZetels column")
    print(f"     1 vote percentage column")
    print(f"     1 seat percentage column")
    print(f"\nDate range: {row_dates.min():.0f} – {row_dates.max():.0f}")
