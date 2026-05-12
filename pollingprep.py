import pandas as pd
import numpy as np
import os
from elecprep import party_to_idx  # shared index

file_paths = [
    r"C:\Users\fleur\AppData\Roaming\JetBrains\PyCharm2023.3\scratches\Data\Polling\Peilingwijzer 2010-2012.tab",
    r"C:\Users\fleur\AppData\Roaming\JetBrains\PyCharm2023.3\scratches\Data\Polling\Peilingwijzer 2012-2017.csv",
    r"C:\Users\fleur\AppData\Roaming\JetBrains\PyCharm2023.3\scratches\Data\Polling\Peilingwijzer 2017-2021.csv",
    r"C:\Users\fleur\AppData\Roaming\JetBrains\PyCharm2023.3\scratches\Data\Polling\Peilingwijzer 2021-2023.csv",
]

poll_aliases = {
    'X50PLUS': '50PLUS',
    'GL':      'GL/PvdA',
    'PvdA':    'GL/PvdA',
    'Denk':    'DENK',
    'GPV':     'CU',
    'RPF':     'CU',
}

n_parties = len(party_to_idx)

all_rows  = []
all_dates = []

# ── Helper: append a matrix row ───────────────────────────────────────
def append_row(party, pct_0_to_100, total_pct_0_to_100, date_ts):
    """pct and total_pct are both in 0–100 scale."""
    if party not in party_to_idx:
        return
    if pd.isna(pct_0_to_100) or pct_0_to_100 <= 0:
        return

    one_hot = np.zeros(n_parties, dtype=np.float32)
    one_hot[party_to_idx[party]] = 1.0

    seats     = (pct_0_to_100 / total_pct_0_to_100 * 150) if total_pct_0_to_100 > 0 else 0
    zetelspct = seats / 150 * 100

    matrix_row = np.concatenate([
        one_hot,
        [np.nan],           # AantalStemmen — not available for polling
        [seats],            # proportional seats, never used as actual
        [pct_0_to_100],     # Percentage (0–100)
        [zetelspct],        # ZetelsPct
    ])
    all_rows.append(matrix_row)
    all_dates.append(date_ts)

# ── SOURCE 1: Peilingwijzer files (2010–2023) ─────────────────────────
raw_frames = {}
for path in file_paths:
    sep = '\t' if path.endswith('.tab') else ','
    df  = pd.read_csv(path, sep=sep)
    df.columns = df.columns.str.strip()
    date_col = 'date' if 'date' in df.columns else df.columns[0]
    df = df.rename(columns={date_col: 'date'})
    df = df.loc[:, ~(df.columns.str.endswith('.low') | df.columns.str.endswith('.high'))]
    df = df.rename(columns=poll_aliases)
    df = df.T.groupby(level=0).sum().T
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df = df.set_index('date')
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df * 100   # convert 0–1 fractions to 0–100
    df = df.round(4)
    raw_frames[os.path.basename(path)] = df

for key, df in raw_frames.items():
    for date, row in df.iterrows():
        total_pct = row.dropna().sum()
        for party in df.columns:
            pct = row.get(party, np.nan)
            append_row(party, pct, total_pct, date.timestamp())

# ── SOURCE 2: data_all_save.csv (1997–2010) ───────────────────────────
# Raw individual poll rows; percentage column is 0–1 scale
DATA_ALL_PATH = r"C:\Users\fleur\AppData\Roaming\JetBrains\PyCharm2023.3\scratches\Data\Polling\data_all_save.csv"

df_all = pd.read_csv(DATA_ALL_PATH)
df_all['date'] = pd.to_datetime(df_all['date'], format='%d-%m-%Y', dayfirst=True)

# Keep only 1997–2010 (exclusive of 2010 since Peilingwijzer covers from mid-2010)
mask = (df_all['date'] >= '1997-01-01') & (df_all['date'] < '2010-01-01')
df_all = df_all[mask].copy()

# Apply aliases
df_all['party'] = df_all['party'].replace(poll_aliases)

# percentage column is 0–1, convert to 0–100
df_all['pct100'] = df_all['percentage'] * 100

# For each poll date, compute the total across all parties polled that day
# (same pollster on same date = one poll)
for (date, company), group in df_all.groupby(['date', 'company']):
    total_pct = group['pct100'].sum()
    for _, poll_row in group.iterrows():
        append_row(poll_row['party'], poll_row['pct100'], total_pct, date.timestamp())

# ── SOURCE 3: polling_1964_2010.csv (election results 1971–1997) ──────
# These are election-day seat counts; convert to percentage via total seats
POLL_HIST_PATH = r"C:\Users\fleur\AppData\Roaming\JetBrains\PyCharm2023.3\scratches\Data\Polling\polling_1964_2010.csv"

df_hist = pd.read_csv(POLL_HIST_PATH)
df_hist['date'] = pd.to_datetime(df_hist['date'], format='%Y-%m-%d')

# Keep only 1971–1997
mask = (df_hist['date'] >= '1971-01-01') & (df_hist['date'] < '1997-01-01')
df_hist = df_hist[mask].copy()

# Apply aliases
df_hist['party'] = df_hist['party'].replace(poll_aliases)

# For each election date, compute total seats across all parties in the file
for date, group in df_hist.groupby('date'):
    total_seats = group['seats'].sum()
    if total_seats == 0:
        continue
    for _, hist_row in group.iterrows():
        party = hist_row['party']
        seats = hist_row['seats']
        if seats <= 0:
            continue
        pct = seats / total_seats * 100   # convert seat share to percentage
        append_row(party, pct, 100.0, date.timestamp())

# ── ASSEMBLE AND SORT ─────────────────────────────────────────────────
all_matrices = np.array(all_rows, dtype=np.float32)
row_dates = np.array(all_dates, dtype=np.float64)

sort_idx = np.argsort(row_dates)
all_matrices = all_matrices[sort_idx]
row_dates = row_dates[sort_idx]

if __name__ == "__main__":
    print("Party encoding guide")
    print("=" * 40)
    for party, idx in party_to_idx.items():
        print(f"  {idx:>3}  {party}")
    print("=" * 40)
    print(f"\nMatrix shape: {all_matrices.shape}")
    print(f"  → {n_parties} party one-hot columns")
    print(f"     1 AantalStemmen column (NaN)")
    print(f"     1 AantalZetels column (proportional)")
    print(f"     1 Percentage column")
    print(f"     1 ZetelsPct column")

    # Date range per source for verification
    from_hist = row_dates[row_dates < pd.Timestamp('1997-01-01').timestamp()]
    from_data = row_dates[(row_dates >= pd.Timestamp('1997-01-01').timestamp()) &
                           (row_dates <  pd.Timestamp('2010-01-01').timestamp())]
    from_pwijz = row_dates[row_dates >= pd.Timestamp('2010-01-01').timestamp()]

    def ts(arr):
        if len(arr) == 0: return "none"
        return f"{pd.Timestamp(arr.min(), unit='s').date()} – {pd.Timestamp(arr.max(), unit='s').date()}"

    print(f"\nRows from polling_1964_2010 (1971–1997): {len(from_hist)}  [{ts(from_hist)}]")
    print(f"Rows from data_all_save     (1997–2010): {len(from_data)}  [{ts(from_data)}]")
    print(f"Rows from Peilingwijzer     (2010–2023): {len(from_pwijz)}  [{ts(from_pwijz)}]")