import numpy as np
import sys
import pandas as pd

sys.path.append(r"C:\Users\fleur\AppData\Roaming\JetBrains\PyCharm2023.3\scratches")
from pollingrprep import poll_df
from elecprep import elec_df

# Aliases that apply to BOTH sources
shared_aliases = {
    'Denk':'DENK',
    'GROENLINKS / Partij van de Arbeid (PvdA)':'GL/PvdA',
    'Forum voor Democratie':'FvD',
    'LP (Libertarische Partij)':'LP',
    'LP (Libertaire Partij)':'LP',
    'Libertarische Partij':'LP',
    'Nieuw Sociaal Contract':'NSC',
    'Partij van de Toekomst (PvdT)':'PvdT',
    'Piratenpartij':'Piratenpartij - De Groenen',
    'VNL (VoorNederland)':'VNL',
}

def apply_aliases(parties, aliases):
    """Apply aliases repeatedly until stable."""
    result = {}
    for k, v in parties.items():
        key = k
        # Keep resolving until the key stops changing
        for _ in range(10):
            new_key = aliases.get(key, key)
            if new_key == key:
                break
            key = new_key
        result[key] = v
    return result

# Merge shared aliases into both before applying
combined_poll_aliases = {**shared_aliases, **poll_aliases}
combined_elec_aliases = {**shared_aliases, **elec_aliases}

poll_parties = apply_aliases(poll_parties, combined_poll_aliases)
elec_parties = apply_aliases(elec_parties, combined_elec_aliases)
# ── 1. Build unified party index ─────────────────────────────────────
all_party_names = sorted(set(poll_parties.keys()) | set(elec_parties.keys()))
unified_idx     = {party: i for i, party in enumerate(all_party_names)}
n_unified       = len(unified_idx)

# ── 2. Realign polling matrix ─────────────────────────────────────────
# Polling columns: [party_0 ... party_N]  (percentages, no stemmen/zetels)
# Output columns:  [party_0 ... party_N | percentage | stemmen | zetels]
# Polling rows:    percentage filled, stemmen=0, zetels=0

def realign_polling(matrix, old_idx, new_idx):
    n_rows  = matrix.shape[0]
    n_cols  = len(new_idx) + 3  # parties + percentage + stemmen + zetels
    result  = np.zeros((n_rows, n_cols), dtype=np.float32)
    for party, old_col in old_idx.items():
        if party in new_idx:
            result[:, new_idx[party]] = matrix[:, old_col]
    # percentage column = sum of all party polling percentages per row
    result[:, len(new_idx)] = matrix.sum(axis=1)
    # stemmen and zetels stay 0 for polling rows
    return result

# ── 3. Realign election matrix ────────────────────────────────────────
# Election columns: [party_0 ... party_N | stemmen | zetels]
# Output columns:   [party_0 ... party_N | percentage | stemmen | zetels]
# Election rows:    percentage=0, stemmen and zetels filled

def realign_election(matrix, old_idx, new_idx):
    n_rows  = matrix.shape[0]
    n_cols  = len(new_idx) + 4  # parties + percentage + stemmen + zetels + seat_pct
    result  = np.zeros((n_rows, n_cols), dtype=np.float32)
    for party, old_col in old_idx.items():
        if party in new_idx:
            result[:, new_idx[party]] = matrix[:, old_col]
    result[:, len(new_idx)]     = matrix[:, -4]  # vote percentage
    result[:, len(new_idx) + 1] = matrix[:, -3]  # AantalStemmen
    result[:, len(new_idx) + 2] = matrix[:, -2]  # AantalZetels
    result[:, len(new_idx) + 3] = matrix[:, -1]  # seat percentage
    return result

poll_realigned = realign_polling(poll_m,  poll_parties,  unified_idx)
elec_realigned = realign_election(elec_m, elec_parties, unified_idx)

#all_matrices = np.vstack((poll_realigned, elec_realigned))

party_to_idx = unified_idx

# ----------4. Printing -----------------------
party_cols  = [f"party_{i}_{name}" for name, i in sorted(unified_idx.items(), key=lambda x: x[1])]
extra_cols  = ["percentage", "AantalStemmen", "AantalZetels", "source"]

col_names   = party_cols + extra_cols

print(col_names)

# Add source column: 0 = polling, 1 = election
poll_source = np.zeros((len(poll_realigned), 1), dtype=np.float32)
elec_source = np.ones((len(elec_realigned), 1), dtype=np.float32)

poll_with_source = np.concatenate([poll_realigned, poll_source], axis=1)
elec_with_source = np.concatenate([elec_realigned, elec_source], axis=1)

party_cols = [f"party_{i}_{name}" for name, i in sorted(unified_idx.items(), key=lambda x: x[1])]
extra_cols_poll = ["percentage", "AantalStemmen", "AantalZetels", "source"]
extra_cols_elec = ["percentage", "AantalStemmen", "AantalZetels", "seat_pct", "source"]

df_poll = pd.DataFrame(poll_with_source, columns=party_cols + extra_cols_poll)
df_elec = pd.DataFrame(elec_with_source, columns=party_cols + extra_cols_elec)

poll_nonzero = [c for c in party_cols if df_poll[c].any()]
elec_nonzero = [c for c in party_cols if df_elec[c].any()]

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

#print("\n── Polling matrix ──")
#print(df_poll[poll_nonzero + extra_cols_poll])

#print("\n── Election matrix ──")
#print(df_elec[elec_nonzero + extra_cols_elec])

#print("\n--Combined matrix--")
#print(all_matrices)