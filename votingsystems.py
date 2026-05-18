import pandas as pd
import numpy as np
from itertools import combinations
from elecprep   import all_matrices as elec_m, row_dates as elec_dates, party_to_idx as elec_parties
from nkoprep    import get_nko_vector, NKO_FEATURE_DIM,  all_waves, feature_cols

# ----ROBUSTNESS TESTS-------------------------------------------------------------------------------
def test_ballot_perturbation(ballots, parties, date=None, n_trials=100):
    """
    For each trial, randomly flip a fraction of voters' top-2 scores.
    Simulates survey respondents who were uncertain between two parties.
    """
    print(f"\n=== Voter Perturbation Robustness ({n_trials} trials) ===")
    results = {sys: [] for sys in ['Condorcet', 'Borda', 'Ranked', 'IRV']}

    for _ in range(n_trials):
        perturbed = []
        for ballot in ballots:
            if len(ballot) < 2 or np.random.random() > 0.2:  # perturb 20% of voters
                perturbed.append(ballot)
                continue

            # Swap the scores of the top two ranked parties
            ranking   = sorted(ballot.items(), key=lambda x: -x[1])
            new_ballot = dict(ballot)
            p1, s1    = ranking[0]
            p2, s2    = ranking[1]
            new_ballot[p1] = s2
            new_ballot[p2] = s1
            perturbed.append(new_ballot)

        results['Condorcet'].append(condorcet_ballots(perturbed, parties)['winner'])
        results['Borda'].append(borda_ballots(perturbed, parties)['winner'])
        results['Ranked'].append(ranked_voting_ballots(perturbed, parties)['winner'])
        results['IRV'].append(instant_runoff_ballots(perturbed, parties,
                                                      date=date,
                                                      actual_by_date=actual_by_date)['winner'])

    for sys, winners in results.items():
        counts    = pd.Series(winners).value_counts()
        stability = counts.iloc[0] / n_trials * 100
        flag      = '⚠ UNSTABLE' if stability < 80 else '✅'
        print(f"  {sys:<12} → {counts.index[0]:<20} {stability:.0f}% {flag}")

def test_ballot_bootstrap(ballots, parties, date=None, n_trials=100):
    """
    Sample WITH replacement — each trial simulates a different possible
    survey sample drawn from the same population.
    """
    print(f"\n=== Ballot Bootstrap Robustness ({n_trials} trials) ===")
    results = {sys: [] for sys in ['Condorcet', 'Borda', 'Ranked', 'IRV']}

    for _ in range(n_trials):
        sample = [ballots[i] for i in np.random.randint(0, len(ballots), len(ballots))]

        results['Condorcet'].append(condorcet_ballots(sample, parties)['winner'])
        results['Borda'].append(borda_ballots(sample, parties)['winner'])
        results['Ranked'].append(ranked_voting_ballots(sample, parties)['winner'])
        results['IRV'].append(instant_runoff_ballots(sample, parties,
                                                      date=date,
                                                      actual_by_date=actual_by_date)['winner'])

    for sys, winners in results.items():
        counts    = pd.Series(winners).value_counts()
        stability = counts.iloc[0] / n_trials * 100
        runner_up = f", runner-up: {counts.index[1]} ({counts.iloc[1]/n_trials:.0%})" \
                    if len(counts) > 1 else ""
        flag      = '⚠ UNSTABLE' if stability < 80 else '✅'
        print(f"  {sys:<12} → {counts.index[0]:<20} {stability:.0f}% {flag}{runner_up}")

SUBGROUP_LABELS = {
    'left_right':          ['left', 'centre', 'right'],
    'education_complete':  ['low', 'medium', 'high'],
    'income':              ['low', 'medium', 'high'],
    'age':                 ['young', 'middle', 'older'],
    'social_class':        ['working', 'middle', 'upper'],
}

def test_ballot_subgroups(ballots, parties, date=None, group_col='left_right',
                           year_df=None):
    """
    Split ballots by a demographic (e.g. left/right orientation) and check
    whether each subgroup produces the same winner.
    Requires the original year_df to have the grouping column.
    """
    if year_df is None or group_col not in year_df.columns:
        print(f"\n  Skipping subgroup test — {group_col} not available")
        return

    print(f"\n=== Subgroup Robustness by {group_col} ===")

    # Bin into thirds: left / centre / right
    col      = year_df[group_col].dropna()
    thirds   = [col.quantile(0.33), col.quantile(0.66)]
    labels = SUBGROUP_LABELS.get(group_col, ['low', 'medium', 'high'])

    for label, (lo, hi) in zip(labels, [
        (col.min(), thirds[0]),
        (thirds[0], thirds[1]),
        (thirds[1], col.max())
    ]):
        mask    = (year_df[group_col] >= lo) & (year_df[group_col] <= hi)
        indices = year_df[mask].index.tolist()
        sub     = [ballots[i] for i in indices if i < len(ballots)]

        if len(sub) < 10:
            print(f"  {label}: too few respondents ({len(sub)}), skipping")
            continue

        c = condorcet_ballots(sub, parties)['winner']
        b = borda_ballots(sub, parties)['winner']
        r = ranked_voting_ballots(sub, parties)['winner']
        v = instant_runoff_ballots(sub, parties, date=date,
                                    actual_by_date=actual_by_date)['winner']
        print(f"  {label:<8} ({len(sub):4d} voters) | "
              f"Condorcet: {c:<12} Borda: {b:<12} Ranked: {r:<12} IRV: {v}")

def test_system_agreement(ballots, parties, date=None):
    """Check whether all four systems agree — disagreement signals fragility."""
    winners = {
        'Condorcet': condorcet_ballots(ballots, parties)['winner'],
        'Borda':     borda_ballots(ballots, parties)['winner'],
        'Ranked':    ranked_voting_ballots(ballots, parties)['winner'],
        'IRV':       instant_runoff_ballots(ballots, parties,date=date, actual_by_date=actual_by_date)['winner'],
    }

    print(f"\n=== Cross-System Agreement ===")
    for sys, w in winners.items():
        print(f"  {sys:<12} → {w}")

    unique = set(winners.values())
    if len(unique) == 1:
        print(f"  All systems agree: {unique.pop()}")
    else:
        print(f"  DISAGREEMENT — {len(unique)} different winners: {unique}")
    return winners

def test_party_ablation(ballots, parties, date=None):
    """
    Remove one party at a time and check if the winner changes.
    Tests robustness against a single party's absence.
    """
    ABLATION_PARTIES = ['GL/PvdA', 'VVD', 'PVV', 'CDA', 'D66']

    print(f"\n=== Party Ablation Test ===")
    print(f"  Baseline (all parties):")

    baseline_parties = list(parties)
    results = {
        'Condorcet': condorcet_ballots(ballots, baseline_parties)['winner'],
        'Borda':     borda_ballots(ballots, baseline_parties)['winner'],
        'Ranked':    ranked_voting_ballots(ballots, baseline_parties)['winner'],
        'IRV':       instant_runoff_ballots(ballots, baseline_parties, date=date, actual_by_date=actual_by_date)['winner'],
    }
    for sys, w in results.items():
        print(f"    {sys:<12} → {w}")

    print()
    for removed in ABLATION_PARTIES:
        if removed not in parties:
            print(f"  [{removed} not in party list, skipping]")
            continue

        reduced = [p for p in parties if p != removed]

        ablated = {
            'Condorcet': condorcet_ballots(ballots, reduced)['winner'],
            'Borda':     borda_ballots(ballots, reduced)['winner'],
            'Ranked':    ranked_voting_ballots(ballots, reduced)['winner'],
            'IRV':       instant_runoff_ballots(ballots, reduced, date=date, actual_by_date=actual_by_date)['winner'],
        }

        print(f"  Remove {removed}:")
        for sys, w in ablated.items():
            changed = 'CHANGED' if w != results[sys] else 'stable'
            print(f"    {sys:<12} → {w:<20} {changed}")
        print()

# ── BUILD ACTUAL RESULTS DIRECTLY FROM ELECTION DATA ─────────────────────────
idx_to_party = {v: k for k, v in elec_parties.items()}
n_parties = len(elec_parties)
TARGET_COL = n_parties + 2

actual_by_date = {}
for row, date in zip(elec_m, elec_dates):
    one_hot = row[:n_parties]
    if one_hot.sum() == 0:
        continue
    p          = int(np.argmax(one_hot))
    party_name = idx_to_party[p]
    date_key   = pd.Timestamp(date, unit='s').date()
    pct        = float(row[TARGET_COL])
    if date_key not in actual_by_date:
        actual_by_date[date_key] = {}
    actual_by_date[date_key][party_name] = pct

def build_voter_ballots(date) -> list[dict]:
    """
    Build a list of individual voter ballots from NKO prob_ scores.
    Each ballot is {party_name: prob_score} for one respondent.
    Only respondents with at least 3 non-null prob_ scores are included.

    Uses the most recent NKO wave <= election date.This is always 2012 in this case.
    Returns list of dicts: [{party: score, ...}, ...]
    """

    def get_nko_year(date):
        year = pd.Timestamp(date).year
        valid = [y for y in all_waves.keys() if y <= year]
        return max(valid) if valid else None

    nko_year = get_nko_year(date)
    if nko_year is None:
        return []

    year_df = all_waves[nko_year]

    SUFFIX_TO_PARTY = {
        'pvda':    'PvdA',
        'cda':     'CDA',
        'vvd':     'VVD',
        'd66':     'D66',
        'gl':      'GroenLinks',
        'sgp':     'SGP',
        'cu':      'ChristenUnie',
        'gpv':     'ChristenUnie',
        'rpf':     'ChristenUnie',
        'sp':      'SP',
        'lpf':     'LPF',
        'pvdd':    'PvdD',
        'pvv':     'PVV',
        'ln':      'LN',
    }

    # Find all prob_ cols present in this wave
    prob_cols = {col: SUFFIX_TO_PARTY[col.replace('prob_', '')]
                 for col in year_df.columns
                 if col.startswith('prob_')
                 and col.replace('prob_', '') in SUFFIX_TO_PARTY}

    if not prob_cols:
        print(f"  WARNING: no prob_ cols found for year {nko_year}")
        return []

    ballots = []
    for _, respondent in year_df.iterrows():
        ballot = {}
        for col, party in prob_cols.items():
            val = respondent[col]
            if pd.notna(val):
                # If multiple cols map to same party (gpv/rpf → CU), take max
                if party in ballot:
                    ballot[party] = max(ballot[party], float(val))
                else:
                    ballot[party] = float(val)

        # Only include respondents with at least 3 valid prob scores
        if len(ballot) >= 3:
            ballots.append(ballot)

    print(f"  Built {len(ballots)} voter ballots from NKO {nko_year}")
    return ballots


def ballots_to_ranking(ballot: dict) -> list:
    """Sort a single ballot dict into a ranked list, highest prob first."""
    return sorted(ballot.keys(), key=lambda p: (-ballot[p], p))

# ── VOTING SYSTEM MENU ────────────────────────────────────────────────────────
VOTING_SYSTEMS = {
    '1': 'Condorcet',
    '2': 'Borda Count',
    '3': 'Ranked Voting',
    '4': 'Instant Runoff',
    '5': 'All',
}

print("\n=== Voting System Selection ===\n")
for key, name in VOTING_SYSTEMS.items():
    print(f"  {key}. {name}")

while True:
    choice = input("\nChoose a voting system: ").strip()
    if choice in VOTING_SYSTEMS:
        USE_CONDORCET = choice in ('1', '5')
        USE_BORDA     = choice in ('2', '5')
        USE_RANKED    = choice in ('3', '5')
        USE_IRV       = choice in ('4', '5')
        print(f"\nRunning: {VOTING_SYSTEMS[choice]}\n")
        break
    print("  Please enter 1, 2, 3, 4, or 5.")

# ── CONDORCET FROM BALLOTS ────────────────────────────────────────────────────
def condorcet_ballots(ballots: list, parties: list) -> dict:
    """
    Run Condorcet on individual ballots.
    For each pair (a, b), count how many voters ranked a above b vs b above a.
    Winner of the pair = majority preference.
    Condorcet winner = beats all others in pairwise majority.
    """
    wins = {p: 0 for p in parties}
    losses = {p: 0 for p in parties}
    pairwise_results = {}

    for a, b in combinations(parties, 2):
        prefer_a = 0
        prefer_b = 0

        for ballot in ballots:
            score_a = ballot.get(a, -1)  # missing = lowest preference
            score_b = ballot.get(b, -1)

            # Only skip if BOTH are missing (voter has no opinion on either)
            if score_a == -1 and score_b == -1:
                continue

            if score_a > score_b:
                prefer_a += 1
            elif score_b > score_a:
                prefer_b += 1
            # exact tie → no vote counted

        if prefer_a > prefer_b:
            winner_pair, loser_pair = a, b
        elif prefer_b > prefer_a:
            winner_pair, loser_pair = b, a
        else:
            pairwise_results[(a, b)] = 'tie'
            continue

        wins[winner_pair] += 1
        losses[loser_pair] += 1
        pairwise_results[(a, b)] = winner_pair

    n = len(parties)
    winner = next((p for p in parties if wins[p] == n - 1), None)

    return {
        'winner': winner,
        'wins': wins,
        'losses': losses,
        'pairwise_results': pairwise_results,
        'pairwise_counts': {(a, b): (
            sum(1 for bal in ballots
                if bal.get(a, -1) > bal.get(b, -1)
                or (bal.get(a, -1) != -1 and bal.get(b, -1) == -1)),
            sum(1 for bal in ballots
                if bal.get(b, -1) > bal.get(a, -1)
                or (bal.get(b, -1) != -1 and bal.get(a, -1) == -1))
        ) for a, b in combinations(parties, 2)},
        'cycle': winner is None,
    }


# ── BORDA FROM BALLOTS ────────────────────────────────────────────────────────
def borda_ballots(ballots: list, parties: list) -> dict:
    """
    Run Borda on individual ballots.
    Each ballot assigns N-1 points to 1st choice, N-2 to 2nd, etc.
    For parties missing from a ballot, they receive 0 points from that voter.
    Total Borda points summed across all ballots.
    """
    total_points = {p: 0.0 for p in parties}
    n = len(parties)

    for ballot in ballots:
        # Rank only the parties present in this ballot
        ranking = ballots_to_ranking({p: ballot[p] for p in parties
                                      if p in ballot})
        for rank, party in enumerate(ranking):
            total_points[party] += (n - 1 - rank)

    winner = max(total_points, key=total_points.get)
    ranking = sorted(total_points.items(), key=lambda x: -x[1])

    return {
        'winner':  winner,
        'points':  total_points,
        'ranking': ranking,
    }


# ── RANKED VOTING FROM BALLOTS ────────────────────────────────────────────────
def ranked_voting_ballots(ballots: list, parties: list) -> dict:
    """
    Positional scoring: 1st place = N points, 2nd = N-1, ..., last = 1.
    Simpler than Borda (no zero), sums across all ballots.
    """
    total_scores = {p: 0.0 for p in parties}
    n = len(parties)

    for ballot in ballots:
        ranking = ballots_to_ranking({p: ballot[p] for p in parties
                                      if p in ballot})
        for rank, party in enumerate(ranking):
            total_scores[party] += (n - rank)   # 1st gets N, last gets 1

    winner = max(total_scores, key=total_scores.get)
    ranking = sorted(total_scores.items(), key=lambda x: -x[1])

    return {
        'winner':  winner,
        'scores':  total_scores,
        'ranking': ranking,
    }


# ── INSTANT RUNOFF FROM BALLOTS ───────────────────────────────────────────────
def instant_runoff_ballots(ballots: list, parties: list,
                            date=None, actual_by_date: dict = None) -> dict:
    """
    True IRV on individual ballots.
    Tiebreak on elimination: party with lower actual votes in most recent
    prior election is eliminated first. If still tied, it is a true tie
    and both are eliminated.
    """
    remaining  = list(parties)
    rounds     = []
    eliminated = []

    # Build prior-election tiebreak lookup: most recent election BEFORE date
    prior_shares = {}
    if date and actual_by_date:
        prior_dates = [d for d in actual_by_date if d < date]
        if prior_dates:
            last_date = max(prior_dates)
            prior_shares = actual_by_date[last_date]

    def get_prior_share(p):
        return prior_shares.get(p, 0.0)

    while True:
        vote_counts = {p: 0 for p in remaining}
        for ballot in ballots:
            ranking = ballots_to_ranking({p: ballot[p] for p in remaining
                                          if p in ballot})
            if ranking:
                vote_counts[ranking[0]] += 1

        total = sum(vote_counts.values())
        if total == 0:
            break

        standings = dict(vote_counts)

        # Check majority
        for p in remaining:
            if vote_counts[p] / total > 0.5:
                rounds.append({'round': len(rounds)+1, 'standings': standings,
                                'eliminated': None, 'winner': p})
                return {'winner': p, 'rounds': rounds,
                        'eliminated_order': eliminated}

        if len(remaining) == 1:
            rounds.append({'round': len(rounds)+1, 'standings': standings,
                            'eliminated': None, 'winner': remaining[0]})
            return {'winner': remaining[0], 'rounds': rounds,
                    'eliminated_order': eliminated}

        # Find the minimum vote count
        min_votes = min(vote_counts.values())
        tied_losers = [p for p in remaining if vote_counts[p] == min_votes]

        if len(tied_losers) == 1:
            loser = tied_losers[0]
        else:
            # Tiebreak: eliminate whichever had lowest prior election share
            min_prior = min(get_prior_share(p) for p in tied_losers)
            still_tied = [p for p in tied_losers
                          if get_prior_share(p) == min_prior]

            if len(still_tied) == 1:
                loser = still_tied[0]
            else:
                # True tie: eliminate all of them simultaneously
                for loser in still_tied:
                    rounds.append({'round': len(rounds)+1, 'standings': standings,
                                    'eliminated': f'{loser} (true tie)',
                                    'winner': None})
                    eliminated.append(loser)
                    remaining.remove(loser)
                continue  # restart the while loop with all tied parties removed

        rounds.append({'round': len(rounds)+1, 'standings': standings,
                        'eliminated': loser, 'winner': None})
        eliminated.append(loser)
        remaining.remove(loser)

        if not remaining:
            # all parties eliminated in true ties — return the last eliminated as winner
            winner = eliminated[-1] if eliminated else None
            return {'winner': winner, 'rounds': rounds, 'eliminated_order': eliminated}

        winner = max(remaining, key=lambda p: vote_counts.get(p, 0))
        return {'winner': winner, 'rounds': rounds, 'eliminated_order': eliminated}



# ── VOTE-SHARE FALLBACK VERSIONS (used for actual results & no-ballot fallback) ──
def condorcet(shares: dict) -> dict:
    """Condorcet from vote shares: higher share = preferred in every pairwise match."""
    parties = list(shares.keys())
    wins, losses = {p: 0 for p in parties}, {p: 0 for p in parties}
    pairwise_results = {}

    for a, b in combinations(parties, 2):
        if shares[a] > shares[b]:
            winner_pair, loser_pair = a, b
        elif shares[b] > shares[a]:
            winner_pair, loser_pair = b, a
        else:
            pairwise_results[(a, b)] = 'tie'
            continue
        wins[winner_pair] += 1
        losses[loser_pair] += 1
        pairwise_results[(a, b)] = winner_pair

    n = len(parties)
    winner = next((p for p in parties if wins[p] == n - 1), None)
    return {
        'winner': winner,
        'wins': wins,
        'losses': losses,
        'pairwise_results': pairwise_results,
        'cycle': winner is None,
    }


def borda(shares: dict) -> dict:
    """Borda from vote shares: rank by share descending, assign N-1 down to 0."""
    parties = list(shares.keys())
    n = len(parties)
    ranking_order = sorted(parties, key=lambda p: -shares[p])
    points = {p: (n - 1 - i) for i, p in enumerate(ranking_order)}
    winner = max(points, key=points.get)
    return {
        'winner': winner,
        'points': points,
        'ranking': sorted(points.items(), key=lambda x: -x[1]),
    }


def ranked_voting(shares: dict) -> dict:
    """Ranked voting from vote shares: rank by share, assign N down to 1."""
    parties = list(shares.keys())
    n = len(parties)
    ranking_order = sorted(parties, key=lambda p: -shares[p])
    scores = {p: (n - i) for i, p in enumerate(ranking_order)}
    winner = max(scores, key=scores.get)
    return {
        'winner': winner,
        'scores': scores,
        'ranking': sorted(scores.items(), key=lambda x: -x[1]),
    }


def instant_runoff(shares: dict) -> dict:
    """IRV from vote shares: each round eliminate lowest share, repeat."""
    remaining = sorted(shares.keys(), key=lambda p: -shares[p])
    rounds, eliminated = [], []

    while len(remaining) > 1:
        total = sum(shares[p] for p in remaining)
        standings = {p: shares[p] for p in remaining}

        for p in remaining:
            if shares[p] / total > 0.5:
                rounds.append({'round': len(rounds)+1, 'standings': standings,
                                'eliminated': None, 'winner': p})
                return {'winner': p, 'rounds': rounds, 'eliminated_order': eliminated}

        loser = min(remaining, key=lambda p: (shares[p], p))
        rounds.append({'round': len(rounds)+1, 'standings': standings,
                        'eliminated': loser, 'winner': None})
        eliminated.append(loser)
        remaining.remove(loser)

    rounds.append({'round': len(rounds)+1, 'standings': {remaining[0]: shares[remaining[0]]},
                    'eliminated': None, 'winner': remaining[0]})
    return {'winner': remaining[0], 'rounds': rounds, 'eliminated_order': eliminated}

# ── run_per_election ──────────────────────────────────────────────────
def run_per_election(df_results: pd.DataFrame):
    rows = []

    for i, (date, group) in enumerate(df_results.groupby('Date')):
        pred_shares   = dict(zip(group['Party'], group['Predicted %']))
        actual_shares = actual_by_date.get(date, {})

        if not actual_shares:
            print(f"  WARNING: no actual election data for {date}, skipping.")
            continue

        # Build voter ballots from NKO prob_ scores
        ballots = build_voter_ballots(date)
        use_ballots = len(ballots) > 0

        # Parties to include: union of predicted + actual
        all_parties = sorted(set(pred_shares) | set(actual_shares))

        if use_ballots and i == 0:
            test_system_agreement(ballots, all_parties, date=date)
            test_ballot_bootstrap(ballots, all_parties, date=date)
            test_ballot_perturbation(ballots, all_parties, date=date)
            for group_col in ['left_right', 'education_complete', 'income', 'age', 'social_class']:
                test_ballot_subgroups(ballots, all_parties, date=date,
                                      group_col=group_col,
                                      year_df=all_waves[2012])
            test_party_ablation(ballots, all_parties, date=date)

        row = {'Date': date}

        # ── Condorcet ─────────────────────────────────────────────────────────
        if USE_CONDORCET:
            if use_ballots:
                pred_c = condorcet_ballots(ballots, all_parties)
            else:
                pred_c = condorcet(pred_shares)
            actual_c = condorcet(actual_shares)

            row['Condorcet Predicted Winner'] = pred_c['winner'] or 'NO WINNER (cycle)'
            row['Condorcet Actual Winner'] = actual_c['winner'] or 'NO WINNER (cycle)'

        # ── Borda ─────────────────────────────────────────────────────────────
        if USE_BORDA:
            if use_ballots:
                pred_b = borda_ballots(ballots, all_parties)
            else:
                pred_b = borda(pred_shares)
            actual_b = borda(actual_shares)

            row['Borda Predicted Winner'] = pred_b['winner']
            row['Borda Actual Winner'] = actual_b['winner']

        # ── Ranked Voting ─────────────────────────────────────────────────────
        if USE_RANKED:
            if use_ballots:
                pred_r = ranked_voting_ballots(ballots, all_parties)
            else:
                pred_r = ranked_voting(pred_shares)
            actual_r = ranked_voting(actual_shares)

            row['Ranked Predicted Winner'] = pred_r['winner']
            row['Ranked Actual Winner'] = actual_r['winner']

        # ── Instant Runoff ────────────────────────────────────────────────────
        if USE_IRV:
            if use_ballots:
                pred_i = instant_runoff_ballots(ballots, all_parties, date=date, actual_by_date=actual_by_date)
            else:
                pred_i = instant_runoff(pred_shares)
            actual_i = instant_runoff(actual_shares)

            row['IRV Predicted Winner'] = pred_i['winner']
            row['IRV Actual Winner'] = actual_i['winner']

        rows.append(row)

    return pd.DataFrame(rows)

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from LSTM import df_results

    summary = run_per_election(df_results)

    print("\n\n=== Summary Across All Elections ===\n")
    print(summary.to_string(index=False))
