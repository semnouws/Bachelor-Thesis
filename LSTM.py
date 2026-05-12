import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from elecprep    import all_matrices as elec_m,  row_dates as elec_dates,  party_to_idx as elec_parties
from pollingprep import all_matrices as poll_m,  row_dates as poll_dates
from nkoprep     import get_nko_vector, NKO_FEATURE_DIM, SURVEY_FEATURES as feature_names

# ── CONFIG ─────────────────────────────────────────────
SEQ_LEN      = 5
TARGET_COL   = len(elec_parties) + 2
SEATS_COL    = len(elec_parties) + 1
TOTAL_SEATS  = 150
TRAIN_CUTOFF = pd.Timestamp('2012-01-01').timestamp()

idx_to_party = {v: k for k, v in elec_parties.items()}
n_parties    = len(elec_parties)

# ── DATA SOURCE SELECTION ──────────────────────────────
print("\n=== Data Source Selection ===")
print("Choose which data sources to include:\n")

def ask_yes_no(prompt):
    while True:
        ans = input(f"  {prompt} [y/n]: ").strip().lower()
        if ans in ('y', 'n'):
            return ans == 'y'
        print("  Please enter y or n.")

USE_ELEC    = ask_yes_no("Include real election results?")
USE_POLLING = ask_yes_no("Include polling data?")
USE_NKO     = ask_yes_no("Include NKO survey data?")

if not USE_ELEC and not USE_POLLING:
    raise ValueError("At least one of election or polling data must be enabled.")

input_size = 1 + (NKO_FEATURE_DIM if USE_NKO else 0)

print(f"\nRunning with: "
      f"{'elections ' if USE_ELEC else ''}"
      f"{'polling ' if USE_POLLING else ''}"
      f"{'NKO' if USE_NKO else ''}")
print(f"Input size per timestep: {input_size}\n")

# ── CLEAN AND TAG DATA ──────────────────────────────
elec_data = np.nan_to_num(elec_m, nan=0.0)
poll_data = np.nan_to_num(poll_m, nan=0.0)

all_tagged = []
if USE_ELEC:
    all_tagged += [(row, date, True) for row, date in zip(elec_data, elec_dates)]
if USE_POLLING:
    all_tagged += [(row, date, False) for row, date in zip(poll_data, poll_dates)]

all_tagged = sorted(all_tagged, key=lambda x: x[1])

# ── BUILD PER-PARTY TIME SERIES ─────────────────────
party_series  = {p: [] for p in range(n_parties)}
party_dates   = {p: [] for p in range(n_parties)}
party_seats   = {p: [] for p in range(n_parties)}
party_is_elec = {p: [] for p in range(n_parties)}

for row, date, is_election in all_tagged:
    one_hot = row[:n_parties]
    if one_hot.sum() == 0:
        continue
    p = int(np.argmax(one_hot))
    party_series[p].append(float(row[TARGET_COL]))
    party_dates[p].append(date)
    party_seats[p].append(int(row[SEATS_COL]) if is_election else -1)
    party_is_elec[p].append(is_election)

# ── CREATE SEQUENCES ────────────────────────────────
def make_sequences(series, dates, seats, is_elec, seq_len, party_name, use_nko):
    X, y, d, s, e = [], [], [], [], []
    for i in range(len(series) - seq_len):
        seq_steps = []
        for t in range(seq_len):
            poll_val = np.array([series[i + t]], dtype=np.float32)
            if use_nko:
                nko_vec = get_nko_vector(dates[i + t], party_name)
                step = np.concatenate([poll_val, nko_vec])
            else:
                step = poll_val
            seq_steps.append(step)
        X.append(np.array(seq_steps))
        y.append(series[i + seq_len])
        d.append(dates[i + seq_len])
        s.append(seats[i + seq_len])
        e.append(is_elec[i + seq_len])
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32),
            np.array(d), np.array(s, dtype=np.int32), np.array(e, dtype=bool))

# ── SPLIT PER-PARTY ─────────────────────────────────
train_X_list, train_y_list = [], []
test_records = []

for p in range(n_parties):
    party_name = idx_to_party[p]
    series = np.array(party_series[p],  dtype=np.float32)
    pdates = np.array(party_dates[p])
    pseats = np.array(party_seats[p],   dtype=np.int32)
    pielec = np.array(party_is_elec[p], dtype=bool)

    if len(series) <= SEQ_LEN:
        continue

    Xp, yp, dp, sp, ep = make_sequences(series, pdates, pseats, pielec, SEQ_LEN, party_name, USE_NKO)

    train_mask = dp < TRAIN_CUTOFF
    test_mask  = dp >= TRAIN_CUTOFF

    if train_mask.sum() == 0:
        continue

    train_X_list.append(Xp[train_mask])
    train_y_list.append(yp[train_mask])

    for seq, target, date, seats, is_elec in zip(
        Xp[test_mask], yp[test_mask], dp[test_mask], sp[test_mask], ep[test_mask]
    ):
        if is_elec:
            test_records.append((p, seq, float(target), date, int(seats)))

if not train_X_list:
    raise ValueError("No training data built. Check party_to_idx alignment.")

# ── TRAIN / VAL SPLIT ────────────────────────────────────────────────────
X_all = np.concatenate(train_X_list)
y_all = np.concatenate(train_y_list)

assert X_all.ndim == 3, f"Expected 3D tensor, got {X_all.ndim}D: {X_all.shape}"

# Chronological 80/20
split  = int(len(X_all) * 0.8)
trainX = torch.tensor(X_all[:split])
trainY = torch.tensor(y_all[:split])
valX   = torch.tensor(X_all[split:])
valY   = torch.tensor(y_all[split:])

print(f"Full dataset: {X_all.shape} | Train: {trainX.shape} | Val: {valX.shape}")

# ── ROBUSTNESS TESTS ────────────────────────────────────────────────────
def test_lstm_noise(model, valX, valY, criterion, noise_levels=[0.01, 0.05, 0.1]):
    model.eval()
    print("\n=== LSTM Noise Robustness ===")
    with torch.no_grad():
        baseline_loss = criterion(model(valX), valY).item()
        print(f"  Baseline Val Loss: {baseline_loss:.4f}")
        for noise in noise_levels:
            noisy = valX + torch.randn_like(valX) * noise
            loss  = criterion(model(noisy), valY).item()
            delta = loss - baseline_loss
            print(f"  Noise {noise:.2f} | Loss: {loss:.4f} | Δ {delta:+.4f}")

def test_lstm_ablation(model, valX, valY, criterion, feature_names):
    """Zero out one feature at a time across all timesteps."""
    model.eval()
    print("\n=== LSTM Feature Ablation ===")
    with torch.no_grad():
        baseline_loss = criterion(model(valX), valY).item()
        print(f"  Baseline Val Loss: {baseline_loss:.4f}\n")
        impacts = []
        for i, name in enumerate(feature_names):
            ablated = valX.clone()
            ablated[:, :, i] = 0
            loss = criterion(model(ablated), valY).item()
            delta = loss - baseline_loss
            impacts.append((name, loss, delta))

        # Sort by impact — biggest loss increase = most important feature
        for name, loss, delta in sorted(impacts, key=lambda x: -x[2]):
            flag = 'HIGH IMPACT' if delta > 1.0 else ''
            print(f"  {name:<25} Loss: {loss:.4f}  Δ {delta:+.4f}  {flag}")

def test_lstm_subsampling(model, valX, valY, criterion, fractions=[0.5, 0.7, 0.9], n_trials=20):
    model.eval()
    print("\n=== LSTM Subsampling Robustness ===")
    with torch.no_grad():
        baseline_loss = criterion(model(valX), valY).item()
        print(f"  Baseline Val Loss: {baseline_loss:.4f}")
        for frac in fractions:
            n = max(10, int(len(valX) * frac))
            losses = []
            for _ in range(n_trials):
                idx = torch.randperm(len(valX))[:n]
                loss = criterion(model(valX[idx]), valY[idx]).item()
                losses.append(loss)
            mean, std = np.mean(losses), np.std(losses)
            print(f"  Fraction {frac:.0%} ({n} samples) | "
                  f"Loss: {mean:.4f} ± {std:.4f}")

# ── MODEL ──────────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()

model = LSTMModel(input_size=input_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# ── TRAIN ──────────────────────────────────────────────────────────────────
train_losses = []
val_losses   = []
train_acc    = []
val_acc      = []

# A prediction is "accurate" if it's within this many percentage points of truth
ACCURACY_TOLERANCE = 2.0

def pct_accuracy(preds, targets, tol=ACCURACY_TOLERANCE):
    """% of predictions within tol percentage points of the target."""
    return ((preds - targets).abs() <= tol).float().mean().item() * 100

for epoch in range(250):
    # ── Train step
    model.train()
    optimizer.zero_grad()
    train_pred = model(trainX)
    train_loss = criterion(train_pred, trainY)
    train_loss.backward()
    optimizer.step()

    # ── Validation step
    model.eval()
    with torch.no_grad():
        val_pred = model(valX)
        val_loss = criterion(val_pred, valY)
        t_acc = pct_accuracy(train_pred.detach(), trainY)
        v_acc = pct_accuracy(val_pred, valY)

    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())
    train_acc.append(t_acc)
    val_acc.append(v_acc)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f} | "
              f"Train Acc: {t_acc:.1f}% | Val Acc: {v_acc:.1f}%")

if USE_NKO:
    feature_names_full = ['poll_value'] + list(feature_names)
else:
    feature_names_full = ['poll_value']
test_lstm_ablation(model, valX, valY, criterion, feature_names_full)
test_lstm_subsampling(model, valX, valY, criterion)
test_lstm_noise(model, valX, valY, criterion)

# ── PREDICT ON TEST SET ─────────────────────────────
model.eval()
results_by_date = {}

for p, seq, target, date, actual_seats in test_records:
    seq_tensor = torch.tensor(seq[None, :, :])
    with torch.no_grad():
        pred = model(seq_tensor).item()

    party_name = idx_to_party[p]
    if date not in results_by_date:
        results_by_date[date] = {"pred": {}, "actual": {}, "actual_seats": {}}

    results_by_date[date]["pred"][party_name]         = pred
    results_by_date[date]["actual"][party_name]       = target
    results_by_date[date]["actual_seats"][party_name] = actual_seats

# ── HELPERS ─────────────────────────────────────────
def normalise(d):
    total = sum(d.values())
    return {k: v / total * 100 for k, v in d.items()} if total else d

def pct_to_seats(pct_dict):
    raw       = {k: v / 100 * TOTAL_SEATS for k, v in pct_dict.items()}
    base      = {k: int(v) for k, v in raw.items()}
    remainder = TOTAL_SEATS - sum(base.values())
    order     = sorted(raw, key=lambda k: raw[k] - base[k], reverse=True)
    for k in order[:remainder]:
        base[k] += 1
    return base

# ── BUILD FINAL TABLE ───────────────────────────────
rows, mae_summary = [], []
epsilon = 1.0  # Huber threshold

for date in sorted(results_by_date.keys()):
    pred         = normalise(results_by_date[date]["pred"])
    actual       = results_by_date[date]["actual"]
    actual_seats = results_by_date[date]["actual_seats"]
    pred_seats   = pct_to_seats(pred)
    parties      = set(pred) | set(actual)

    errors, sq_errors, huber_errors = [], [], []

    for party in parties:
        p_pred   = pred.get(party, 0)
        p_actual = actual.get(party, 0)
        s_pred   = pred_seats.get(party, 0)
        s_actual = actual_seats.get(party, 0)

        err = p_pred - p_actual
        errors.append(abs(err))
        sq_errors.append(err ** 2)
        huber_errors.append(
            0.5 * err ** 2 if abs(err) <= epsilon
            else epsilon * (abs(err) - 0.5 * epsilon)
        )
        rows.append({
            "Date":            pd.Timestamp(date, unit='s').date(),
            "Party":           party,
            "Predicted %":     round(p_pred, 2),
            "Actual %":        round(p_actual, 2),
            "% Error":         round(err, 2),
            "Predicted Seats": s_pred,
            "Actual Seats":    s_actual,
            "Seat Error":      s_pred - s_actual,
        })

    total_seat_error = sum(abs(pred_seats.get(p, 0) - actual_seats.get(p, 0)) for p in parties)

    mae_summary.append({
        "Date":             pd.Timestamp(date, unit='s').date(),
        "MAE (%)":          round(np.mean(errors), 3),
        "MSE (%)":          round(np.mean(sq_errors), 3),
        "Huber Loss":       round(np.mean(huber_errors), 3),
        "Total Seat Error": total_seat_error,
    })

df_results = pd.DataFrame(rows).sort_values(["Date", "Actual Seats"], ascending=[True, False])
df_summary = pd.DataFrame(mae_summary)

print("\n=== Detailed Results ===\n")
print(df_results.to_string(index=False))
print("\n=== Summary per Election ===\n")
print(df_summary.to_string(index=False))

# ── PLOTS ──────────────────────────────────────────
epochs = range(len(train_losses))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

# Loss plot
ax1.plot(epochs, train_losses, label='Train Loss')
ax1.plot(epochs, val_losses,   label='Val Loss')
ax1.set_ylabel('MSE Loss')
ax1.set_title('Training vs Validation Loss (LSTM)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy plot
ax2.plot(epochs, train_acc, label=f'Train Acc (±{ACCURACY_TOLERANCE}%)')
ax2.plot(epochs, val_acc,   label=f'Val Acc (±{ACCURACY_TOLERANCE}%)')
ax2.set_ylabel(f'Accuracy (%)')
ax2.set_xlabel('Epoch')
ax2.set_title(f'Accuracy per Epoch (prediction within ±{ACCURACY_TOLERANCE} percentage points)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
