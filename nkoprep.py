import pandas as pd
import numpy as np
import pyreadstat

# ── PATH ──────────────────────────────────────────────
BASE = r'C:\Users\fleur\AppData\Roaming\JetBrains\PyCharm2023.3\scratches\Data\NKO'

def fast_read(path, encoding='utf-8', usecols=None):
    df, _ = pyreadstat.read_sav(
        path,
        encoding=encoding,
        usecols=usecols,
        apply_value_formats=False,
        formats_as_category=False,
    )
    return df

# 1. RENAME MAPS
rename_questions = {
    1989: {
        'var007': 'interested', 'var008': 'interest_score',
        'var019': 'intent_vote', 'var020': 'intent_party', 'var021': 'obliged_party',
        'var022': 'adherent', 'var024': 'convinced_adherent', 'var027': 'adherent_strength',
        'var028': 'adherent_direction', 'var023': 'adherent_to', 'var025': 'attracted', 'var026': 'attracted_to',
        'var055': 'vote_last', 'var056': 'vote_last_party',
        'var032': 'satisfaction_gov', 'var035': 'satisfaction_policy',
        'var086': 'left_right',
        'var092': 'age', 'var093': 'gender', 'var094': 'education_complete',
        'var095': 'martial', 'var096': 'employment',
        'var122': 'social_class', 'var123': 'income',
        'var146': 'vote', 'var147': 'vote_party',
        'var227': 'cynicism',
        'var158': 'previous_vote', 'var159': 'previous_vote_party',
        'var162': 'consider_novote', 'var163': 'hesitate', 'var164': 'consider_party',
        'var249': 'prob_pvda', 'var250': 'prob_vvd', 'var251': 'prob_d66', 'var252': 'prob_ppr',
        'var253': 'prob_cpn', 'var254': 'prob_cda', 'var255': 'prob_gl', 'var256': 'prob_sgp',
        'var257': 'prob_psp', 'var258': 'prob_gpv', 'var259': 'prob_rpf', 'var260': 'prob_cd',
        'var167': 'novote_party',
    },
    1994: {
        'var008': 'interested', 'var009': 'interest_score',
        'var037': 'intent_vote', 'var038': 'intent_party', 'var054': 'obliged_party',
        'var021': 'adherent', 'var023': 'convinced_adherent', 'var026': 'adherent_strength',
        'var027': 'adherent_direction', 'var022': 'adherent_to', 'var024': 'attracted', 'var025': 'attracted_to',
        'var055': 'vote_last', 'var056': 'vote_last_party',
        'var035': 'satisfaction_gov', 'var036': 'satisfaction_policy',
        'var139': 'left_right',
        'var172': 'age', 'var176': 'gender', 'var174': 'education_complete',
        'var177': 'martial', 'var181': 'employment',
        'var178': 'social_class', 'var179': 'income', 'var180': '#household',
        'var280': 'vote', 'var281': 'vote_party',
        'var432': 'efficacy_internal', 'var433': 'efficacy_external', 'var414': 'cynicism',
        'var289': 'previous_vote', 'var290': 'previous_vote_party',
        'var294': 'consider_novote', 'var295': 'hesitate', 'var296': 'consider_party',
        'var472': 'prob_pvda', 'var473': 'prob_vvd', 'var474': 'prob_d66', 'var475': 'prob_gl',
        'var476': 'prob_cda', 'var477': 'prob_sgp', 'var478': 'prob_gpv', 'var479': 'prob_rpf', 'var480': 'prob_cd',
        'var299': 'novote_party',
    },
    1998: {
        'v0033': 'interested', 'v0034': 'interest_score',
        'v0075': 'intent_vote', 'v0076': 'intent_party', 'v0083': 'obliged_party',
        'v0050': 'adherent', 'v0055': 'convinced_adherent', 'v0058': 'adherent_strength',
        'v0059': 'adherent_direction', 'v0054': 'attracted_to',
        'v0165': 'vote_last', 'v0166': 'vote_last_party',
        'v0073': 'satisfaction_gov', 'v0074': 'satisfaction_policy',
        'v0160': 'left_right',
        'v0316': 'age', 'v0288': 'gender', 'v0352': 'education_complete', 'v0353': 'education_attend',
        'v0351': 'martial', 'v0354': 'employment',
        'v0394': 'social_class', 'v0348': 'income',
        'v0280': '#household', 'v0281': 'composition_household',
        'v0610': 'vote', 'v0611': 'vote_party',
        'v0795': 'efficacy_external', 'v0799': 'efficacy_internal', 'v0803': 'cynicism',
        'v0745': 'satisfaction_democracy',
        'v0625': 'previous_vote', 'v0626': 'previous_vote_party',
        'v0631': 'consider_novote', 'v0632': 'hesitate',
        'v0633': 'consider_party',
        'v0830': 'prob_pvda', 'v0831': 'prob_vvd', 'v0832': 'prob_d66', 'v0833': 'prob_gl',
        'v0834': 'prob_cda', 'v0835': 'prob_sgp', 'v0836': 'prob_gpv', 'v0837': 'prob_rpf',
        'v0838': 'prob_cd', 'v0839': 'prob_sp', 'v0840': 'prob_aov/55+',
        'v0642': 'novote_party', 'v0870': 'social_class',
    },
    2002: {
        'v0004': 'interested', 'v0005': 'interest_score',
        'v0155': 'intent_vote', 'v0156': 'intent_party', 'v0179': 'obliged_party',
        'v0087': 'adherent', 'v0108': 'convinced_adherent', 'v0112': 'adherent_strength',
        'v0088': 'adherent_to_pvda', 'v0089': 'adherent_to_cda', 'v0090': 'adherent_to_vvd', 'v0091': 'adherent_to_d66',
        'v0092': 'adherent_to_gl', 'v0093': 'adherent_to_sgp', 'v0094': 'adherent_to_cu', 'v0095': 'adherent_to_ln',
        'v0096': 'adherent_to_lpf', 'v0097': 'adherent_to_sp', 'v0098': 'adherent_to_vsp', 'v0099': 'adherent_to_ll',
        'v0109': 'attracted', 'v0110': 'attracted_to',
        'v0235': 'vote_last', 'v0236': 'vote_last_party',
        'v0152': 'satisfaction_gov', 'v0153': 'satisfaction_policy', 'v0154': 'satisfaction_democracy',
        'v0234': 'left_right',
        'v0457': 'age', 'v0459': 'gender', 'v0463': 'education_complete',
        'v0460': 'martial', 'v0464': 'employment',
        'v0505': 'social_class', 'v0557': 'income',
        'v0417': '#household', 'v0418': 'composition_household',
        'v0646': 'vote', 'v0647': 'vote_party',
        'v0921': 'efficacy_external', 'v0925': 'efficacy_internal', 'v0929': 'cynicism',
        'v0702': 'previous_vote_party',
        'v0704': 'previous_pvda', 'v0705': 'previous_cda', 'v0706': 'previous_vvd',
        'v0707': 'previous_d66', 'v0708': 'previous_gl', 'v0709': 'previous_sgp',
        'v0710': 'previous_cu', 'v0711': 'previous_ln', 'v0712': 'previous_lpf',
        'v0713': 'previous_sp', 'v0714': 'previous_vsp', 'v0715': 'previous_cd',
        'v0717': 'previous_kvp', 'v0718': 'previous_chu', 'v0719': 'previous_arp',
        'v0720': 'previous_sdap', 'v0721': 'previous_cpn', 'v0722': 'previous_ppr',
        'v0723': 'previous_psp', 'v0724': 'previous_boerenpartij', 'v0725': 'previous_senioren2000',
        'v0726': 'previous_ds70', 'v0727': 'previous_other',
        'v0737': 'consider_novote', 'v0738': 'hesitate',
        'v0739': 'consider_pvda', 'v0740': 'consider_cda', 'v0741': 'consider_vvd',
        'v0742': 'consider_d66', 'v0743': 'consider_gl', 'v0744': 'consider_sgp',
        'v0745': 'consider_cu', 'v0746': 'consider_ln', 'v0747': 'consider_lpf',
        'v0748': 'consider_sp', 'v0749': 'consider_vsp', 'v0750': 'consider_vipo',
        'v0751': 'consider_blanc',
        'v0932': 'prob_cda', 'v0933': 'prob_pvda', 'v0934': 'prob_vvd', 'v0935': 'prob_d66',
        'v0936': 'prob_gl', 'v0937': 'prob_sgp', 'v0938': 'prob_cu', 'v0939': 'prob_ln',
        'v0940': 'prob_sp', 'v0941': 'prob_lpf',
    },
    2003: {
        'x0031': 'interested', 'x0032': 'interest_score',
        'x0195': 'vote', 'x0196': 'vote_party',
        'x0160': 'vote_last', 'x0162': 'vote_last_party',
        'x0339': 'satisfaction_democracy', 'x0374': 'left_right',
        'x0496': 'age', 'x0498': 'gender', 'x0502': 'education_complete',
        'x0499': 'martial', 'x0503': 'employment',
        'x0544': 'social_class', 'x0545': 'income',
        'x0461': '#household', 'x0462': 'composition_household',
        'x0390': 'efficacy_external', 'x0394': 'efficacy_internal', 'x0398': 'cynicism',
        'x0319': 'novote_party',
        'x0236': 'previous_vote_party',
        'x0238': 'previous_pvda', 'x0239': 'previous_cda', 'x0240': 'previous_vvd',
        'x0241': 'previous_d66', 'x0242': 'previous_gl', 'x0243': 'previous_sgp',
        'x0244': 'previous_cu', 'x0245': 'previous_ln', 'x0246': 'previous_lpf',
        'x0247': 'previous_sp', 'x0248': 'previous_vsp', 'x0249': 'previous_cd',
        'x0250': 'previous_evp', 'x0251': 'previous_kvp', 'x0252': 'previous_chu',
        'x0253': 'previous_arp', 'x0254': 'previous_sdap', 'x0255': 'previous_cpn',
        'x0256': 'previous_ppr', 'x0257': 'previous_psp', 'x0258': 'previous_boerenpartij',
        'x0259': 'previous_senioren2000', 'x0260': 'previous_ds70', 'x0261': 'previous_other',
        'x0272': 'hesitate',
        'x0273': 'consider_cda', 'x0274': 'consider_lpf', 'x0275': 'consider_vvd',
        'x0276': 'consider_pvda', 'x0277': 'consider_gl', 'x0278': 'consider_sp',
        'x0279': 'consider_d66', 'x0280': 'consider_cu', 'x0281': 'consider_sgp',
        'x0282': 'consider_ln', 'x0283': 'consider_conservatieven', 'x0284': 'consider_ratelband',
        'x0285': 'consider_pvdd', 'x0286': 'consider_pvdt', 'x0287': 'consider_blanc',
        'x0288': 'consider_dk', 'x0289': 'consider_na',
        'x0404': 'prob_cda', 'x0405': 'prob_pvda', 'x0406': 'prob_vvd', 'x0407': 'prob_d66',
        'x0408': 'prob_gl', 'x0409': 'prob_sgp', 'x0410': 'prob_cu', 'x0411': 'prob_ln',
        'x0412': 'prob_sp', 'x0413': 'prob_lpf',
    },
    2006: {
        'V014': 'interested', 'V015': 'interest_score',
        'V080': 'intent_vote', 'V081': 'intent_party', 'V082': 'obliged_party',
        'V060': 'adherent', 'V061': 'convinced_adherent', 'V065': 'adherent_strength',
        'V064': 'adherent_direction', 'V063': 'adherent_to', 'V062': 'attracted',
        'V220': 'vote_last', 'V221': 'vote_last_party',
        'V073': 'satisfaction_gov', 'V074': 'satisfaction_policy', 'V695': 'satisfaction_democracy',
        'V691': 'left_right',
        'V421': 'age', 'V420': 'gender', 'V430': 'education_complete', 'V431': 'education_attend',
        'V422': 'martial', 'V433': 'employment',
        'V428': 'social_class', 'V413': 'income',
        'V411': '#household', 'V410': 'composition_household',
        'V510': 'vote', 'V512': 'vote_party',
        'V743': 'efficacy_external', 'V748': 'efficacy_internal', 'V753': 'cynicism',
        'V535': 'consider_novote', 'V536': 'hesitate',
        'V537': 'consider_cda', 'V538': 'consider_pvda', 'V539': 'consider_vvd',
        'V540': 'consider_gl', 'V541': 'consider_sp', 'V542': 'consider_d66',
        'V543': 'consider_cu', 'V544': 'consider_sgp', 'V545': 'consider_lpf',
        'V546': 'consider_pvdd', 'V547': 'consider_eennl', 'V548': 'consider_pvv',
        'V549': 'consider_pvn',
        'V098': 'consider_blanc', 'V099': 'consider_dk',
        'V110': 'prob_pvda', 'V111': 'prob_cda', 'V112': 'prob_vvd', 'V113': 'prob_d66',
        'V114': 'prob_gl', 'V115': 'prob_sgp', 'V116': 'prob_cu', 'V117': 'prob_sp',
        'V118': 'prob_lpf', 'V119': 'prob_pvdv', 'V120': 'prob_eennl', 'V121': 'prob_pvn',
        'V572': 'novote_party',
    },
    2012: {
        'V014': 'interested',
        'V490': 'adherent', 'V492': 'convinced_adherent',
        'V493': 'adherent_to_cda', 'V494': 'adherent_to_pvda', 'V495': 'adherent_to_vvd', 'V496': 'adherent_to_gl',
        'V497': 'adherent_to_sp', 'V498': 'adherent_to_d66', 'V499': 'adherent_to_cu', 'V500': 'adherent_to_sgp',
        'V501': 'adherent_to_pvv', 'V502': 'adherent_to_pvdd', 'V503': 'adherent_to_dpk', 'V504': 'adherent_to_50plus',
        'V505': 'adherent_to_other',
        'V491': 'attracted',
        'V150': 'vote_last', 'V151': 'vote_last_party',
        'V232': 'satisfaction_democracy', 'V130': 'left_right',
        'V340': 'age', 'V341': 'gender', 'V344': 'education_complete', 'V345': 'education_attend',
        'V342': 'martial', 'V349': 'employment',
        'V333': 'social_class', 'V359': 'income',
        'V210': 'vote', 'V212': 'vote_party',
        'V243': 'efficacy_external', 'V256': 'cynicism',
        'V480': 'prob_cda', 'V481': 'prob_pvda', 'V482': 'prob_vvd', 'V483': 'prob_d66',
        'V484': 'prob_gl', 'V485': 'prob_sp', 'V486': 'prob_pvv', 'V487': 'prob_cu',
        'V488': 'prob_sgp', 'V489': 'prob_pvdd',
    },
}

# 2. LOAD RAW DATA
df1989 = fast_read(f'{BASE}\\NKO1989.sav', usecols=list(rename_questions[1989].keys()))
df1994 = fast_read(f'{BASE}\\NKO1994.sav', usecols=list(rename_questions[1994].keys()))
df1998 = fast_read(f'{BASE}\\NKO1998.sav', encoding='cp1252',
                   usecols=list(rename_questions[1998].keys()))
df2002 = fast_read(f'{BASE}\\NKO2002.sav', encoding='cp1252',
                   usecols=list(rename_questions[2002].keys()) + list(rename_questions[2003].keys()))
df2006 = fast_read(f'{BASE}\\NKO2006(1).sav', encoding='cp1252',
                   usecols=list(rename_questions[2006].keys()))
df2012 = fast_read(f'{BASE}\\NKO2012.sav', encoding='cp1252',
                   usecols=list(rename_questions[2012].keys()))

raw = {
    1989: df1989,
    1994: df1994,
    1998: df1998,
    2002: df2002,
    2003: df2002,
    2006: df2006,
    2012: df2012,
}

# 3. CUT + RENAME
all_waves = {}

for year, mapping in rename_questions.items():
    df = raw[year][list(mapping.keys())].copy()
    df = df.rename(columns=mapping)
    df['year'] = year

    df = df.loc[:, ~df.columns.duplicated(keep='first')]

    all_waves[year] = df

print("Loaded waves:")
for yr, df in all_waves.items():
    print(f"  {yr}: {df.shape[0]} respondents, {df.shape[1]} vars")


# 4. ENCODING
MISSING_CODES = set(range(990, 1000))

PARTY_MAP = {
    1:1,2:2,3:3,4:4,5:5,6:6,
    7:7,8:8,9:9,10:10,11:10,
    12:11,13:9,14:12,
    24:13,25:14,
    35:35,36:36,
    0:0
}
UNKNOWN_PARTY = 99

BINARY_VARS = [
    'interested','intent_vote','adherent','convinced_adherent',
    'attracted','consider_novote','hesitate','vote_last','vote'
]

PARTY_VARS = [
    'vote_party','intent_party','vote_last_party','adherent_to',
    'attracted_to','adherent_direction','novote_party',
    'obliged_party','previous_vote_party','consider_party'
]

ADHERENT_STRENGTH_MAP = {0:0,1:0,2:1,4:2,6:3,7:4}

def recode_missing(df):
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].mask(df[col].isin(MISSING_CODES), np.nan)
    return df

def encode_party(val):
    if pd.isna(val):
        return np.nan
    return PARTY_MAP.get(int(val), UNKNOWN_PARTY)

def encode_wave(df):
    df = recode_missing(df.copy())

    for col in BINARY_VARS:
        if col in df:
            df[col] = df[col].map({1.0:1, 2.0:0})

    for col in PARTY_VARS:
        if col in df:
            df[col] = df[col].apply(encode_party)

    if 'adherent_strength' in df:
        df['adherent_strength'] = df['adherent_strength'].map(ADHERENT_STRENGTH_MAP)

    return df

for year in all_waves:
    all_waves[year] = encode_wave(all_waves[year])

# 5. COMBINE WAVES
combined = pd.concat(all_waves.values(), ignore_index=True).copy()

# 6. PARTY MAPPING
NKO_PARTY_TO_ELEC = {
    1:'PvdA',2:'CDA',3:'VVD',4:'D66',5:'GroenLinks',
    6:'SGP',7:'ChristenUnie',8:'ChristenUnie',
    12:'ChristenUnie',11:'SP',
    14:'LPF',13:'LN',35:'PvdD',36:'PVV'
}

# Keep only known parties
combined = combined[combined['vote_party'].isin(NKO_PARTY_TO_ELEC.keys())].copy()

# Create party_name
combined = combined.assign(
    party_name=combined['vote_party'].map(NKO_PARTY_TO_ELEC)
)

# 7. FEATURE SELECTION
SURVEY_FEATURES = [
    'interest_score','satisfaction_gov','satisfaction_policy',
    'satisfaction_democracy','left_right',
    'efficacy_internal','efficacy_external',
    'cynicism','adherent_strength',
    'education_complete','income','age'
]

prob_cols = [c for c in combined.columns if c.startswith('prob_')]
candidate_cols = [c for c in SURVEY_FEATURES + prob_cols if c in combined.columns]

coverage = combined[candidate_cols].notna().mean()
feature_cols = coverage[coverage >= 0.30].index.tolist()

print(f"Features kept: {len(feature_cols)}")

# 8. AGGREGATE
nko_agg = (
    combined
    .groupby(['year','party_name'])[feature_cols]
    .mean()
    .reset_index()
)

nko_lookup = {
    (int(r['year']), r['party_name']):
        np.nan_to_num(r[feature_cols].values.astype(np.float32))
    for _, r in nko_agg.iterrows()
}

NKO_FEATURE_DIM = len(feature_cols)
NKO_YEARS = sorted(int(y) for y in combined['year'].unique())

print(f"NKO feature dim: {NKO_FEATURE_DIM}")
print(f"NKO years: {NKO_YEARS}")


# 9. ACCESS FUNCTION
def get_nko_vector(timestamp, party_name):
    year = pd.Timestamp(timestamp, unit='s').year
    valid = [y for y in NKO_YEARS if y <= year]
    if not valid:
        return np.zeros(NKO_FEATURE_DIM, dtype=np.float32)
    return nko_lookup.get((max(valid), party_name),
                          np.zeros(NKO_FEATURE_DIM, dtype=np.float32))