# ══════════════════════════════════════════════════════════════════
# CELDA 1 — Imports
# ══════════════════════════════════════════════════════════════════
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

print("✅ Librerías cargadas")
print("TensorFlow:", tf.__version__)


# ══════════════════════════════════════════════════════════════════
# CELDA 2 — Carga de datos
# ══════════════════════════════════════════════════════════════════
def cargar_partidos():
    urls = [
        "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2122/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2021/E0.csv",
        "https://www.football-data.co.uk/mmz4281/1920/E0.csv",
    ]
    cols = ['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR','B365H','B365D','B365A']
    dfs  = []
    for url in urls:
        try:
            tmp = pd.read_csv(url)
            cols_ok = [c for c in cols if c in tmp.columns]
            dfs.append(tmp[cols_ok])
            print(f"✅ {url.split('/')[-2]}: {len(tmp)} partidos")
        except Exception as e:
            print(f"❌ {url}: {e}")
    df = pd.concat(dfs, ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.dropna(subset=['B365H','B365D','B365A','FTR'])
    return df

df = cargar_partidos()
print(f"\n📊 Total: {len(df)} partidos")
print(f"📅 Rango: {df['Date'].min().date()} → {df['Date'].max().date()}")
print(df['FTR'].value_counts())


# ══════════════════════════════════════════════════════════════════
# CELDA 3 — Feature Engineering
# ══════════════════════════════════════════════════════════════════
N_RESULTS = 10
N_H2H     = 5

def get_last_n_results(df, team, date, n=N_RESULTS):
    mask_h = (df['HomeTeam'] == team) & (df['Date'] < date)
    mask_a = (df['AwayTeam'] == team) & (df['Date'] < date)
    gh = df[mask_h][['Date','FTR']].copy()
    gh['result'] = gh['FTR'].map({'H':1,'D':0,'A':-1})
    ga = df[mask_a][['Date','FTR']].copy()
    ga['result'] = ga['FTR'].map({'A':1,'D':0,'H':-1})
    results = pd.concat([gh, ga]).sort_values('Date').tail(n)['result'].values.astype(np.float32)
    if len(results) < n:
        results = np.pad(results, (n - len(results), 0), constant_values=0)
    return results

def get_streaks(df, team, date):
    results = get_last_n_results(df, team, date)
    no_loss = no_win = 0
    for r in reversed(results):
        if r >= 0: no_loss += 1
        else: break
    for r in reversed(results):
        if r <= 0: no_win += 1
        else: break
    return float(no_loss), float(no_win)

def get_goals_stats(df, team, date, n=N_RESULTS):
    mask_h = (df['HomeTeam'] == team) & (df['Date'] < date)
    mask_a = (df['AwayTeam'] == team) & (df['Date'] < date)
    gh = df[mask_h][['Date','FTHG','FTAG']].tail(n).copy()
    gh['gf'] = gh['FTHG']; gh['gc'] = gh['FTAG']
    ga = df[mask_a][['Date','FTHG','FTAG']].tail(n).copy()
    ga['gf'] = ga['FTAG']; ga['gc'] = ga['FTHG']
    games = pd.concat([gh, ga]).sort_values('Date').tail(n)
    if len(games) == 0:
        return 0.0, 0.0
    return float(games['gf'].mean()), float(games['gc'].mean())

def precalcular_tablas(df):
    fechas_unicas = sorted(df['Date'].unique())
    tablas  = {}
    equipos = {}
    for fecha in fechas_unicas:
        tablas[fecha] = {t: dict(v) for t, v in equipos.items()}
        for _, r in df[df['Date'] == fecha].iterrows():
            home, away, ftr = r['HomeTeam'], r['AwayTeam'], r['FTR']
            for t in [home, away]:
                if t not in equipos:
                    equipos[t] = {'Pts':0,'GF':0,'GC':0}
            if ftr == 'H':
                equipos[home]['Pts'] += 3
                equipos[home]['GF'] += r['FTHG']; equipos[home]['GC'] += r['FTAG']
                equipos[away]['GF'] += r['FTAG']; equipos[away]['GC'] += r['FTHG']
            elif ftr == 'A':
                equipos[away]['Pts'] += 3
                equipos[away]['GF'] += r['FTAG']; equipos[away]['GC'] += r['FTHG']
                equipos[home]['GF'] += r['FTHG']; equipos[home]['GC'] += r['FTAG']
            else:
                equipos[home]['Pts'] += 1; equipos[away]['Pts'] += 1
                equipos[home]['GF'] += r['FTHG']; equipos[home]['GC'] += r['FTAG']
                equipos[away]['GF'] += r['FTAG']; equipos[away]['GC'] += r['FTHG']
    return tablas

def get_position(tablas, team, date):
    tabla = tablas.get(date, {})
    if len(tabla) < 2:
        return 0.5
    ordered = sorted(tabla.items(),
                     key=lambda x: (x[1]['Pts'], x[1]['GF'] - x[1]['GC']),
                     reverse=True)
    nombres = [t for t, _ in ordered]
    if team not in nombres:
        return 0.5
    return (nombres.index(team) + 1) / len(nombres)

def get_h2h(df, home, away, date, n=N_H2H):
    m1 = (df['HomeTeam']==home) & (df['AwayTeam']==away) & (df['Date']<date)
    m2 = (df['HomeTeam']==away) & (df['AwayTeam']==home) & (df['Date']<date)
    h1 = df[m1][['Date','FTHG','FTAG','FTR']].copy()
    h1['gf'] = h1['FTHG']; h1['gc'] = h1['FTAG']
    h1['res'] = h1['FTR'].map({'H':1,'D':0,'A':-1})
    h2 = df[m2][['Date','FTHG','FTAG','FTR']].copy()
    h2['gf'] = h2['FTAG']; h2['gc'] = h2['FTHG']
    h2['res'] = h2['FTR'].map({'A':1,'D':0,'H':-1})
    h2h = pd.concat([h1, h2]).sort_values('Date').tail(n)
    if len(h2h) == 0:
        return 0.0, 0.0, 0.0
    return float(h2h['gf'].mean()), float(h2h['gc'].mean()), float(h2h['res'].mean())

def get_rest_days(df, team, date):
    mask_h = (df['HomeTeam'] == team) & (df['Date'] < date)
    mask_a = (df['AwayTeam'] == team) & (df['Date'] < date)
    fechas = pd.concat([df[mask_h]['Date'], df[mask_a]['Date']])
    if len(fechas) == 0:
        return 7.0
    return float((date - fechas.max()).days)

def implied_probs(row):
    raw = np.array([1/row['B365H'], 1/row['B365D'], 1/row['B365A']])
    return raw / raw.sum()

# Construir dataset
print("⏳ Precalculando tablas...")
tablas = precalcular_tablas(df)

print("⏳ Construyendo features (1-2 min)...")
seq_home_list, seq_away_list, stats_list, odds_list = [], [], [], []

for _, row in df.iterrows():
    home, away, date = row['HomeTeam'], row['AwayTeam'], row['Date']

    seq_h             = get_last_n_results(df, home, date)
    seq_a             = get_last_n_results(df, away, date)
    gf_h, gc_h        = get_goals_stats(df, home, date)
    gf_a, gc_a        = get_goals_stats(df, away, date)
    nl_h, nw_h        = get_streaks(df, home, date)
    nl_a, nw_a        = get_streaks(df, away, date)
    pos_h             = get_position(tablas, home, date)
    pos_a             = get_position(tablas, away, date)
    h2h_gf, h2h_gc, h2h_res = get_h2h(df, home, away, date)
    rest_h            = get_rest_days(df, home, date)
    rest_a            = get_rest_days(df, away, date)
    p_h, p_d, p_a     = implied_probs(row)

    seq_home_list.append(seq_h)
    seq_away_list.append(seq_a)
    stats_list.append([
        gf_h, gc_h, gf_a, gc_a,
        nl_h, nw_h, nl_a, nw_a,
        pos_h, pos_a,
        h2h_gf, h2h_gc, h2h_res,
        rest_h, rest_a,
        p_h, p_d, p_a
    ])
    odds_list.append([row['B365H'], row['B365D'], row['B365A']])

X_seq_home = np.array(seq_home_list).reshape(-1, N_RESULTS, 1)
X_seq_away = np.array(seq_away_list).reshape(-1, N_RESULTS, 1)
X_stats    = np.array(stats_list, dtype=np.float32)
odds_array = np.array(odds_list, dtype=np.float32)
y          = (df['FTR'] == 'H').astype(int).values
N_STATS    = X_stats.shape[1]

print(f"✅ Features construidas")
print(f"   X_seq_home : {X_seq_home.shape}")
print(f"   X_seq_away : {X_seq_away.shape}")
print(f"   X_stats    : {X_stats.shape}")
print(f"   y (binario): {y.shape} | Gana local: {y.mean()*100:.1f}%")


# ══════════════════════════════════════════════════════════════════
# CELDA 4 — Modelo
# ══════════════════════════════════════════════════════════════════
def build_model(n_seq=N_RESULTS, n_stats=N_STATS):
    inp_home  = tf.keras.Input(shape=(n_seq, 1), name='home_seq')
    lstm_home = layers.LSTM(32, name='lstm_home')(inp_home)

    inp_away  = tf.keras.Input(shape=(n_seq, 1), name='away_seq')
    lstm_away = layers.LSTM(32, name='lstm_away')(inp_away)

    inp_stats = tf.keras.Input(shape=(n_stats,), name='stats')
    d1        = layers.Dense(32, activation='relu')(inp_stats)

    combined  = layers.concatenate([lstm_home, lstm_away, d1])
    x         = layers.Dense(64, activation='relu')(combined)
    x         = layers.Dropout(0.3)(x)
    x         = layers.Dense(32, activation='relu')(x)
    out       = layers.Dense(1, activation='sigmoid', name='resultado')(x)

    model = Model(inputs=[inp_home, inp_away, inp_stats], outputs=out)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

build_model().summary()


# ══════════════════════════════════════════════════════════════════
# CELDA 5 — Walk-Forward Validation
# ══════════════════════════════════════════════════════════════════
MIN_TRAIN = 760
TEST_SIZE = 190

all_y_true, all_y_pred, all_y_proba, all_odds_test = [], [], [], []

folds_posibles = (len(y) - MIN_TRAIN) // TEST_SIZE
print(f"Total partidos : {len(y)}")
print(f"Folds posibles : {folds_posibles}")
print(f"\n{'='*60}")
print(f"WALK-FORWARD ({folds_posibles} folds)")
print(f"{'='*60}")

for fold in range(folds_posibles):
    train_end  = MIN_TRAIN + fold * TEST_SIZE
    test_start = train_end
    test_end   = test_start + TEST_SIZE

    if test_end > len(y):
        break

    Xh_tr = X_seq_home[:train_end]
    Xa_tr = X_seq_away[:train_end]
    Xs_tr = X_stats[:train_end]
    y_tr  = y[:train_end]

    Xh_te = X_seq_home[test_start:test_end]
    Xa_te = X_seq_away[test_start:test_end]
    Xs_te = X_stats[test_start:test_end]
    y_te  = y[test_start:test_end]
    od_te = odds_array[test_start:test_end]

    print(f"\nFold {fold+1}: train={len(y_tr)} | test={len(y_te)}")

    cw      = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    cw_dict = dict(enumerate(cw))

    model = build_model()
    model.fit(
        x=[Xh_tr, Xa_tr, Xs_tr],
        y=y_tr,
        validation_split=0.15,
        epochs=100,
        batch_size=32,
        callbacks=[EarlyStopping(monitor='val_loss', patience=15,
                                 restore_best_weights=True, verbose=0)],
        class_weight=cw_dict,
        verbose=0
    )

    proba = model.predict([Xh_te, Xa_te, Xs_te], verbose=0).flatten()
    preds = (proba >= 0.5).astype(int)
    acc   = np.mean(preds == y_te)

    all_y_true.extend(y_te)
    all_y_pred.extend(preds)
    all_y_proba.extend(proba)
    all_odds_test.extend(od_te)

    print(f"  Accuracy fold {fold+1}: {acc:.3f}")

all_y_true  = np.array(all_y_true)
all_y_pred  = np.array(all_y_pred)
all_y_proba = np.array(all_y_proba)
all_odds    = np.array(all_odds_test)

print(f"\n✅ Evaluados {len(all_y_true)} partidos en total")


# ══════════════════════════════════════════════════════════════════
# CELDA 6 — Métricas
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("MÉTRICAS FINALES — modelo binario")
print(f"{'='*60}")

y_pred_bin = (all_y_proba >= 0.5).astype(int)

print(classification_report(
    all_y_true, y_pred_bin,
    target_names=['No gana local', 'Gana local']
))

auc   = roc_auc_score(all_y_true, all_y_proba)
brier = np.mean((all_y_proba - all_y_true) ** 2)

print(f"AUC         : {auc:.4f}  (random=0.50, perfecto=1.0)")
print(f"Brier Score : {brier:.4f}  (random≈0.25, perfecto=0.0)")
print(f"Accuracy    : {np.mean(all_y_true == y_pred_bin):.3f}")

cm = confusion_matrix(all_y_true, y_pred_bin)
print("\nMatriz de confusión:")
print(pd.DataFrame(cm,
      index=['No gana','Gana local'],
      columns=['Pred: No gana','Pred: Gana']))


# ══════════════════════════════════════════════════════════════════
# CELDA 7 — Backtest Bet365
# ══════════════════════════════════════════════════════════════════
HOUSE_MARGIN = 0.06
STAKE_FIJO   = 10.0

def backtest_bet365(y_true, y_proba, odds,
                    edge_threshold=0.05,
                    min_odd=1.5, max_odd=5.0,
                    bankroll_ini=100.0):
    bankroll = bankroll_ini
    bets = wins = 0
    wagered = profit = 0.0
    bankroll_history = [bankroll_ini]

    for i in range(len(y_true)):
        prob_home = float(y_proba[i])
        odd_home  = odds[i][0]

        if odd_home < min_odd or odd_home > max_odd:
            continue

        prob_mkt  = 1.0 / odd_home
        edge_real = prob_home - prob_mkt - HOUSE_MARGIN / 3

        if edge_real > edge_threshold:
            gano     = (y_true[i] == 1)
            profit_i = (odd_home - 1) * STAKE_FIJO if gano else -STAKE_FIJO

            bankroll += profit_i
            profit   += profit_i
            wagered  += STAKE_FIJO
            bets     += 1
            bankroll_history.append(bankroll)
            if gano:
                wins += 1

    if bets == 0:
        return None, []
    return {
        'bets':     bets,
        'win_rate': wins / bets * 100,
        'roi':      profit / wagered * 100,
        'profit':   profit,
        'bankroll': bankroll
    }, bankroll_history

estrategias_bet365 = [
    ("Edge  5%", 0.05, 1.5, 5.0),
    ("Edge  8%", 0.08, 1.5, 5.0),
    ("Edge 10%", 0.10, 1.5, 5.0),
    ("Edge 12%", 0.12, 1.5, 4.5),
    ("Edge 15%", 0.15, 1.8, 4.0),
]

print(f"\n{'='*60}")
print(f"BACKTEST BET365 — Apuesta fija ${STAKE_FIJO}")
print(f"{'='*60}")

for nombre, edge, min_o, max_o in estrategias_bet365:
    res, _ = backtest_bet365(
        all_y_true, all_y_proba, all_odds,
        edge_threshold=edge, min_odd=min_o, max_odd=max_o
    )
    if res:
        print(f"\n{nombre}")
        print(f"  Apuestas: {res['bets']:4d} | Win: {res['win_rate']:.1f}% | "
              f"ROI: {res['roi']:+.2f}% | Profit: ${res['profit']:+.2f} | "
              f"Bankroll: ${res['bankroll']:.2f}")
    else:
        print(f"\n{nombre}: Sin apuestas")


# ══════════════════════════════════════════════════════════════════
# CELDA 8 — Backtest Polymarket
# ══════════════════════════════════════════════════════════════════
def backtest_polymarket(y_true, y_proba, odds,
                        edge_threshold=0.05,
                        min_precio=0.30,
                        max_precio=0.80,
                        bankroll_ini=100.0):
    MARGEN_POLYMARKET = 0.02
    bankroll = bankroll_ini
    bets = wins = 0
    wagered = profit = 0.0
    bankroll_history = [bankroll_ini]

    for i in range(len(y_true)):
        prob_modelo = float(y_proba[i])

        raw_h = 1 / odds[i][0]
        raw_d = 1 / odds[i][1]
        raw_a = 1 / odds[i][2]
        precio_poly = raw_h / (raw_h + raw_d + raw_a)

        if precio_poly < min_precio or precio_poly > max_precio:
            continue

        edge_real = prob_modelo - precio_poly - MARGEN_POLYMARKET

        if edge_real > edge_threshold:
            ganancia_si_gana  = STAKE_FIJO * (1 - precio_poly) / precio_poly
            gano     = (y_true[i] == 1)
            profit_i = ganancia_si_gana if gano else -STAKE_FIJO

            bankroll += profit_i
            profit   += profit_i
            wagered  += STAKE_FIJO
            bets     += 1
            bankroll_history.append(bankroll)
            if gano:
                wins += 1

    if bets == 0:
        return None, []
    return {
        'bets':     bets,
        'win_rate': wins / bets * 100,
        'roi':      profit / wagered * 100,
        'profit':   profit,
        'bankroll': bankroll
    }, bankroll_history

estrategias_poly = [
    ("Edge  3%", 0.03, 0.30, 0.80),
    ("Edge  5%", 0.05, 0.30, 0.80),
    ("Edge  8%", 0.08, 0.30, 0.75),
    ("Edge 10%", 0.10, 0.35, 0.70),
]

print(f"\n{'='*60}")
print(f"BACKTEST POLYMARKET — Apuesta fija ${STAKE_FIJO} | Margen 2%")
print(f"{'='*60}")

for nombre, edge, min_p, max_p in estrategias_poly:
    res, _ = backtest_polymarket(
        all_y_true, all_y_proba, all_odds,
        edge_threshold=edge, min_precio=min_p, max_precio=max_p
    )
    if res:
        print(f"\n{nombre}")
        print(f"  Apuestas: {res['bets']:4d} | Win: {res['win_rate']:.1f}% | "
              f"ROI: {res['roi']:+.2f}% | Profit: ${res['profit']:+.2f} | "
              f"Bankroll: ${res['bankroll']:.2f}")
    else:
        print(f"\n{nombre}: Sin apuestas")
