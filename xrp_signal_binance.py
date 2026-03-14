#!/usr/bin/env python3
"""
XRP/USD Sell Signal Detector - Binance Edition
===============================================
Detecte les signaux de vente avant la panique et les signaux
d'achat dans les creux pour une strategie d'accumulation.

Mouvement minimum: 0.10 USD (couvre les frais Binance)
Paire: XRP/USDT sur Binance

Dependances:
    pip install python-binance pandas numpy ta-lib requests

Usage:
    python xrp_signal_binance.py
    python xrp_signal_binance.py --interval 15m --threshold 50
"""

import argparse
import sys
import time
from datetime import datetime, timezone

try:
    import numpy as np
    import pandas as pd
except ImportError:
    print("ERREUR: pip install pandas numpy")
    sys.exit(1)

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
except ImportError:
    print("ERREUR: pip install python-binance")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

MIN_MOVE_USD = 0.10          # Mouvement minimum pour couvrir les frais
SYMBOL = "XRPUSDT"           # Paire Binance
DEFAULT_INTERVAL = "15m"     # Timeframe par defaut
LOOKBACK = 300               # Nombre de bougies a recuperer

# Parametres indicateurs
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2.0
EMA_FAST = 9
EMA_MID = 21
EMA_SLOW = 50
VOL_MA_LEN = 20
VOL_SPIKE_MULT = 1.5


# =============================================================================
# CALCUL DES INDICATEURS (sans ta-lib, utilise pandas/numpy)
# =============================================================================

def calc_rsi(series, period=14):
    """Calcule le RSI."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_ema(series, period):
    """Calcule l'EMA."""
    return series.ewm(span=period, adjust=False).mean()


def calc_sma(series, period):
    """Calcule la SMA."""
    return series.rolling(window=period).mean()


def calc_macd(series, fast=12, slow=26, signal=9):
    """Calcule le MACD."""
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_bollinger(series, period=20, std=2.0):
    """Calcule les bandes de Bollinger."""
    sma = calc_sma(series, period)
    std_dev = series.rolling(window=period).std()
    upper = sma + std * std_dev
    lower = sma - std * std_dev
    return upper, sma, lower


def calc_atr(high, low, close, period=14):
    """Calcule l'ATR."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calc_stoch_rsi(rsi, period=14, smooth_k=3):
    """Calcule le Stochastic RSI."""
    min_rsi = rsi.rolling(window=period).min()
    max_rsi = rsi.rolling(window=period).max()
    stoch = (rsi - min_rsi) / (max_rsi - min_rsi) * 100
    k = calc_sma(stoch, smooth_k)
    d = calc_sma(k, smooth_k)
    return k, d


def calc_obv(close, volume):
    """Calcule l'OBV."""
    sign = np.sign(close.diff())
    obv = (sign * volume).cumsum()
    return obv


# =============================================================================
# RECUPERATION DONNEES BINANCE
# =============================================================================

def get_binance_data(api_key="", api_secret="", interval=DEFAULT_INTERVAL):
    """
    Recupere les donnees XRP/USDT depuis Binance.
    Fonctionne sans cle API pour les donnees publiques.
    """
    try:
        client = Client(api_key, api_secret)
        klines = client.get_klines(
            symbol=SYMBOL,
            interval=interval,
            limit=LOOKBACK
        )
    except BinanceAPIException as e:
        print(f"Erreur Binance API: {e}")
        return None
    except Exception as e:
        print(f"Erreur connexion: {e}")
        print("Tentative sans authentification...")
        client = Client("", "")
        klines = client.get_klines(
            symbol=SYMBOL,
            interval=interval,
            limit=LOOKBACK
        )

    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    return df


# =============================================================================
# ANALYSE ET SCORING
# =============================================================================

def analyze_signals(df):
    """
    Analyse les donnees et calcule les scores de vente/achat.
    Retourne le dataframe enrichi + scores actuels.
    """
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # --- Indicateurs ---
    df['rsi'] = calc_rsi(close, RSI_PERIOD)
    df['macd'], df['macd_signal'], df['macd_hist'] = calc_macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    df['bb_upper'], df['bb_mid'], df['bb_lower'] = calc_bollinger(close, BB_PERIOD, BB_STD)
    df['ema_fast'] = calc_ema(close, EMA_FAST)
    df['ema_mid'] = calc_ema(close, EMA_MID)
    df['ema_slow'] = calc_ema(close, EMA_SLOW)
    df['atr'] = calc_atr(high, low, close, 14)
    df['vol_ma'] = calc_sma(volume, VOL_MA_LEN)
    df['vol_ratio'] = volume / df['vol_ma']
    df['stoch_k'], df['stoch_d'] = calc_stoch_rsi(df['rsi'])
    df['obv'] = calc_obv(close, volume)
    df['obv_ema'] = calc_ema(df['obv'], 21)

    # --- Scoring SELL ---
    df['sell_score'] = 0.0

    # RSI surachat (max 20)
    df.loc[df['rsi'] > RSI_OVERBOUGHT, 'sell_score'] += np.minimum(
        (df['rsi'] - RSI_OVERBOUGHT) * 1.5, 20
    )

    # MACD croisement baissier (15)
    macd_cross_down = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
    df.loc[macd_cross_down, 'sell_score'] += 15

    # MACD histogramme en baisse (5)
    macd_declining = (df['macd_hist'] < df['macd_hist'].shift(1)) & (df['macd_hist'].shift(1) < df['macd_hist'].shift(2))
    df.loc[macd_declining, 'sell_score'] += 5

    # Prix > BB superieure (10)
    df.loc[close > df['bb_upper'], 'sell_score'] += 10

    # Volume spike bougie rouge (10)
    vol_spike = df['vol_ratio'] > VOL_SPIKE_MULT
    red_candle = close < df['open']
    df.loc[vol_spike & red_candle, 'sell_score'] += 10

    # EMA croisement baissier (10)
    ema_cross_down = (df['ema_fast'] < df['ema_mid']) & (df['ema_fast'].shift(1) >= df['ema_mid'].shift(1))
    df.loc[ema_cross_down, 'sell_score'] += 10

    # Prix passe sous EMA rapide (5)
    below_ema = (close < df['ema_fast']) & (close.shift(1) > df['ema_fast'].shift(1))
    df.loc[below_ema, 'sell_score'] += 5

    # Divergence baissiere (15)
    for i in range(5, len(df)):
        if (high.iloc[i] > high.iloc[i-5:i].max() and
            df['rsi'].iloc[i] < df['rsi'].iloc[i-5:i].max() and
            df['rsi'].iloc[i] > 60):
            df.iloc[i, df.columns.get_loc('sell_score')] += 15

    # OBV en baisse prix en hausse (10)
    obv_down = (df['obv'] < df['obv_ema']) & (df['obv'] < df['obv'].shift(3))
    price_up = close > close.shift(3)
    df.loc[obv_down & price_up, 'sell_score'] += 10

    # StochRSI surachat + croisement (5)
    stoch_sell = (df['stoch_k'] > 80) & (df['stoch_k'] < df['stoch_d']) & (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))
    df.loc[stoch_sell, 'sell_score'] += 5

    df['sell_score'] = df['sell_score'].clip(upper=100)

    # --- Scoring BUY ---
    df['buy_score'] = 0.0

    # RSI survente (max 20)
    df.loc[df['rsi'] < RSI_OVERSOLD, 'buy_score'] += np.minimum(
        (RSI_OVERSOLD - df['rsi']) * 1.5, 20
    )

    # MACD croisement haussier (15)
    macd_cross_up = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    df.loc[macd_cross_up, 'buy_score'] += 15

    # MACD histogramme en hausse (5)
    macd_rising = (df['macd_hist'] > df['macd_hist'].shift(1)) & (df['macd_hist'].shift(1) > df['macd_hist'].shift(2))
    df.loc[macd_rising, 'buy_score'] += 5

    # Prix < BB inferieure (10)
    df.loc[close < df['bb_lower'], 'buy_score'] += 10

    # Volume spike bougie verte (10)
    green_candle = close > df['open']
    df.loc[vol_spike & green_candle, 'buy_score'] += 10

    # EMA croisement haussier (10)
    ema_cross_up = (df['ema_fast'] > df['ema_mid']) & (df['ema_fast'].shift(1) <= df['ema_mid'].shift(1))
    df.loc[ema_cross_up, 'buy_score'] += 10

    # Prix repasse au dessus EMA (5)
    above_ema = (close > df['ema_fast']) & (close.shift(1) < df['ema_fast'].shift(1))
    df.loc[above_ema, 'buy_score'] += 5

    # Divergence haussiere (15)
    for i in range(5, len(df)):
        if (low.iloc[i] < low.iloc[i-5:i].min() and
            df['rsi'].iloc[i] > df['rsi'].iloc[i-5:i].min() and
            df['rsi'].iloc[i] < 40):
            df.iloc[i, df.columns.get_loc('buy_score')] += 15

    # OBV en hausse prix en baisse (10)
    obv_up = (df['obv'] > df['obv_ema']) & (df['obv'] > df['obv'].shift(3))
    price_down = close < close.shift(3)
    df.loc[obv_up & price_down, 'buy_score'] += 10

    # StochRSI survente + croisement (5)
    stoch_buy = (df['stoch_k'] < 20) & (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))
    df.loc[stoch_buy, 'buy_score'] += 5

    df['buy_score'] = df['buy_score'].clip(upper=100)

    # --- Mouvement potentiel ---
    df['potential_move'] = df['atr'] * 1.5
    df['move_ok'] = df['potential_move'] >= MIN_MOVE_USD

    return df


# =============================================================================
# AFFICHAGE RESULTATS
# =============================================================================

def print_analysis(df):
    """Affiche l'analyse actuelle."""
    last = df.iloc[-1]
    prev = df.iloc[-2]

    price = last['close']
    sell_score = last['sell_score']
    buy_score = last['buy_score']
    rsi = last['rsi']
    potential_move = last['potential_move']
    move_ok = last['move_ok']

    # Tendance EMA
    if last['ema_fast'] > last['ema_mid'] > last['ema_slow']:
        trend = "HAUSSE"
        trend_icon = "^"
    elif last['ema_fast'] < last['ema_mid'] < last['ema_slow']:
        trend = "BAISSE"
        trend_icon = "v"
    else:
        trend = "NEUTRE"
        trend_icon = "-"

    # Signal principal
    if sell_score >= 70 and move_ok:
        signal = ">>> VENDRE MAINTENANT <<<"
        signal_level = "FORT"
    elif sell_score >= 50 and move_ok:
        signal = ">> SIGNAL DE VENTE <<"
        signal_level = "MOYEN"
    elif sell_score >= 40 and move_ok:
        signal = "> Vente possible <"
        signal_level = "FAIBLE"
    elif buy_score >= 70 and move_ok:
        signal = ">>> ACHETER MAINTENANT <<<"
        signal_level = "FORT"
    elif buy_score >= 50 and move_ok:
        signal = ">> SIGNAL D'ACHAT <<"
        signal_level = "MOYEN"
    elif buy_score >= 40 and move_ok:
        signal = "> Achat possible <"
        signal_level = "FAIBLE"
    else:
        signal = "-- PAS DE SIGNAL --"
        signal_level = "AUCUN"

    sell_target = price - potential_move
    buy_target = price + potential_move

    print("\n" + "=" * 60)
    print("  XRP/USDT - DETECTION DE SIGNAUX - ACCUMULATION")
    print("=" * 60)
    print(f"  Heure UTC : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Prix      : {price:.4f} USDT")
    print("-" * 60)
    print(f"\n  {signal}")
    print(f"  Force du signal : {signal_level}\n")
    print("-" * 60)
    print(f"  % Probabilite BAISSE : {sell_score:.0f}%", end="")
    if sell_score >= 70:
        print("  [!!! TRES ELEVE !!!]")
    elif sell_score >= 50:
        print("  [! ELEVE !]")
    else:
        print()

    print(f"  % Probabilite HAUSSE : {buy_score:.0f}%", end="")
    if buy_score >= 70:
        print("  [!!! TRES ELEVE !!!]")
    elif buy_score >= 50:
        print("  [! ELEVE !]")
    else:
        print()

    print("-" * 60)
    print(f"  RSI ({RSI_PERIOD})         : {rsi:.1f}", end="")
    if rsi > RSI_OVERBOUGHT:
        print("  [SURACHETE]")
    elif rsi < RSI_OVERSOLD:
        print("  [SURVENDU]")
    else:
        print("  [Neutre]")

    macd_dir = "HAUSSIER" if last['macd'] > last['macd_signal'] else "BAISSIER"
    print(f"  MACD            : {macd_dir}")
    print(f"  Tendance EMA    : {trend} {trend_icon}")
    print(f"  Volume Ratio    : {last['vol_ratio']:.2f}x", end="")
    if last['vol_ratio'] > VOL_SPIKE_MULT:
        print("  [SPIKE!]")
    else:
        print()

    print(f"  ATR             : {last['atr']:.4f}")
    print(f"  Mouvement Est.  : {potential_move:.4f} USD", end="")
    if move_ok:
        print(f"  [> {MIN_MOVE_USD} OK]")
    else:
        print(f"  [< {MIN_MOVE_USD} INSUFFISANT]")

    print("-" * 60)
    print(f"  Bollinger Sup.  : {last['bb_upper']:.4f}")
    print(f"  Bollinger Moy.  : {last['bb_mid']:.4f}")
    print(f"  Bollinger Inf.  : {last['bb_lower']:.4f}")
    print("-" * 60)
    print(f"  EMA {EMA_FAST}           : {last['ema_fast']:.4f}")
    print(f"  EMA {EMA_MID}          : {last['ema_mid']:.4f}")
    print(f"  EMA {EMA_SLOW}          : {last['ema_slow']:.4f}")
    print("-" * 60)
    print("  OBJECTIFS:")
    print(f"    Objectif VENTE  : {sell_target:.4f} (-{potential_move:.4f})")
    print(f"    Objectif RACHAT : {buy_target:.4f} (+{potential_move:.4f})")
    print("=" * 60)

    # Historique des derniers signaux
    recent = df.tail(20)
    sells = recent[recent['sell_score'] >= 40]
    buys = recent[recent['buy_score'] >= 40]

    if len(sells) > 0 or len(buys) > 0:
        print("\n  HISTORIQUE RECENT DES SIGNAUX:")
        print("-" * 60)
        for idx, row in sells.iterrows():
            level = "FORT" if row['sell_score'] >= 70 else "MOYEN" if row['sell_score'] >= 50 else "FAIBLE"
            print(f"  SELL {level:6s} | {idx} | Prix: {row['close']:.4f} | Score: {row['sell_score']:.0f}%")
        for idx, row in buys.iterrows():
            level = "FORT" if row['buy_score'] >= 70 else "MOYEN" if row['buy_score'] >= 50 else "FAIBLE"
            print(f"  BUY  {level:6s} | {idx} | Prix: {row['close']:.4f} | Score: {row['buy_score']:.0f}%")
        print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="XRP/USDT Signal Detector - Binance")
    parser.add_argument("--key", default="", help="Binance API key (optionnel pour data publique)")
    parser.add_argument("--secret", default="", help="Binance API secret")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL,
                        choices=["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"],
                        help="Timeframe (defaut: 15m)")
    parser.add_argument("--threshold", type=int, default=40,
                        help="Seuil minimum du score pour afficher les signaux (defaut: 40)")
    parser.add_argument("--loop", action="store_true",
                        help="Mode boucle - rafraichit automatiquement")
    parser.add_argument("--loop-delay", type=int, default=60,
                        help="Delai en secondes entre chaque rafraichissement (defaut: 60)")
    parser.add_argument("--min-move", type=float, default=MIN_MOVE_USD,
                        help=f"Mouvement minimum en USD (defaut: {MIN_MOVE_USD})")
    args = parser.parse_args()

    global MIN_MOVE_USD
    MIN_MOVE_USD = args.min_move

    print("\n  Demarrage XRP Signal Detector...")
    print(f"  Paire: {SYMBOL} | Timeframe: {args.interval}")
    print(f"  Mouvement minimum: {MIN_MOVE_USD} USD")
    print(f"  Seuil signal: {args.threshold}%\n")

    while True:
        try:
            df = get_binance_data(args.key, args.secret, args.interval)
            if df is None:
                print("Impossible de recuperer les donnees. Nouvelle tentative dans 30s...")
                time.sleep(30)
                continue

            df = analyze_signals(df)
            print_analysis(df)

            if not args.loop:
                break

            print(f"\n  Prochain rafraichissement dans {args.loop_delay}s... (Ctrl+C pour arreter)")
            time.sleep(args.loop_delay)

        except KeyboardInterrupt:
            print("\n\n  Arret du scanner. A bientot!")
            break
        except Exception as e:
            print(f"\n  Erreur: {e}")
            if not args.loop:
                break
            print(f"  Nouvelle tentative dans 30s...")
            time.sleep(30)


if __name__ == "__main__":
    main()
