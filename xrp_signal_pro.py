#!/usr/bin/env python3
"""
XRP Signal Detector PRO - Bare Metal Edition
=============================================
Version haute performance exploitant la puissance d'un serveur dedie:
- Multi-timeframe parallele (1m, 5m, 15m, 1h analyses simultanement)
- Websocket temps reel (reaction en millisecondes)
- Scoring croise avec confirmation multi-TF
- Architecture asyncio non-bloquante

Dependances:
    pip install python-binance pandas numpy websockets

Usage:
    # Mode analyse multi-timeframe (pas de cles requises)
    python xrp_signal_pro.py

    # Mode websocket temps reel
    python xrp_signal_pro.py --realtime

    # Multi-timeframe + websocket + boucle
    python xrp_signal_pro.py --realtime --loop

    # Trading reel
    python xrp_signal_pro.py --realtime --trade --key CLE --secret SECRET --loop

    # Choisir les timeframes
    python xrp_signal_pro.py --timeframes 1m 5m 15m 1h 4h
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN

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

# Websocket optionnel (fallback sur polling si absent)
try:
    from binance import ThreadedWebsocketManager
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False
    print("  [INFO] python-binance websocket non disponible, mode polling uniquement")


# =============================================================================
# CONFIGURATION
# =============================================================================

SYMBOL = "XRPUSDT"
MIN_MOVE_USD = 0.10
LOOKBACK = 300

# Indicateurs
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

# Multi-timeframe: poids de chaque TF dans le score final
TIMEFRAME_WEIGHTS = {
    '1m':  0.05,   # Bruit, mais utile pour timing d'entree
    '3m':  0.08,
    '5m':  0.15,
    '15m': 0.30,   # TF principal
    '30m': 0.15,
    '1h':  0.20,   # Confirmation tendance
    '2h':  0.04,
    '4h':  0.03,
}

# Timeframes par defaut
DEFAULT_TIMEFRAMES = ['1m', '5m', '15m', '1h']

# Fichiers
TRADE_LOG_FILE = "xrp_trades_pro.log"
SIGNAL_HISTORY_FILE = "xrp_signal_history_pro.json"

# Logging
trade_logger = logging.getLogger("xrp_trader_pro")
trade_logger.setLevel(logging.INFO)


# =============================================================================
# CALCUL DES INDICATEURS (vectorise avec numpy)
# =============================================================================

def calc_rsi(series, period=14):
    """RSI vectorise."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calc_sma(series, period):
    return series.rolling(window=period).mean()


def calc_macd(series, fast=12, slow=26, signal_period=9):
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_bollinger(series, period=20, std=2.0):
    sma = calc_sma(series, period)
    std_dev = series.rolling(window=period).std()
    upper = sma + std * std_dev
    lower = sma - std * std_dev
    return upper, sma, lower


def calc_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calc_stoch_rsi(rsi, period=14, smooth_k=3):
    min_rsi = rsi.rolling(window=period).min()
    max_rsi = rsi.rolling(window=period).max()
    stoch = (rsi - min_rsi) / (max_rsi - min_rsi) * 100
    k = calc_sma(stoch, smooth_k)
    d = calc_sma(k, smooth_k)
    return k, d


def calc_obv(close, volume):
    sign = np.sign(close.diff())
    return (sign * volume).cumsum()


# =============================================================================
# DIVERGENCE VECTORISEE (remplace la boucle O(n))
# =============================================================================

def detect_divergences_vectorized(df):
    """
    Detecte les divergences prix/RSI de maniere vectorisee.
    ~50x plus rapide que la boucle for originale.
    """
    high = df['high']
    low = df['low']
    rsi = df['rsi']

    # Rolling max/min sur 5 periodes
    high_max_5 = high.rolling(5).max()
    low_min_5 = low.rolling(5).min()
    rsi_max_5 = rsi.rolling(5).max()
    rsi_min_5 = rsi.rolling(5).min()

    # Divergence baissiere: prix fait un higher high mais RSI fait un lower high
    bearish_div = (high > high_max_5.shift(1)) & (rsi < rsi_max_5.shift(1)) & (rsi > 60)

    # Divergence haussiere: prix fait un lower low mais RSI fait un higher low
    bullish_div = (low < low_min_5.shift(1)) & (rsi > rsi_min_5.shift(1)) & (rsi < 40)

    return bearish_div, bullish_div


# =============================================================================
# ANALYSE ET SCORING (une seule timeframe)
# =============================================================================

def analyze_signals(df):
    """Analyse les donnees et calcule les scores. Version optimisee."""
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # Indicateurs
    df['rsi'] = calc_rsi(close, RSI_PERIOD)
    df['macd'], df['macd_signal'], df['macd_hist'] = calc_macd(
        close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    df['bb_upper'], df['bb_mid'], df['bb_lower'] = calc_bollinger(
        close, BB_PERIOD, BB_STD)
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
        (df['rsi'] - RSI_OVERBOUGHT) * 1.5, 20)

    # MACD croisement baissier (15)
    macd_cross_down = ((df['macd'] < df['macd_signal']) &
                       (df['macd'].shift(1) >= df['macd_signal'].shift(1)))
    df.loc[macd_cross_down, 'sell_score'] += 15

    # MACD histogramme en baisse (5)
    macd_declining = ((df['macd_hist'] < df['macd_hist'].shift(1)) &
                      (df['macd_hist'].shift(1) < df['macd_hist'].shift(2)))
    df.loc[macd_declining, 'sell_score'] += 5

    # Prix > BB superieure (10)
    df.loc[close > df['bb_upper'], 'sell_score'] += 10

    # Volume spike bougie rouge (10)
    vol_spike = df['vol_ratio'] > VOL_SPIKE_MULT
    red_candle = close < df['open']
    df.loc[vol_spike & red_candle, 'sell_score'] += 10

    # EMA croisement baissier (10)
    ema_cross_down = ((df['ema_fast'] < df['ema_mid']) &
                      (df['ema_fast'].shift(1) >= df['ema_mid'].shift(1)))
    df.loc[ema_cross_down, 'sell_score'] += 10

    # Prix passe sous EMA rapide (5)
    below_ema = ((close < df['ema_fast']) &
                 (close.shift(1) > df['ema_fast'].shift(1)))
    df.loc[below_ema, 'sell_score'] += 5

    # Divergences vectorisees (15)
    bearish_div, bullish_div = detect_divergences_vectorized(df)
    df.loc[bearish_div, 'sell_score'] += 15

    # OBV en baisse prix en hausse (10)
    obv_down = (df['obv'] < df['obv_ema']) & (df['obv'] < df['obv'].shift(3))
    price_up = close > close.shift(3)
    df.loc[obv_down & price_up, 'sell_score'] += 10

    # StochRSI surachat + croisement (5)
    stoch_sell = ((df['stoch_k'] > 80) & (df['stoch_k'] < df['stoch_d']) &
                  (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1)))
    df.loc[stoch_sell, 'sell_score'] += 5

    df['sell_score'] = df['sell_score'].clip(upper=100)

    # --- Scoring BUY ---
    df['buy_score'] = 0.0

    # RSI survente (max 20)
    df.loc[df['rsi'] < RSI_OVERSOLD, 'buy_score'] += np.minimum(
        (RSI_OVERSOLD - df['rsi']) * 1.5, 20)

    # MACD croisement haussier (15)
    macd_cross_up = ((df['macd'] > df['macd_signal']) &
                     (df['macd'].shift(1) <= df['macd_signal'].shift(1)))
    df.loc[macd_cross_up, 'buy_score'] += 15

    # MACD histogramme en hausse (5)
    macd_rising = ((df['macd_hist'] > df['macd_hist'].shift(1)) &
                   (df['macd_hist'].shift(1) > df['macd_hist'].shift(2)))
    df.loc[macd_rising, 'buy_score'] += 5

    # Prix < BB inferieure (10)
    df.loc[close < df['bb_lower'], 'buy_score'] += 10

    # Volume spike bougie verte (10)
    green_candle = close > df['open']
    df.loc[vol_spike & green_candle, 'buy_score'] += 10

    # EMA croisement haussier (10)
    ema_cross_up = ((df['ema_fast'] > df['ema_mid']) &
                    (df['ema_fast'].shift(1) <= df['ema_mid'].shift(1)))
    df.loc[ema_cross_up, 'buy_score'] += 10

    # Prix repasse au dessus EMA (5)
    above_ema = ((close > df['ema_fast']) &
                 (close.shift(1) < df['ema_fast'].shift(1)))
    df.loc[above_ema, 'buy_score'] += 5

    # Divergences haussiere (15)
    df.loc[bullish_div, 'buy_score'] += 15

    # OBV en hausse prix en baisse (10)
    obv_up = (df['obv'] > df['obv_ema']) & (df['obv'] > df['obv'].shift(3))
    price_down = close < close.shift(3)
    df.loc[obv_up & price_down, 'buy_score'] += 10

    # StochRSI survente + croisement (5)
    stoch_buy = ((df['stoch_k'] < 20) & (df['stoch_k'] > df['stoch_d']) &
                 (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1)))
    df.loc[stoch_buy, 'buy_score'] += 5

    df['buy_score'] = df['buy_score'].clip(upper=100)

    # Mouvement potentiel
    df['potential_move'] = df['atr'] * 1.5
    df['move_ok'] = df['potential_move'] >= MIN_MOVE_USD

    return df


# =============================================================================
# MULTI-TIMEFRAME ENGINE
# =============================================================================

class MultiTimeframeEngine:
    """
    Moteur multi-timeframe parallele.
    Fetch et analyse N timeframes simultanement via ThreadPool.
    """

    def __init__(self, api_key="", api_secret="", timeframes=None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.timeframes = timeframes or DEFAULT_TIMEFRAMES
        self.results = {}  # tf -> {'df': DataFrame, 'score': {...}}
        self.executor = ThreadPoolExecutor(
            max_workers=len(self.timeframes),
            thread_name_prefix="tf_worker"
        )

    def _fetch_and_analyze(self, interval):
        """Fetch + analyse pour un seul timeframe (execute dans un thread)."""
        try:
            client = Client(self.api_key, self.api_secret)
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

            df = analyze_signals(df)
            last = df.iloc[-1]

            return interval, {
                'df': df,
                'sell_score': last['sell_score'],
                'buy_score': last['buy_score'],
                'rsi': last['rsi'],
                'macd_dir': 'HAUSSIER' if last['macd'] > last['macd_signal'] else 'BAISSIER',
                'trend': self._get_trend(last),
                'price': last['close'],
                'atr': last['atr'],
                'potential_move': last['potential_move'],
                'move_ok': last['move_ok'],
                'vol_ratio': last['vol_ratio'],
                'ema_fast': last['ema_fast'],
                'ema_mid': last['ema_mid'],
                'ema_slow': last['ema_slow'],
            }
        except Exception as e:
            return interval, {'error': str(e)}

    def _get_trend(self, row):
        if row['ema_fast'] > row['ema_mid'] > row['ema_slow']:
            return "HAUSSE"
        elif row['ema_fast'] < row['ema_mid'] < row['ema_slow']:
            return "BAISSE"
        return "NEUTRE"

    def analyze_all(self):
        """
        Lance l'analyse de tous les TF en parallele.
        Retourne les resultats combines.
        """
        start_time = time.time()
        futures = {
            self.executor.submit(self._fetch_and_analyze, tf): tf
            for tf in self.timeframes
        }

        self.results = {}
        for future in futures:
            tf, result = future.result(timeout=30)
            self.results[tf] = result

        elapsed = time.time() - start_time

        # Calculer le score combine
        combined = self._combine_scores()
        combined['fetch_time_ms'] = int(elapsed * 1000)

        return combined

    def _combine_scores(self):
        """
        Combine les scores de tous les TF avec ponderation.
        Applique un bonus de confirmation quand plusieurs TF sont d'accord.
        """
        total_weight = 0
        weighted_sell = 0
        weighted_buy = 0
        confirmations_sell = 0
        confirmations_buy = 0
        tf_details = {}

        for tf in self.timeframes:
            result = self.results.get(tf, {})
            if 'error' in result:
                tf_details[tf] = {'error': result['error']}
                continue

            weight = TIMEFRAME_WEIGHTS.get(tf, 0.1)
            total_weight += weight
            weighted_sell += result['sell_score'] * weight
            weighted_buy += result['buy_score'] * weight

            # Compter les confirmations (score >= 40)
            if result['sell_score'] >= 40:
                confirmations_sell += 1
            if result['buy_score'] >= 40:
                confirmations_buy += 1

            tf_details[tf] = {
                'sell_score': result['sell_score'],
                'buy_score': result['buy_score'],
                'rsi': result['rsi'],
                'macd': result['macd_dir'],
                'trend': result['trend'],
                'weight': weight,
            }

        # Normaliser
        if total_weight > 0:
            combined_sell = weighted_sell / total_weight
            combined_buy = weighted_buy / total_weight
        else:
            combined_sell = 0
            combined_buy = 0

        # Bonus de confirmation multi-TF
        n_tf = len([tf for tf in self.timeframes if tf in self.results and 'error' not in self.results[tf]])
        if n_tf > 0:
            # Bonus: +10% par TF confirme au dela du premier
            sell_confirmation_bonus = max(0, confirmations_sell - 1) * 5
            buy_confirmation_bonus = max(0, confirmations_buy - 1) * 5
            combined_sell = min(100, combined_sell + sell_confirmation_bonus)
            combined_buy = min(100, combined_buy + buy_confirmation_bonus)

        # Prendre le TF principal (15m ou le plus eleve disponible) pour les infos de prix
        primary_tf = '15m' if '15m' in self.results else self.timeframes[0]
        primary = self.results.get(primary_tf, {})

        return {
            'combined_sell': combined_sell,
            'combined_buy': combined_buy,
            'confirmations_sell': confirmations_sell,
            'confirmations_buy': confirmations_buy,
            'total_tf': n_tf,
            'tf_details': tf_details,
            'price': primary.get('price', 0),
            'rsi': primary.get('rsi', 0),
            'atr': primary.get('atr', 0),
            'potential_move': primary.get('potential_move', 0),
            'move_ok': primary.get('move_ok', False),
            'trend': primary.get('trend', 'N/A'),
            'ema_fast': primary.get('ema_fast', 0),
            'ema_mid': primary.get('ema_mid', 0),
            'ema_slow': primary.get('ema_slow', 0),
            'vol_ratio': primary.get('vol_ratio', 0),
            'primary_tf': primary_tf,
        }

    def shutdown(self):
        self.executor.shutdown(wait=False)


# =============================================================================
# WEBSOCKET REAL-TIME ENGINE
# =============================================================================

class RealtimeEngine:
    """
    Moteur websocket temps reel.
    Maintient un buffer local de bougies mis a jour en streaming.
    Declenche l'analyse a chaque cloture de bougie.
    """

    def __init__(self, api_key="", api_secret="", timeframes=None,
                 on_signal=None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.timeframes = timeframes or DEFAULT_TIMEFRAMES
        self.on_signal = on_signal  # callback(combined_result)

        # Buffer de bougies par TF
        self.candle_buffers = {tf: deque(maxlen=LOOKBACK) for tf in self.timeframes}
        self.current_candles = {}  # TF -> candle en cours (non fermee)
        self.initialized = {tf: False for tf in self.timeframes}
        self.twm = None
        self.running = False
        self._lock = threading.Lock()
        self._mt_engine = MultiTimeframeEngine(api_key, api_secret, timeframes)

    def _init_historical(self):
        """Charge les donnees historiques initiales pour chaque TF."""
        print("\n  [WS] Chargement des donnees historiques...")
        combined = self._mt_engine.analyze_all()

        for tf in self.timeframes:
            result = self._mt_engine.results.get(tf, {})
            if 'error' not in result and 'df' in result:
                df = result['df']
                for _, row in df.iterrows():
                    self.candle_buffers[tf].append({
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume'],
                    })
                self.initialized[tf] = True
                print(f"  [WS] {tf}: {len(self.candle_buffers[tf])} bougies chargees")

        return combined

    def _handle_kline(self, msg):
        """Callback appele a chaque update de kline websocket."""
        if msg.get('e') == 'error':
            print(f"  [WS] Erreur: {msg}")
            return

        kline = msg.get('k', {})
        tf = kline.get('i', '')
        is_closed = kline.get('x', False)

        candle = {
            'open': float(kline.get('o', 0)),
            'high': float(kline.get('h', 0)),
            'low': float(kline.get('l', 0)),
            'close': float(kline.get('c', 0)),
            'volume': float(kline.get('v', 0)),
        }

        with self._lock:
            self.current_candles[tf] = candle

            if is_closed and self.initialized.get(tf, False):
                # Bougie fermee -> ajouter au buffer
                self.candle_buffers[tf].append(candle)
                self._trigger_analysis(tf)

    def _trigger_analysis(self, closed_tf):
        """Declenche une analyse multi-TF quand une bougie se ferme."""
        try:
            tf_results = {}
            for tf in self.timeframes:
                if not self.initialized.get(tf, False):
                    continue

                # Construire le DataFrame depuis le buffer
                candles = list(self.candle_buffers[tf])
                # Ajouter la bougie en cours si differente du TF qui vient de fermer
                if tf != closed_tf and tf in self.current_candles:
                    candles = candles + [self.current_candles[tf]]

                if len(candles) < 50:
                    continue

                df = pd.DataFrame(candles)
                df = analyze_signals(df)
                last = df.iloc[-1]

                tf_results[tf] = {
                    'sell_score': last['sell_score'],
                    'buy_score': last['buy_score'],
                    'rsi': last['rsi'],
                    'macd_dir': 'HAUSSIER' if last['macd'] > last['macd_signal'] else 'BAISSIER',
                    'trend': self._get_trend(last),
                    'price': last['close'],
                    'atr': last['atr'],
                    'potential_move': last['potential_move'],
                    'move_ok': last['move_ok'],
                    'vol_ratio': last['vol_ratio'],
                    'ema_fast': last['ema_fast'],
                    'ema_mid': last['ema_mid'],
                    'ema_slow': last['ema_slow'],
                }

            if tf_results:
                combined = self._combine_results(tf_results, closed_tf)
                if self.on_signal:
                    self.on_signal(combined)

        except Exception as e:
            print(f"  [WS] Erreur analyse: {e}")

    def _get_trend(self, row):
        if row['ema_fast'] > row['ema_mid'] > row['ema_slow']:
            return "HAUSSE"
        elif row['ema_fast'] < row['ema_mid'] < row['ema_slow']:
            return "BAISSE"
        return "NEUTRE"

    def _combine_results(self, tf_results, trigger_tf):
        """Combine les resultats multi-TF (meme logique que MultiTimeframeEngine)."""
        total_weight = 0
        weighted_sell = 0
        weighted_buy = 0
        confirmations_sell = 0
        confirmations_buy = 0
        tf_details = {}

        for tf, result in tf_results.items():
            weight = TIMEFRAME_WEIGHTS.get(tf, 0.1)
            total_weight += weight
            weighted_sell += result['sell_score'] * weight
            weighted_buy += result['buy_score'] * weight

            if result['sell_score'] >= 40:
                confirmations_sell += 1
            if result['buy_score'] >= 40:
                confirmations_buy += 1

            tf_details[tf] = {
                'sell_score': result['sell_score'],
                'buy_score': result['buy_score'],
                'rsi': result['rsi'],
                'macd': result['macd_dir'],
                'trend': result['trend'],
                'weight': weight,
            }

        if total_weight > 0:
            combined_sell = weighted_sell / total_weight
            combined_buy = weighted_buy / total_weight
        else:
            combined_sell = 0
            combined_buy = 0

        sell_confirmation_bonus = max(0, confirmations_sell - 1) * 5
        buy_confirmation_bonus = max(0, confirmations_buy - 1) * 5
        combined_sell = min(100, combined_sell + sell_confirmation_bonus)
        combined_buy = min(100, combined_buy + buy_confirmation_bonus)

        primary_tf = '15m' if '15m' in tf_results else list(tf_results.keys())[0]
        primary = tf_results.get(primary_tf, {})

        return {
            'combined_sell': combined_sell,
            'combined_buy': combined_buy,
            'confirmations_sell': confirmations_sell,
            'confirmations_buy': confirmations_buy,
            'total_tf': len(tf_results),
            'tf_details': tf_details,
            'price': primary.get('price', 0),
            'rsi': primary.get('rsi', 0),
            'atr': primary.get('atr', 0),
            'potential_move': primary.get('potential_move', 0),
            'move_ok': primary.get('move_ok', False),
            'trend': primary.get('trend', 'N/A'),
            'ema_fast': primary.get('ema_fast', 0),
            'ema_mid': primary.get('ema_mid', 0),
            'ema_slow': primary.get('ema_slow', 0),
            'vol_ratio': primary.get('vol_ratio', 0),
            'primary_tf': primary_tf,
            'trigger_tf': trigger_tf,
            'realtime': True,
        }

    def start(self):
        """Demarre le streaming websocket."""
        if not HAS_WEBSOCKET:
            print("  [WS] Websocket non disponible. Utilisez pip install python-binance")
            return None

        # Charger l'historique d'abord
        initial = self._init_historical()

        # Demarrer les websockets pour chaque TF
        self.twm = ThreadedWebsocketManager(
            api_key=self.api_key or "",
            api_secret=self.api_secret or ""
        )
        self.twm.start()
        self.running = True

        for tf in self.timeframes:
            self.twm.start_kline_socket(
                callback=self._handle_kline,
                symbol=SYMBOL.lower(),
                interval=tf
            )
            print(f"  [WS] Stream {tf} demarre")

        print(f"  [WS] {len(self.timeframes)} streams actifs - mode temps reel")
        return initial

    def stop(self):
        """Arrete les websockets."""
        self.running = False
        if self.twm:
            self.twm.stop()
            print("  [WS] Streams arretes")


# =============================================================================
# SIGNAL HISTORY
# =============================================================================

def load_signal_history():
    if os.path.exists(SIGNAL_HISTORY_FILE):
        try:
            with open(SIGNAL_HISTORY_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {'signals': [], 'version': 2}


def save_signal_history(history):
    with open(SIGNAL_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2, default=str)


def record_signal(signal_type, score, price, rsi, trend, potential_move,
                  timeframes_info, traded=False, trade_result=None):
    """Enregistre un signal multi-TF dans l'historique."""
    history = load_signal_history()
    entry = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'type': signal_type,
        'score': round(score, 1),
        'price': round(price, 6),
        'rsi': round(rsi, 1),
        'trend': trend,
        'potential_move': round(potential_move, 6),
        'timeframes': timeframes_info,
        'traded': traded,
        'trade_result': trade_result,
        'target_price': round(
            price - potential_move if signal_type == "SELL" else price + potential_move, 6
        ),
        'verified': False,
        'outcome': None,
        'price_after': None,
        'actual_move': None,
        'pnl_pct': None,
    }
    history['signals'].append(entry)
    save_signal_history(history)
    return entry


# =============================================================================
# AFFICHAGE MULTI-TIMEFRAME
# =============================================================================

def print_multi_tf_analysis(combined):
    """Affiche l'analyse multi-timeframe combinee."""
    price = combined['price']
    combined_sell = combined['combined_sell']
    combined_buy = combined['combined_buy']
    confirmations_sell = combined['confirmations_sell']
    confirmations_buy = combined['confirmations_buy']
    total_tf = combined['total_tf']
    rsi = combined['rsi']
    potential_move = combined['potential_move']
    move_ok = combined['move_ok']
    trend = combined['trend']
    fetch_time = combined.get('fetch_time_ms', 0)
    is_realtime = combined.get('realtime', False)

    # Signal principal
    if combined_sell >= 70 and move_ok:
        signal = ">>> VENDRE MAINTENANT <<<"
        signal_level = "FORT"
    elif combined_sell >= 50 and move_ok:
        signal = ">> SIGNAL DE VENTE <<"
        signal_level = "MOYEN"
    elif combined_sell >= 40 and move_ok:
        signal = "> Vente possible <"
        signal_level = "FAIBLE"
    elif combined_buy >= 70 and move_ok:
        signal = ">>> ACHETER MAINTENANT <<<"
        signal_level = "FORT"
    elif combined_buy >= 50 and move_ok:
        signal = ">> SIGNAL D'ACHAT <<"
        signal_level = "MOYEN"
    elif combined_buy >= 40 and move_ok:
        signal = "> Achat possible <"
        signal_level = "FAIBLE"
    else:
        signal = "-- PAS DE SIGNAL --"
        signal_level = "AUCUN"

    mode_str = "TEMPS REEL" if is_realtime else "POLLING"

    print("\n" + "=" * 70)
    print("  XRP/USDT - MULTI-TIMEFRAME PRO - BARE METAL EDITION")
    print("=" * 70)
    print(f"  Heure UTC  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Prix       : {price:.4f} USDT")
    print(f"  Mode       : {mode_str} | {total_tf} timeframes")
    if fetch_time > 0:
        print(f"  Latence    : {fetch_time}ms (fetch+analyse parallele)")

    print("-" * 70)
    print(f"\n  {signal}")
    print(f"  Force du signal : {signal_level}")
    print()

    # Tableau multi-timeframe
    print("-" * 70)
    print("  ANALYSE PAR TIMEFRAME")
    print("-" * 70)
    print(f"  {'TF':>4s} | {'POIDS':>5s} | {'SELL':>5s} | {'BUY':>5s} | {'RSI':>5s} | {'MACD':>9s} | {'TREND':>7s}")
    print("  " + "-" * 60)

    for tf in sorted(combined['tf_details'].keys(),
                     key=lambda x: _tf_to_minutes(x)):
        detail = combined['tf_details'][tf]
        if 'error' in detail:
            print(f"  {tf:>4s} | {'ERR':>5s} | {'-':>5s} | {'-':>5s} | {'-':>5s} | {'-':>9s} | {detail['error'][:15]}")
            continue

        weight_pct = f"{detail['weight']*100:.0f}%"
        sell_str = f"{detail['sell_score']:.0f}%"
        buy_str = f"{detail['buy_score']:.0f}%"
        rsi_str = f"{detail['rsi']:.0f}"

        # Indicateurs visuels
        sell_flag = " !" if detail['sell_score'] >= 50 else ""
        buy_flag = " !" if detail['buy_score'] >= 50 else ""

        print(f"  {tf:>4s} | {weight_pct:>5s} | {sell_str:>4s}{sell_flag} | "
              f"{buy_str:>4s}{buy_flag} | {rsi_str:>5s} | {detail['macd']:>9s} | {detail['trend']:>7s}")

    print("  " + "-" * 60)

    # Scores combines
    print(f"\n  SCORE COMBINE SELL : {combined_sell:.1f}%"
          f"  ({confirmations_sell}/{total_tf} TF confirment)", end="")
    if combined_sell >= 70:
        print("  [!!! TRES ELEVE !!!]")
    elif combined_sell >= 50:
        print("  [! ELEVE !]")
    else:
        print()

    print(f"  SCORE COMBINE BUY  : {combined_buy:.1f}%"
          f"  ({confirmations_buy}/{total_tf} TF confirment)", end="")
    if combined_buy >= 70:
        print("  [!!! TRES ELEVE !!!]")
    elif combined_buy >= 50:
        print("  [! ELEVE !]")
    else:
        print()

    # Confirmation multi-TF
    print()
    if confirmations_sell >= 3:
        print(f"  [CONFIRMATION FORTE] {confirmations_sell} timeframes confirment le signal SELL")
    elif confirmations_sell >= 2:
        print(f"  [CONFIRMATION] {confirmations_sell} timeframes confirment le signal SELL")

    if confirmations_buy >= 3:
        print(f"  [CONFIRMATION FORTE] {confirmations_buy} timeframes confirment le signal BUY")
    elif confirmations_buy >= 2:
        print(f"  [CONFIRMATION] {confirmations_buy} timeframes confirment le signal BUY")

    # Indicateurs primaires
    print("\n" + "-" * 70)
    print(f"  RSI ({combined['primary_tf']:>3s})    : {rsi:.1f}", end="")
    if rsi > RSI_OVERBOUGHT:
        print("  [SURACHETE]")
    elif rsi < RSI_OVERSOLD:
        print("  [SURVENDU]")
    else:
        print("  [Neutre]")

    print(f"  Tendance       : {trend}")
    print(f"  Volume Ratio   : {combined['vol_ratio']:.2f}x", end="")
    if combined['vol_ratio'] > VOL_SPIKE_MULT:
        print("  [SPIKE!]")
    else:
        print()

    print(f"  ATR            : {combined['atr']:.4f}")
    print(f"  Mouvement Est. : {potential_move:.4f} USD", end="")
    if move_ok:
        print(f"  [> {MIN_MOVE_USD} OK]")
    else:
        print(f"  [< {MIN_MOVE_USD} INSUFFISANT]")

    # Objectifs
    sell_target = price - potential_move
    buy_target = price + potential_move
    print("-" * 70)
    print("  OBJECTIFS:")
    print(f"    Objectif VENTE  : {sell_target:.4f} (-{potential_move:.4f})")
    print(f"    Objectif RACHAT : {buy_target:.4f} (+{potential_move:.4f})")
    print("=" * 70)


def _tf_to_minutes(tf):
    """Convertit un timeframe en minutes pour le tri."""
    multipliers = {'m': 1, 'h': 60, 'd': 1440}
    unit = tf[-1]
    value = int(tf[:-1])
    return value * multipliers.get(unit, 1)


# =============================================================================
# TRADING ENGINE (reutilise la logique de l'original)
# =============================================================================

def setup_trade_logging():
    handler = logging.FileHandler(TRADE_LOG_FILE)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"))
    trade_logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("  [TRADE] %(message)s"))
    trade_logger.addHandler(console)


def get_account_balance(client, asset="XRP"):
    try:
        account = client.get_account()
        for balance in account['balances']:
            if balance['asset'] == asset:
                return {
                    'free': float(balance['free']),
                    'locked': float(balance['locked']),
                    'total': float(balance['free']) + float(balance['locked'])
                }
    except BinanceAPIException as e:
        trade_logger.error(f"Erreur lecture solde {asset}: {e}")
    return {'free': 0.0, 'locked': 0.0, 'total': 0.0}


def get_symbol_info(client, symbol=SYMBOL):
    try:
        info = client.get_symbol_info(symbol)
        filters = {f['filterType']: f for f in info['filters']}
        lot_size = filters.get('LOT_SIZE', {})
        min_notional = filters.get('NOTIONAL', filters.get('MIN_NOTIONAL', {}))
        price_filter = filters.get('PRICE_FILTER', {})
        return {
            'min_qty': float(lot_size.get('minQty', 1)),
            'max_qty': float(lot_size.get('maxQty', 999999)),
            'step_size': float(lot_size.get('stepSize', 0.1)),
            'min_notional': float(min_notional.get('minNotional', 10)),
            'tick_size': float(price_filter.get('tickSize', 0.0001)),
        }
    except Exception as e:
        trade_logger.error(f"Erreur lecture symbol info: {e}")
        return {
            'min_qty': 1.0, 'max_qty': 9999999.0, 'step_size': 0.1,
            'min_notional': 10.0, 'tick_size': 0.0001,
        }


def round_step_size(quantity, step_size):
    return float(Decimal(str(quantity)).quantize(
        Decimal(str(step_size)), rounding=ROUND_DOWN))


def round_tick_size(price, tick_size):
    return float(Decimal(str(price)).quantize(
        Decimal(str(tick_size)), rounding=ROUND_DOWN))


def execute_combined_trade(client, combined, trade_config):
    """Execute un trade base sur le score multi-TF combine."""
    sell_score = combined['combined_sell']
    buy_score = combined['combined_buy']
    move_ok = combined['move_ok']
    price = combined['price']
    potential_move = combined['potential_move']
    rsi = combined['rsi']
    trend = combined['trend']

    dry_run = trade_config['dry_run']
    quantity = trade_config['quantity']
    sell_threshold = trade_config['sell_threshold']
    buy_threshold = trade_config['buy_threshold']
    order_type = trade_config['order_type']
    symbol_info = trade_config.get('symbol_info')
    record_mode = trade_config.get('record', True)

    # TF info pour l'historique
    tf_info = {tf: {'sell': d.get('sell_score', 0), 'buy': d.get('buy_score', 0)}
               for tf, d in combined['tf_details'].items() if 'error' not in d}

    xrp_balance = get_account_balance(client, "XRP")
    usdt_balance = get_account_balance(client, "USDT")

    print("\n" + "-" * 60)
    print("  TRADING ENGINE (MULTI-TF)")
    print("-" * 60)
    print(f"  Mode        : {'DRY-RUN' if dry_run else 'REEL'}")
    print(f"  Solde XRP   : {xrp_balance['free']:.2f} / {xrp_balance['total']:.2f}")
    print(f"  Solde USDT  : {usdt_balance['free']:.2f} / {usdt_balance['total']:.2f}")
    print(f"  Confirmations : SELL {combined['confirmations_sell']}/{combined['total_tf']} | "
          f"BUY {combined['confirmations_buy']}/{combined['total_tf']}")

    if sell_score >= sell_threshold and move_ok:
        trade_qty = min(quantity, xrp_balance['free'])
        if trade_qty <= 0:
            print(f"\n  [!] Signal SELL {sell_score:.0f}% mais solde XRP insuffisant")
            return

        sell_price = price if order_type == "LIMIT" else None
        print(f"\n  >>> EXECUTION VENTE | {trade_qty} XRP | Score: {sell_score:.0f}%"
              f" | {combined['confirmations_sell']} TF confirment")

        result = _place_order(client, 'SELL', trade_qty, sell_price,
                              dry_run, order_type, symbol_info)

        if record_mode:
            record_signal("SELL", sell_score, price, rsi, trend,
                          potential_move, tf_info, traded=result is not None,
                          trade_result="DRY-RUN" if dry_run else "EXECUTED")

    elif buy_score >= buy_threshold and move_ok:
        buy_cost = quantity * price
        if usdt_balance['free'] < buy_cost:
            max_qty = usdt_balance['free'] / price
            if symbol_info:
                max_qty = round_step_size(max_qty, symbol_info['step_size'])
            if max_qty <= 0:
                print(f"\n  [!] Signal BUY {buy_score:.0f}% mais solde USDT insuffisant")
                return
            trade_qty = max_qty
        else:
            trade_qty = quantity

        buy_price = price if order_type == "LIMIT" else None
        print(f"\n  >>> EXECUTION ACHAT | {trade_qty} XRP | Score: {buy_score:.0f}%"
              f" | {combined['confirmations_buy']} TF confirment")

        result = _place_order(client, 'BUY', trade_qty, buy_price,
                              dry_run, order_type, symbol_info)

        if record_mode:
            record_signal("BUY", buy_score, price, rsi, trend,
                          potential_move, tf_info, traded=result is not None,
                          trade_result="DRY-RUN" if dry_run else "EXECUTED")
    else:
        print(f"\n  Pas de trade (SELL: {sell_score:.0f}% | BUY: {buy_score:.0f}%)")
        if record_mode:
            if sell_score >= 40 and move_ok:
                record_signal("SELL", sell_score, price, rsi, trend,
                              potential_move, tf_info, traded=False)
            elif buy_score >= 40 and move_ok:
                record_signal("BUY", buy_score, price, rsi, trend,
                              potential_move, tf_info, traded=False)


def _place_order(client, side, quantity, price, dry_run, order_type, symbol_info):
    """Place un ordre sur Binance."""
    if symbol_info:
        quantity = round_step_size(quantity, symbol_info['step_size'])

    if dry_run:
        trade_logger.info(
            f"[DRY-RUN] {side} {quantity} XRP @ "
            f"{'MARKET' if order_type == 'MARKET' else f'{price:.4f}'}")
        return {'dry_run': True, 'side': side, 'status': 'SIMULATED'}

    try:
        if side == 'SELL':
            if order_type == "MARKET":
                return client.order_market_sell(symbol=SYMBOL, quantity=quantity)
            else:
                price = round_tick_size(price, symbol_info['tick_size'])
                return client.order_limit_sell(
                    symbol=SYMBOL, quantity=quantity,
                    price=str(price), timeInForce='GTC')
        else:
            if order_type == "MARKET":
                return client.order_market_buy(symbol=SYMBOL, quantity=quantity)
            else:
                price = round_tick_size(price, symbol_info['tick_size'])
                return client.order_limit_buy(
                    symbol=SYMBOL, quantity=quantity,
                    price=str(price), timeInForce='GTC')
    except BinanceAPIException as e:
        trade_logger.error(f"ERREUR {side}: {e}")
        return None


# =============================================================================
# HISTORIQUE & STATS
# =============================================================================

def print_signal_history():
    """Affiche le rapport de performance."""
    history = load_signal_history()
    if not history['signals']:
        print("\n  Aucun signal enregistre.")
        print(f"  Lancez avec --record pour commencer a enregistrer.")
        return

    signals = history['signals']
    verified = [s for s in signals if s['verified']]
    pending = len(signals) - len(verified)

    print("\n" + "=" * 70)
    print("  HISTORIQUE DES SIGNAUX PRO - RAPPORT MULTI-TF")
    print("=" * 70)
    print(f"\n  Total signaux  : {len(signals)}")
    print(f"  Verifies       : {len(verified)}")
    print(f"  En attente     : {pending}")

    if verified:
        correct = len([s for s in verified if s['outcome'] == "CORRECT"])
        partial = len([s for s in verified if s['outcome'] == "PARTIAL"])
        wrong = len([s for s in verified if s['outcome'] == "WRONG"])
        pnls = [s['pnl_pct'] for s in verified if s['pnl_pct'] is not None]
        avg_pnl = sum(pnls) / len(pnls) if pnls else 0

        print(f"\n  --- TAUX DE REUSSITE ---")
        print(f"  Corrects       : {correct} ({correct/len(verified)*100:.1f}%)")
        print(f"  Partiels       : {partial}")
        print(f"  Faux           : {wrong}")
        print(f"  P&L moyen      : {avg_pnl:+.2f}%")
        print(f"  P&L cumule     : {sum(pnls):+.2f}%")

    # Derniers signaux
    print("\n" + "-" * 70)
    print("  DERNIERS SIGNAUX (20 plus recents)")
    print("-" * 70)
    print(f"  {'TYPE':5s} | {'SCORE':5s} | {'PRIX':>10s} | {'RESULTAT':8s} | {'P&L':>7s} | {'TFs':>4s} | {'DATE':16s}")
    print("  " + "-" * 68)

    recent = signals[-20:]
    for s in reversed(recent):
        outcome = s.get('outcome') or 'EN ATT.'
        pnl = f"{s['pnl_pct']:+.2f}%" if s.get('pnl_pct') is not None else "  -   "
        icon = {"CORRECT": "[OK]", "PARTIAL": "[~~]", "WRONG": "[XX]"}.get(outcome, "[..]")
        n_tfs = len(s.get('timeframes', {})) if isinstance(s.get('timeframes'), dict) else 1
        print(f"  {s['type']:5s} | {s['score']:5.0f} | {s['price']:10.4f} | "
              f"{icon:4s} {outcome:8s} | {pnl:>7s} | {n_tfs:>4d} | {s['timestamp'][:16]}")

    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    global MIN_MOVE_USD

    parser = argparse.ArgumentParser(
        description="XRP Signal Detector PRO - Multi-TF & Websocket",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Multi-timeframe analyse (4 TF en parallele)
  python xrp_signal_pro.py

  # Temps reel avec websocket
  python xrp_signal_pro.py --realtime --loop

  # Choisir les timeframes
  python xrp_signal_pro.py --timeframes 1m 5m 15m 1h 4h

  # Dry-run multi-TF
  python xrp_signal_pro.py --realtime --trade --dry-run --key CLE --secret SECRET --loop
        """
    )

    # General
    grp_gen = parser.add_argument_group("General")
    grp_gen.add_argument("--key", default="",
                         help="Binance API key")
    grp_gen.add_argument("--secret", default="",
                         help="Binance API secret")
    grp_gen.add_argument("--timeframes", nargs='+',
                         default=DEFAULT_TIMEFRAMES,
                         choices=["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"],
                         help="Timeframes a analyser (defaut: 1m 5m 15m 1h)")
    grp_gen.add_argument("--realtime", action="store_true",
                         help="Mode websocket temps reel")
    grp_gen.add_argument("--loop", action="store_true",
                         help="Mode boucle continue")
    grp_gen.add_argument("--loop-delay", type=int, default=30,
                         help="Delai entre analyses en mode polling (defaut: 30s)")
    grp_gen.add_argument("--min-move", type=float, default=MIN_MOVE_USD,
                         help=f"Mouvement minimum USD (defaut: {MIN_MOVE_USD})")

    # Trading
    grp_trade = parser.add_argument_group("Trading")
    grp_trade.add_argument("--trade", action="store_true",
                           help="Activer le trading")
    grp_trade.add_argument("--dry-run", action="store_true",
                           help="Simuler les ordres")
    grp_trade.add_argument("--quantity", type=float, default=10.0,
                           help="Quantite XRP par trade")
    grp_trade.add_argument("--sell-threshold", type=int, default=70,
                           help="Score min pour vendre (defaut: 70)")
    grp_trade.add_argument("--buy-threshold", type=int, default=70,
                           help="Score min pour acheter (defaut: 70)")
    grp_trade.add_argument("--order-type", default="LIMIT",
                           choices=["LIMIT", "MARKET"])

    # Historique
    grp_hist = parser.add_argument_group("Historique")
    grp_hist.add_argument("--history", action="store_true",
                          help="Afficher le rapport de performance")
    grp_hist.add_argument("--record", action="store_true",
                          help="Enregistrer les signaux (mode observation)")
    grp_hist.add_argument("--clear-history", action="store_true",
                          help="Effacer l'historique")

    args = parser.parse_args()
    api_key = args.key or os.environ.get("BINANCE_API_KEY", "")
    api_secret = args.secret or os.environ.get("BINANCE_API_SECRET", "")
    MIN_MOVE_USD = args.min_move

    # Commandes rapides
    if args.history:
        print_signal_history()
        return

    if args.clear_history:
        save_signal_history({'signals': [], 'version': 2})
        print("  Historique efface.")
        return

    # Verifier cle API pour trading
    if args.trade and not api_key:
        print("\n  ERREUR: Le trading necessite une cle API Binance.")
        print("  Utilisez --key/--secret ou BINANCE_API_KEY/BINANCE_API_SECRET")
        sys.exit(1)

    if args.trade:
        setup_trade_logging()

    # Header
    print("\n" + "=" * 70)
    print("  XRP SIGNAL DETECTOR PRO - BARE METAL EDITION")
    print("=" * 70)
    print(f"  Paire           : {SYMBOL}")
    print(f"  Timeframes      : {', '.join(args.timeframes)}")
    print(f"  Mode            : {'WEBSOCKET TEMPS REEL' if args.realtime else 'POLLING PARALLELE'}")
    print(f"  Mouvement min.  : {MIN_MOVE_USD} USD")
    if args.trade:
        print(f"  Trading         : {'DRY-RUN' if args.dry_run else 'REEL'}")
        print(f"  Quantite/trade  : {args.quantity} XRP")
        print(f"  Seuil SELL/BUY  : {args.sell_threshold}% / {args.buy_threshold}%")
    print("=" * 70)

    # Trade config
    trade_config = None
    client = None
    if args.trade or args.record:
        client = Client(api_key, api_secret)
        trade_config = {
            'dry_run': args.dry_run if args.trade else True,
            'quantity': args.quantity,
            'sell_threshold': args.sell_threshold if args.trade else 40,
            'buy_threshold': args.buy_threshold if args.trade else 40,
            'order_type': args.order_type,
            'symbol_info': get_symbol_info(client) if args.trade else None,
            'record': True,
        }

    # === MODE WEBSOCKET TEMPS REEL ===
    if args.realtime and HAS_WEBSOCKET:
        def on_signal(combined):
            """Callback appele a chaque nouveau signal temps reel."""
            print_multi_tf_analysis(combined)
            if trade_config and client:
                execute_combined_trade(client, combined, trade_config)

        rt_engine = RealtimeEngine(
            api_key=api_key,
            api_secret=api_secret,
            timeframes=args.timeframes,
            on_signal=on_signal
        )

        initial = rt_engine.start()
        if initial:
            print_multi_tf_analysis(initial)
            if trade_config and client:
                execute_combined_trade(client, initial, trade_config)

        print(f"\n  Mode temps reel actif. Ctrl+C pour arreter.")
        try:
            while rt_engine.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n  Arret...")
        finally:
            rt_engine.stop()
        return

    # === MODE POLLING MULTI-TF ===
    mt_engine = MultiTimeframeEngine(
        api_key=api_key,
        api_secret=api_secret,
        timeframes=args.timeframes
    )

    while True:
        try:
            combined = mt_engine.analyze_all()
            print_multi_tf_analysis(combined)

            if trade_config and client:
                execute_combined_trade(client, combined, trade_config)

            if not args.loop:
                break

            print(f"\n  Prochain scan dans {args.loop_delay}s... (Ctrl+C pour arreter)")
            time.sleep(args.loop_delay)

        except KeyboardInterrupt:
            print("\n\n  Arret du scanner PRO. A bientot!")
            break
        except Exception as e:
            print(f"\n  Erreur: {e}")
            if not args.loop:
                break
            print("  Nouvelle tentative dans 15s...")
            time.sleep(15)

    mt_engine.shutdown()

    if args.trade:
        print(f"\n  Historique: {TRADE_LOG_FILE}")


if __name__ == "__main__":
    main()
