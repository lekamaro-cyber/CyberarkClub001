#!/usr/bin/env python3
"""
XRP/USD Sell Signal Detector & Auto-Trader - Binance Edition
=============================================================
Detecte les signaux de vente avant la panique et les signaux
d'achat dans les creux pour une strategie d'accumulation.
Peut placer des ordres automatiquement sur Binance.

Mouvement minimum: 0.10 USD (couvre les frais Binance)
Paire: XRP/USDT sur Binance

Dependances:
    pip install python-binance pandas numpy

Usage:
    # Analyse seule (pas d'ordres)
    python xrp_signal_binance.py

    # Mode auto-trade (place les ordres)
    python xrp_signal_binance.py --trade --key CLE --secret SECRET

    # Dry-run (simule sans passer d'ordres reels)
    python xrp_signal_binance.py --trade --dry-run --key CLE --secret SECRET

    # Configurer la quantite et le seuil
    python xrp_signal_binance.py --trade --quantity 50 --sell-threshold 70 --buy-threshold 70
"""

import argparse
import json
import logging
import os
import sys
import time
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
# TRADING - PASSAGE D'ORDRES BINANCE
# =============================================================================

# Logging pour le trading
trade_logger = logging.getLogger("xrp_trader")
trade_logger.setLevel(logging.INFO)

TRADE_LOG_FILE = "xrp_trades.log"


def setup_trade_logging():
    """Configure le logging des trades dans un fichier."""
    handler = logging.FileHandler(TRADE_LOG_FILE)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    ))
    trade_logger.addHandler(handler)
    # Aussi en console
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("  [TRADE] %(message)s"))
    trade_logger.addHandler(console)


def get_account_balance(client, asset="XRP"):
    """Recupere le solde d'un asset sur Binance."""
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
    """Recupere les regles de trading (lot size, min notional, etc.)."""
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
            'min_price': float(price_filter.get('minPrice', 0)),
            'max_price': float(price_filter.get('maxPrice', 0)),
        }
    except Exception as e:
        trade_logger.error(f"Erreur lecture symbol info: {e}")
        # Valeurs par defaut pour XRP/USDT
        return {
            'min_qty': 1.0,
            'max_qty': 9999999.0,
            'step_size': 0.1,
            'min_notional': 10.0,
            'tick_size': 0.0001,
            'min_price': 0.0,
            'max_price': 0.0,
        }


def round_step_size(quantity, step_size):
    """Arrondit la quantite au step_size de Binance."""
    precision = len(str(step_size).rstrip('0').split('.')[-1])
    return float(Decimal(str(quantity)).quantize(
        Decimal(str(step_size)), rounding=ROUND_DOWN
    ))


def round_tick_size(price, tick_size):
    """Arrondit le prix au tick_size de Binance."""
    return float(Decimal(str(price)).quantize(
        Decimal(str(tick_size)), rounding=ROUND_DOWN
    ))


def place_sell_order(client, quantity, price=None, dry_run=False,
                     order_type="LIMIT", symbol_info=None):
    """
    Place un ordre de vente XRP sur Binance.

    Args:
        client: Binance client authentifie
        quantity: Quantite de XRP a vendre
        price: Prix limite (None = market order)
        dry_run: True = simuler sans passer l'ordre
        order_type: "LIMIT" ou "MARKET"
        symbol_info: Infos du symbole (step_size, etc.)

    Returns:
        dict avec le resultat de l'ordre ou None en cas d'erreur
    """
    if symbol_info is None:
        symbol_info = get_symbol_info(client)

    # Arrondir au step_size
    quantity = round_step_size(quantity, symbol_info['step_size'])

    # Verifications
    if quantity < symbol_info['min_qty']:
        trade_logger.warning(
            f"Quantite {quantity} < minimum {symbol_info['min_qty']}")
        return None

    current_price = float(client.get_symbol_ticker(symbol=SYMBOL)['price'])
    notional = quantity * (price or current_price)
    if notional < symbol_info['min_notional']:
        trade_logger.warning(
            f"Valeur ordre {notional:.2f} USDT < minimum {symbol_info['min_notional']} USDT")
        return None

    if dry_run:
        trade_logger.info(
            f"[DRY-RUN] SELL {quantity} XRP @ "
            f"{'MARKET' if order_type == 'MARKET' else f'{price:.4f}'} "
            f"| Valeur: ~{notional:.2f} USDT")
        return {
            'dry_run': True,
            'side': 'SELL',
            'quantity': quantity,
            'price': price or current_price,
            'notional': notional,
            'status': 'SIMULATED'
        }

    try:
        if order_type == "MARKET":
            order = client.order_market_sell(
                symbol=SYMBOL,
                quantity=quantity
            )
        else:
            if price is None:
                price = current_price
            price = round_tick_size(price, symbol_info['tick_size'])
            order = client.order_limit_sell(
                symbol=SYMBOL,
                quantity=quantity,
                price=str(price),
                timeInForce='GTC'
            )

        trade_logger.info(
            f"SELL EXECUTE | {quantity} XRP @ {order.get('price', 'MARKET')} "
            f"| Order ID: {order['orderId']} | Status: {order['status']}")
        return order

    except BinanceAPIException as e:
        trade_logger.error(f"ERREUR SELL: {e}")
        return None


def place_buy_order(client, quantity=None, quote_amount=None, price=None,
                    dry_run=False, order_type="LIMIT", symbol_info=None):
    """
    Place un ordre d'achat XRP sur Binance.

    Args:
        client: Binance client authentifie
        quantity: Quantite de XRP a acheter (prioritaire)
        quote_amount: Montant en USDT a depenser (si quantity=None)
        price: Prix limite (None = market order)
        dry_run: True = simuler sans passer l'ordre
        order_type: "LIMIT" ou "MARKET"
        symbol_info: Infos du symbole

    Returns:
        dict avec le resultat de l'ordre ou None en cas d'erreur
    """
    if symbol_info is None:
        symbol_info = get_symbol_info(client)

    current_price = float(client.get_symbol_ticker(symbol=SYMBOL)['price'])
    target_price = price or current_price

    # Calculer la quantite si on a un montant USDT
    if quantity is None and quote_amount is not None:
        quantity = quote_amount / target_price

    if quantity is None:
        trade_logger.error("Ni quantite ni montant USDT specifie")
        return None

    quantity = round_step_size(quantity, symbol_info['step_size'])

    if quantity < symbol_info['min_qty']:
        trade_logger.warning(
            f"Quantite {quantity} < minimum {symbol_info['min_qty']}")
        return None

    notional = quantity * target_price
    if notional < symbol_info['min_notional']:
        trade_logger.warning(
            f"Valeur ordre {notional:.2f} USDT < minimum {symbol_info['min_notional']} USDT")
        return None

    if dry_run:
        trade_logger.info(
            f"[DRY-RUN] BUY {quantity} XRP @ "
            f"{'MARKET' if order_type == 'MARKET' else f'{target_price:.4f}'} "
            f"| Cout: ~{notional:.2f} USDT")
        return {
            'dry_run': True,
            'side': 'BUY',
            'quantity': quantity,
            'price': target_price,
            'notional': notional,
            'status': 'SIMULATED'
        }

    try:
        if order_type == "MARKET":
            order = client.order_market_buy(
                symbol=SYMBOL,
                quantity=quantity
            )
        else:
            price_rounded = round_tick_size(target_price, symbol_info['tick_size'])
            order = client.order_limit_buy(
                symbol=SYMBOL,
                quantity=quantity,
                price=str(price_rounded),
                timeInForce='GTC'
            )

        trade_logger.info(
            f"BUY EXECUTE | {quantity} XRP @ {order.get('price', 'MARKET')} "
            f"| Order ID: {order['orderId']} | Status: {order['status']}")
        return order

    except BinanceAPIException as e:
        trade_logger.error(f"ERREUR BUY: {e}")
        return None


def place_oco_sell(client, quantity, target_price, stop_price, stop_limit_price,
                   dry_run=False, symbol_info=None):
    """
    Place un ordre OCO (One-Cancels-Other) pour vendre avec take-profit ET stop-loss.

    Args:
        client: Binance client
        quantity: Quantite XRP
        target_price: Prix take-profit (limit)
        stop_price: Prix de declenchement du stop
        stop_limit_price: Prix limite du stop (un peu en dessous du stop)
        dry_run: Simulation
        symbol_info: Infos du symbole
    """
    if symbol_info is None:
        symbol_info = get_symbol_info(client)

    quantity = round_step_size(quantity, symbol_info['step_size'])
    target_price = round_tick_size(target_price, symbol_info['tick_size'])
    stop_price = round_tick_size(stop_price, symbol_info['tick_size'])
    stop_limit_price = round_tick_size(stop_limit_price, symbol_info['tick_size'])

    if dry_run:
        trade_logger.info(
            f"[DRY-RUN] OCO SELL {quantity} XRP | "
            f"Take-Profit: {target_price:.4f} | "
            f"Stop: {stop_price:.4f} | "
            f"Stop-Limit: {stop_limit_price:.4f}")
        return {'dry_run': True, 'type': 'OCO_SELL', 'status': 'SIMULATED'}

    try:
        order = client.create_oco_order(
            symbol=SYMBOL,
            side='SELL',
            quantity=quantity,
            price=str(target_price),
            stopPrice=str(stop_price),
            stopLimitPrice=str(stop_limit_price),
            stopLimitTimeInForce='GTC'
        )
        trade_logger.info(
            f"OCO SELL PLACE | {quantity} XRP | "
            f"TP: {target_price} | Stop: {stop_price}")
        return order
    except BinanceAPIException as e:
        trade_logger.error(f"ERREUR OCO: {e}")
        return None


def cancel_open_orders(client, dry_run=False):
    """Annule tous les ordres ouverts sur XRP/USDT."""
    try:
        open_orders = client.get_open_orders(symbol=SYMBOL)
        if not open_orders:
            trade_logger.info("Aucun ordre ouvert a annuler")
            return []

        results = []
        for order in open_orders:
            if dry_run:
                trade_logger.info(
                    f"[DRY-RUN] Annulation ordre {order['orderId']} "
                    f"| {order['side']} {order['origQty']} @ {order['price']}")
                results.append({'dry_run': True, 'orderId': order['orderId']})
            else:
                result = client.cancel_order(
                    symbol=SYMBOL, orderId=order['orderId'])
                trade_logger.info(f"Ordre {order['orderId']} annule")
                results.append(result)
        return results
    except BinanceAPIException as e:
        trade_logger.error(f"Erreur annulation ordres: {e}")
        return []


def execute_signal_trade(client, df, trade_config):
    """
    Execute un trade basé sur les signaux detectes.

    Args:
        client: Binance client authentifie
        df: DataFrame avec les scores
        trade_config: dict avec la configuration de trading
    """
    last = df.iloc[-1]
    sell_score = last['sell_score']
    buy_score = last['buy_score']
    move_ok = last['move_ok']
    price = last['close']
    potential_move = last['potential_move']

    dry_run = trade_config['dry_run']
    quantity = trade_config['quantity']
    sell_threshold = trade_config['sell_threshold']
    buy_threshold = trade_config['buy_threshold']
    order_type = trade_config['order_type']
    use_stop_loss = trade_config['use_stop_loss']
    stop_loss_pct = trade_config['stop_loss_pct']
    symbol_info = trade_config.get('symbol_info')

    # Recuperer les soldes
    xrp_balance = get_account_balance(client, "XRP")
    usdt_balance = get_account_balance(client, "USDT")

    print("\n" + "-" * 60)
    print("  TRADING ENGINE")
    print("-" * 60)
    print(f"  Mode        : {'DRY-RUN (simulation)' if dry_run else 'REEL'}")
    print(f"  Solde XRP   : {xrp_balance['free']:.2f} (dispo) / {xrp_balance['total']:.2f} (total)")
    print(f"  Solde USDT  : {usdt_balance['free']:.2f} (dispo) / {usdt_balance['total']:.2f} (total)")
    print(f"  Quantite    : {quantity} XRP par trade")
    print(f"  Type ordre  : {order_type}")
    print(f"  Seuil SELL  : {sell_threshold}%")
    print(f"  Seuil BUY   : {buy_threshold}%")

    # ---- SIGNAL DE VENTE ----
    if sell_score >= sell_threshold and move_ok:
        trade_qty = min(quantity, xrp_balance['free'])
        if trade_qty <= 0:
            print(f"\n  [!] Signal SELL {sell_score:.0f}% mais solde XRP insuffisant ({xrp_balance['free']:.2f})")
            trade_logger.warning(f"SELL bloque - solde XRP insuffisant: {xrp_balance['free']}")
            return

        sell_price = None
        if order_type == "LIMIT":
            # Vendre au prix actuel ou un peu au dessus
            sell_price = price

        print(f"\n  >>> EXECUTION VENTE | {trade_qty} XRP | Score: {sell_score:.0f}%")

        result = place_sell_order(
            client, trade_qty, price=sell_price,
            dry_run=dry_run, order_type=order_type,
            symbol_info=symbol_info
        )

        if result:
            # Placer un ordre de rachat en dessous (accumulation)
            rebuy_price = round_tick_size(
                price - potential_move,
                symbol_info['tick_size'] if symbol_info else 0.0001
            )
            print(f"  >>> ORDRE RACHAT PLACE | {trade_qty} XRP @ {rebuy_price:.4f}")

            place_buy_order(
                client, quantity=trade_qty, price=rebuy_price,
                dry_run=dry_run, order_type="LIMIT",
                symbol_info=symbol_info
            )

    # ---- SIGNAL D'ACHAT ----
    elif buy_score >= buy_threshold and move_ok:
        buy_cost = quantity * price
        if usdt_balance['free'] < buy_cost:
            # Ajuster la quantite au solde disponible
            max_qty = usdt_balance['free'] / price
            if symbol_info:
                max_qty = round_step_size(max_qty, symbol_info['step_size'])
            if max_qty <= 0:
                print(f"\n  [!] Signal BUY {buy_score:.0f}% mais solde USDT insuffisant ({usdt_balance['free']:.2f})")
                trade_logger.warning(f"BUY bloque - solde USDT insuffisant: {usdt_balance['free']}")
                return
            trade_qty = max_qty
        else:
            trade_qty = quantity

        buy_price = None
        if order_type == "LIMIT":
            buy_price = price

        print(f"\n  >>> EXECUTION ACHAT | {trade_qty} XRP | Score: {buy_score:.0f}%")

        result = place_buy_order(
            client, quantity=trade_qty, price=buy_price,
            dry_run=dry_run, order_type=order_type,
            symbol_info=symbol_info
        )

        if result and use_stop_loss:
            # Placer un stop-loss
            stop_price = round_tick_size(
                price * (1 - stop_loss_pct / 100),
                symbol_info['tick_size'] if symbol_info else 0.0001
            )
            print(f"  >>> STOP-LOSS PLACE @ {stop_price:.4f} (-{stop_loss_pct}%)")

            place_sell_order(
                client, trade_qty, price=stop_price,
                dry_run=dry_run, order_type="LIMIT",
                symbol_info=symbol_info
            )

    else:
        print(f"\n  Pas de trade (SELL: {sell_score:.0f}% < {sell_threshold}% | BUY: {buy_score:.0f}% < {buy_threshold}%)")


def print_open_orders(client):
    """Affiche les ordres ouverts."""
    try:
        orders = client.get_open_orders(symbol=SYMBOL)
        if not orders:
            print("  Aucun ordre ouvert")
            return

        print(f"\n  ORDRES OUVERTS ({len(orders)}):")
        print("-" * 60)
        for o in orders:
            side_icon = "SELL" if o['side'] == 'SELL' else "BUY "
            print(f"  {side_icon} | {o['origQty']} XRP @ {o['price']} "
                  f"| Type: {o['type']} | ID: {o['orderId']}")
    except BinanceAPIException as e:
        print(f"  Erreur lecture ordres: {e}")


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
        return None, None
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

    return df, client


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
    global MIN_MOVE_USD

    parser = argparse.ArgumentParser(
        description="XRP/USDT Signal Detector & Auto-Trader - Binance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Analyse seule
  python xrp_signal_binance.py

  # Dry-run (simule les trades)
  python xrp_signal_binance.py --trade --dry-run --key CLE --secret SECRET --loop

  # Trading reel - ATTENTION: passe de vrais ordres!
  python xrp_signal_binance.py --trade --key CLE --secret SECRET --quantity 50 --loop
        """
    )

    # --- Parametres generaux ---
    grp_gen = parser.add_argument_group("General")
    grp_gen.add_argument("--key", default="",
                         help="Binance API key (optionnel pour analyse seule)")
    grp_gen.add_argument("--secret", default="",
                         help="Binance API secret")
    grp_gen.add_argument("--interval", default=DEFAULT_INTERVAL,
                         choices=["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"],
                         help="Timeframe (defaut: 15m)")
    grp_gen.add_argument("--loop", action="store_true",
                         help="Mode boucle - rafraichit automatiquement")
    grp_gen.add_argument("--loop-delay", type=int, default=60,
                         help="Delai entre rafraichissements en secondes (defaut: 60)")
    grp_gen.add_argument("--min-move", type=float, default=MIN_MOVE_USD,
                         help=f"Mouvement minimum en USD (defaut: {MIN_MOVE_USD})")

    # --- Parametres trading ---
    grp_trade = parser.add_argument_group("Trading")
    grp_trade.add_argument("--trade", action="store_true",
                           help="Activer le passage d'ordres")
    grp_trade.add_argument("--dry-run", action="store_true",
                           help="Simuler les ordres sans les passer reellement")
    grp_trade.add_argument("--quantity", type=float, default=10.0,
                           help="Quantite de XRP par trade (defaut: 10)")
    grp_trade.add_argument("--sell-threshold", type=int, default=70,
                           help="Score minimum pour declencher une vente (defaut: 70)")
    grp_trade.add_argument("--buy-threshold", type=int, default=70,
                           help="Score minimum pour declencher un achat (defaut: 70)")
    grp_trade.add_argument("--order-type", default="LIMIT",
                           choices=["LIMIT", "MARKET"],
                           help="Type d'ordre (defaut: LIMIT)")
    grp_trade.add_argument("--stop-loss", action="store_true",
                           help="Placer un stop-loss apres achat")
    grp_trade.add_argument("--stop-loss-pct", type=float, default=3.0,
                           help="Pourcentage du stop-loss (defaut: 3%%)")
    grp_trade.add_argument("--cancel-orders", action="store_true",
                           help="Annuler tous les ordres ouverts et quitter")
    grp_trade.add_argument("--show-orders", action="store_true",
                           help="Afficher les ordres ouverts et quitter")

    args = parser.parse_args()

    # Cles API depuis env si pas en argument
    api_key = args.key or os.environ.get("BINANCE_API_KEY", "")
    api_secret = args.secret or os.environ.get("BINANCE_API_SECRET", "")

    MIN_MOVE_USD = args.min_move

    # Verifier que les cles sont presentes pour le trading
    if args.trade and not api_key:
        print("\n  ERREUR: Le trading necessite une cle API Binance.")
        print("  Utilisez --key et --secret, ou les variables d'environnement:")
        print("    export BINANCE_API_KEY=votre_cle")
        print("    export BINANCE_API_SECRET=votre_secret")
        sys.exit(1)

    # Setup logging pour le trading
    if args.trade:
        setup_trade_logging()

    print("\n" + "=" * 60)
    print("  XRP SIGNAL DETECTOR" + (" & AUTO-TRADER" if args.trade else ""))
    print("=" * 60)
    print(f"  Paire           : {SYMBOL}")
    print(f"  Timeframe       : {args.interval}")
    print(f"  Mouvement min.  : {MIN_MOVE_USD} USD")
    if args.trade:
        print(f"  Mode            : {'DRY-RUN (simulation)' if args.dry_run else 'TRADING REEL'}")
        print(f"  Quantite/trade  : {args.quantity} XRP")
        print(f"  Seuil SELL      : {args.sell_threshold}%")
        print(f"  Seuil BUY       : {args.buy_threshold}%")
        print(f"  Type ordre      : {args.order_type}")
        print(f"  Stop-loss       : {'Oui (' + str(args.stop_loss_pct) + '%)' if args.stop_loss else 'Non'}")
        if not args.dry_run:
            print("\n  *** ATTENTION: MODE REEL - DE VRAIS ORDRES SERONT PASSES ***")
    print("=" * 60)

    # Commandes rapides: show-orders, cancel-orders
    if args.show_orders or args.cancel_orders:
        client = Client(api_key, api_secret)
        if args.show_orders:
            print_open_orders(client)
        if args.cancel_orders:
            cancel_open_orders(client, dry_run=args.dry_run)
        return

    # Config trading
    trade_config = None
    if args.trade:
        trade_config = {
            'dry_run': args.dry_run,
            'quantity': args.quantity,
            'sell_threshold': args.sell_threshold,
            'buy_threshold': args.buy_threshold,
            'order_type': args.order_type,
            'use_stop_loss': args.stop_loss,
            'stop_loss_pct': args.stop_loss_pct,
            'symbol_info': None,  # sera rempli au premier cycle
        }

    while True:
        try:
            result = get_binance_data(api_key, api_secret, args.interval)
            if result[0] is None:
                print("Impossible de recuperer les donnees. Nouvelle tentative dans 30s...")
                time.sleep(30)
                continue

            df, client = result
            df = analyze_signals(df)
            print_analysis(df)

            # Trading automatique
            if trade_config is not None:
                if trade_config['symbol_info'] is None:
                    trade_config['symbol_info'] = get_symbol_info(client)
                execute_signal_trade(client, df, trade_config)
                print_open_orders(client)

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

    if args.trade:
        print(f"\n  Historique des trades sauvegarde dans: {TRADE_LOG_FILE}")


if __name__ == "__main__":
    main()
