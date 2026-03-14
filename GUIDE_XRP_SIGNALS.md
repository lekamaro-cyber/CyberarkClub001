# XRP/USD - Detecteur de Signaux de Vente & Accumulation

## Strategie

Vendre **avant la panique** quand les indicateurs montrent une faiblesse, puis **racheter dans le creux** pour accumuler plus de XRP.

Mouvement minimum requis : **0.10 USD** (couvre les frais Binance maker/taker).

---

## Composant 1 : TradingView (Pine Script)

### Installation

1. Ouvrir TradingView > graphique XRP/USD ou XRP/USDT
2. Cliquer sur **Pine Editor** (en bas)
3. Coller le contenu de `xrp_sell_signal_tradingview.pine`
4. Cliquer **Ajouter au graphique**

### Lecture du tableau de bord

| Champ | Description |
|-------|-------------|
| **% BAISSE** | Probabilite que le prix baisse (0-100%) |
| **% HAUSSE** | Probabilite que le prix monte (0-100%) |
| **Mouvement Est.** | Mouvement potentiel estime en USD |
| **Min Requis** | Indique si le mouvement couvre les frais (OK/NON) |
| **Obj. Vente** | Prix cible a la baisse |
| **Obj. Rachat** | Prix cible de rachat |

### Signaux visuels

- **Triangle rouge GRAND** = SELL FORT (>= 70%) - **VENDRE**
- **Triangle orange** = SELL MOYEN (50-69%) - Preparer la vente
- **Triangle jaune petit** = SELL FAIBLE (40-49%) - Surveiller
- **Triangle vert GRAND** = BUY FORT (>= 70%) - **ACHETER**
- **Triangle vert clair** = BUY MOYEN (50-69%) - Preparer l'achat
- **Triangle teal petit** = BUY FAIBLE (40-49%) - Surveiller

### Alertes TradingView

1. Sur le graphique > clic droit > **Ajouter alerte**
2. Condition : choisir l'indicateur "XRP Sell Signal Detector"
3. Selectionner : "SELL FORT XRP" ou "SIGNAL FORT XRP"
4. Notification : Push / Email / Webhook

---

## Composant 2 : Script Python (Binance)

### Installation

```bash
pip install python-binance pandas numpy
```

### Utilisation

```bash
# Analyse unique (15m par defaut)
python xrp_signal_binance.py

# Timeframe 1h
python xrp_signal_binance.py --interval 1h

# Mode boucle (rafraichit toutes les 60s)
python xrp_signal_binance.py --loop --loop-delay 60

# Avec cles API Binance
python xrp_signal_binance.py --key VOTRE_CLE --secret VOTRE_SECRET

# Changer le mouvement minimum
python xrp_signal_binance.py --min-move 0.15
```

### Sortie

```
============================================================
  XRP/USDT - DETECTION DE SIGNAUX - ACCUMULATION
============================================================
  Prix      : 2.3456 USDT

  >>> VENDRE MAINTENANT <<<
  Force du signal : FORT

  % Probabilite BAISSE : 75%  [!!! TRES ELEVE !!!]
  % Probabilite HAUSSE : 15%
  Mouvement Est.  : 0.1523 USD  [> 0.1 OK]
  Objectif VENTE  : 2.1933 (-0.1523)
  Objectif RACHAT : 2.4979 (+0.1523)
============================================================
```

---

## Systeme de scoring (8 criteres, 100 points max)

| Critere | Points | Description |
|---------|--------|-------------|
| RSI surachat/survente | 0-20 | Plus le RSI est extreme, plus le score monte |
| MACD croisement | 15 | Croisement baissier/haussier de la ligne signal |
| MACD histogramme | 5 | 3 bougies consecutives en declin/hausse |
| Bollinger Bands | 10 | Prix hors des bandes |
| Volume spike | 10 | Volume > 1.5x moyenne sur bougie rouge/verte |
| EMA croisement | 10 | EMA 9 croise EMA 21 |
| Divergence RSI | 15 | Prix fait un nouveau plus haut, RSI non |
| OBV divergence | 10 | Volume ne confirme pas le mouvement du prix |
| StochRSI | 5 | Stochastic RSI en zone extreme + croisement |

### Interpretation des scores

| Score | Niveau | Action |
|-------|--------|--------|
| 70-100% | FORT | Agir immediatement |
| 50-69% | MOYEN | Se preparer, confirmer sur timeframe superieur |
| 40-49% | FAIBLE | Surveiller, ne pas encore agir |
| 0-39% | - | Pas de signal |

---

## Timeframes recommandes

| Timeframe | Usage |
|-----------|-------|
| 5m / 15m | Scalping, entrees/sorties precises |
| 1h | Swing trading intraday |
| 4h | Swing trading multi-jours |
| 1d | Vision macro, confirmation de tendance |

**Conseil** : Confirmer les signaux du 15m avec le 1h ou le 4h avant d'agir.

---

## Avertissement

Cet outil est un **aide a la decision**. Les pourcentages representent la convergence des indicateurs techniques, pas une prediction garantie. Toujours utiliser un stop-loss et ne risquer que ce que vous pouvez perdre.
