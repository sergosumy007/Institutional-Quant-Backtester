import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pybit.unified_trading import HTTP
import yfinance as yf
import threading
import time
import requests
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# ── Аналитические функции — из smc_core.py (единый источник правды) ──────────
from smc_core import (
    RF_FEATURE_NAMES,
    # CHECK-функции
    _check_bos_retest_long, _check_bos_retest_short, _check_bos_retest,
    _ftb_check, _check_sweep, _check_sweep_confirmed,
    _check_in_demand_ftr, _check_in_supply_ftr,
    _find_demand_ftr_zone, _find_supply_ftr_zone,
    # Order Flow
    apply_order_flow,
    # ZigZag / Market Phase
    calc_zigzag, calc_market_phase,
    # SMC структура
    _smc_structure_intraday, _smc_structure_daily, _smc_structure_fractal,
    build_structure, detect_key_levels, find_balance_start,
    is_structural_breakout, detect_swings,
    get_struct_levels,
    # Volume Profile
    calc_poc_fast, calc_value_area,
    build_balance_profiles,
    # FTR
    get_ftr_params, calc_ftr_zones,
    # Свинги / BOS / FVG / дивергенции
    find_swings, find_bos_ob_fvg, find_divergences,
    # dPOC / HVN
    calc_dynamic_poc, _calc_dpoc_rolling, find_hvn_zones,
    # Auction Context
    analyze_auction_context,
)


# ── Liq.Grab с подтверждением 2-й свечи ─────────────────────────────────────
def _sweep_confirmed(df, level, direction):
    """
    Обёртка для скринера: исключает живой бар df.iloc[-1] перед вызовом
    _check_sweep_confirmed из smc_core (единый источник истины).

      df.iloc[-3]  — свип-свеча (ЗАКРЫТА)
      df.iloc[-2]  — подтверждение (ЗАКРЫТА)
      df.iloc[-1]  — текущий живой бар (не используется)
    """
    if len(df) < 4:
        return False
    return _check_sweep_confirmed(df.iloc[:-1], level, direction)


# ── RF МОДЕЛИ — загружаем при старте ─────────────────────────────────────────
_RF_MODEL_DIR = '/opt/myscreener/backtest_data'

# ── Статистика бэктеста v3.6 — для обогащения TG сообщений ──────────────────
_BT_STATS: dict = {}   # {(pattern, direction, split): {wr, pnl, n}}
def _load_bt_stats():
    """Загружает pattern_stats_v35.json для показа исторического WR в TG."""
    import json as _json
    _paths = [
        'D:/MyScreener/backtest_data/pattern_stats_v35.json',
        '/opt/myscreener/backtest_data/pattern_stats_v35.json',
    ]
    for _p in _paths:
        try:
            with open(_p, 'r', encoding='utf-8') as _f:
                rows = _json.load(_f)
            d = {}
            for r in rows:
                d[(r['pattern'], r['direction'], r['split'])] = r
            print(f"[BT_STATS] Загружено {len(d)} записей из {_p}")
            return d
        except: pass
    print("[BT_STATS] pattern_stats_v35.json не найден")
    return {}
_BT_STATS = _load_bt_stats()

def _load_rf_model(path):
    """
    Загружает RF модель из pkl файла.
    Возвращает (model, calibrator, feature_names, threshold) или (None, None, [], 0.5).
    v2.0: model=rf (полная), calibrator=LogisticRegression (Platt), threshold=opt_threshold.
    """
    try:
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        except Exception:
            import joblib as _jl
            data = _jl.load(path)
        model      = data.get('model') or data
        calibrator = data.get('calibrator', None) if isinstance(data, dict) else None
        features   = data.get('feature_names', [])
        # opt_threshold — ключ v2.0; fallback на 'threshold' (v1.x) или 0.5
        threshold  = float(data.get('opt_threshold',
                           data.get('threshold', 0.5))) if isinstance(data, dict) else 0.5
        calib_str  = 'OOB-Platt' if calibrator is not None else 'none'
        print(f"[RF] Загружена: {path} ({len(features)} признаков, "
              f"thr={threshold:.3f}, calib={calib_str})")
        return model, calibrator, features, threshold
    except FileNotFoundError:
        print(f"[RF] Модель не найдена: {path}")
        return None, None, [], 0.5
    except Exception as e:
        print(f"[RF] Ошибка загрузки {path}: {e}")
        return None, None, [], 0.5

# SMC модель (бэктест v3.6) — для всех SMC паттернов скринера
# Порог 0.1922: WR@2.0 28.9%→32.4% на OOS, слабые паттерны (dPOC, CVD) +8-10pp
_rf_smc_model, _rf_smc_calib, _rf_smc_features, _rf_smc_threshold = \
    _load_rf_model(f'{_RF_MODEL_DIR}/rf_model_smc_v2.pkl')
if _rf_smc_model is None:
    _rf_smc_model, _rf_smc_calib, _rf_smc_features, _rf_smc_threshold = \
        _load_rf_model('D:/MyScreener/backtest_data/rf_model_smc_v2.pkl')

# Combined модель — лучший фильтр для Renko сигналов (91% обучения = Renko, ROC-AUC OOS 0.8056)
_rf_renko_model, _rf_renko_calib, _rf_renko_features, _rf_renko_threshold = \
    _load_rf_model(f'{_RF_MODEL_DIR}/rf_model_combined.pkl')
if _rf_renko_model is None:
    _rf_renko_model, _rf_renko_calib, _rf_renko_features, _rf_renko_threshold = \
        _load_rf_model('D:/MyScreener/backtest_data/rf_model_combined.pkl')

# Balance Zone Fakeout модель (bt v3.2) — для "Balance Reversal" сигналов
# 8 признаков: sweep_atr, poc_r, sl_dist, va_width_r, poc_pos, is_long, tf_15m, tf_1h
_rf_bzone_model, _rf_bzone_calib, _rf_bzone_features, _rf_bzone_threshold = \
    _load_rf_model(f'{_RF_MODEL_DIR}/rf_model_balance_zone.no_compress.joblib')
if _rf_bzone_model is None:
    _rf_bzone_model, _rf_bzone_calib, _rf_bzone_features, _rf_bzone_threshold = \
        _load_rf_model('D:/MyScreener/backtest_data/rf_model_balance_zone.no_compress.joblib')

# Backward compat aliases
_rf_combined_model     = _rf_smc_model
_rf_combined_calib     = _rf_smc_calib
_rf_combined_features  = _rf_smc_features
_rf_combined_threshold = _rf_smc_threshold

# RF фильтр скринера — порог SMC модели
_RF_THRESHOLD = _rf_smc_threshold if _rf_smc_model is not None else 0.0


def _rf_score(signal_features: dict, is_renko: bool = False) -> float:
    """
    Возвращает калиброванную вероятность WIN [0..1].
    SMC сигналы → rf_model_smc_v2 (порог 0.1922).
    Renko сигналы → rf_model_renko_v2 (порог 0.6392).
    """
    if is_renko:
        model, calibrator, features = _rf_renko_model, _rf_renko_calib, _rf_renko_features
    else:
        model, calibrator, features = _rf_smc_model, _rf_smc_calib, _rf_smc_features

    if model is None or not features:
        return -1.0

    try:
        import numpy as np
        # Добавляем source-признак — Combined модель различает SMC и Renko
        enriched = dict(signal_features)
        enriched['_source_renko'] = 1.0 if is_renko else 0.0
        # market признак — для крипто сигналов скринера
        if 'market_crypto' not in enriched and 'market_forex' not in enriched:
            enriched['market_crypto'] = 1.0

        row = []
        for f in features:
            val = enriched.get(f, 0)
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = 0.0
            if val != val:  # NaN
                val = 0.0
            row.append(val)

        X = np.array([row])
        raw_prob = float(model.predict_proba(X)[0][1])

        # Platt scaling (OOB-калиброванный) — применяем если доступен
        if calibrator is not None:
            import numpy as np
            raw_arr = np.array([[raw_prob]])
            prob = float(calibrator.predict_proba(raw_arr)[0][1])
        else:
            prob = raw_prob

        return prob
    except Exception as e:
        print(f"[RF] Ошибка предсказания: {e}")
        return -1.0


def _rf_score_bzone(bzone_feats: dict) -> float:
    """Вероятность WIN для Balance Reversal (8 признаков баланс-зоны)."""
    if _rf_bzone_model is None or not bzone_feats:
        return -1.0
    try:
        import numpy as np
        row = [float(bzone_feats.get(f, 0.0)) for f in _rf_bzone_features]
        X = np.array([row], dtype=np.float32)
        raw_prob = float(_rf_bzone_model.predict_proba(X)[0][1])
        if _rf_bzone_calib is not None:
            raw_prob = float(_rf_bzone_calib.predict_proba(np.array([[raw_prob]]))[0][1])
        return raw_prob
    except Exception as e:
        print(f"[RF_BZONE] Ошибка предсказания: {e}")
        return -1.0


def _rf_is_signal(prob: float, is_renko: bool = False) -> bool:
    """
    True если вероятность выше порога модели.
    Если модель не загружена (prob == -1.0) — не фильтруем.
    """
    if prob < 0:
        return True
    return prob >= _RF_THRESHOLD
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv не установлен — читаем из os.environ напрямую

# Telegram переменные — читаем сразу после загрузки .env
TG_TOKEN   = os.environ.get("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID",  "")

# ─────────────────────────────────────────────────────────────────────────────
# 1. НАСТРОЙКИ
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Pro Order Flow Screener", layout="wide")

if 'session' not in st.session_state:
    st.session_state.session = HTTP(testnet=False)


LIST_ALTS_MAIN = ["XRPUSDT","XLMUSDT","SOLUSDT","TRXUSDT","BCHUSDT","LTCUSDT",
                  "AAVEUSDT","BNBUSDT","AVAXUSDT","ETCUSDT","LINKUSDT","DOGEUSDT",
                  "NEARUSDT","ATOMUSDT","UNIUSDT","POLUSDT","ADAUSDT","DOTUSDT",
                  "ETHUSDT","OPUSDT","FILUSDT","ARBUSDT"]
LIST_ALTS_2    = ["CAKEUSDT","WOOUSDT","VETUSDT","ZILUSDT","STXUSDT","WAVESUSDT",
                  "ZRXUSDT","XTZUSDT","JASMYUSDT","ORDIUSDT","TONUSDT","INJUSDT",
                  "SEIUSDT","PEOPLEUSDT","FLOKIUSDT","YGGUSDT","GALAUSDT","RUNEUSDT",
                  "CRVUSDT","ICPUSDT","MASKUSDT","APEUSDT","SUIUSDT","APTUSDT",
                  "MANAUSDT","THETAUSDT","GMTUSDT","HBARUSDT","SANDUSDT","KSMUSDT"]
LIST_ALTS_3    = ["ANKRUSDT","ASTRUSDT","RVNUSDT","GASUSDT","BOMEUSDT","ONTUSDT",
                  "MEMEUSDT","ARUSDT","BANDUSDT","CKBUSDT","KAVAUSDT","BLURUSDT",
                  "1000PEPEUSDT","GLMUSDT","1000BONKUSDT","PEPEUSDT","WIFUSDT",
                  "POPCATUSDT","DYDXUSDT","SKLUSDT","AGIUSDT","IMXUSDT","LRCUSDT",
                  "MUBARAKUSDT","SUSHIUSDT","XVSUSDT","MAGICUSDT","ZETAUSDT",
                  "ENSUSDT","PENGUUSDT","COTIUSDT"]

# ─────────────────────────────────────────────────────────────────────────────
# 2. ЗАГРУЗКА ДАННЫХ
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _load_crypto_parquet(symbol: str, tf: str) -> pd.DataFrame:
    """
    Загружает 1m parquet и ресемплирует в нужный TF.
    TTL=300с — файл меняется только когда download_data.py обновляет данные.
    """
    from pathlib import Path as _Path
    parquet_path = _Path(__file__).parent / 'backtest_data' / f'{symbol}_1m.parquet'
    if not parquet_path.exists():
        return pd.DataFrame()
    try:
        tf_rule = {'15m': '15min', '1h': '1h', '1D': '1D'}.get(tf, '15min')
        # Берём только нужный хвост: 15m→35 дней, 1h→100 дней, 1D→730 дней
        days = {'15m': 35, '1h': 100, '1D': 730}.get(tf, 35)

        df1m = pd.read_parquet(parquet_path,
                               columns=['timestamp','open','high','low','close','volume','turnover'])
        # Убираем timezone (parquet хранит UTC tz-aware, API даёт naive)
        if df1m['timestamp'].dt.tz is not None:
            df1m['timestamp'] = df1m['timestamp'].dt.tz_localize(None)

        cutoff = df1m['timestamp'].max() - pd.Timedelta(days=days)
        df1m = df1m[df1m['timestamp'] >= cutoff].copy()

        df = (df1m.set_index('timestamp')
                  .resample(tf_rule)
                  .agg(open=('open','first'), high=('high','max'),
                       low=('low','min'), close=('close','last'),
                       volume=('volume','sum'), turnover=('turnover','sum'))
                  .dropna(subset=['open'])
                  .reset_index())
        return df
    except Exception as _e:
        print(f"[parquet] {symbol} {tf}: {_e}")
        return pd.DataFrame()


@st.cache_data(ttl=15)
def fetch_main_data(symbol, tf, is_renko, src):
    # Renko требует больше баров; свечи: 30 дней на 15m = 2880, 1h = 720, 1D = 365
    limit = 5000 if is_renko else {'15m': 2880, '1h': 720, '1D': 365}.get(tf, 2880)
    if src == "crypto":
        tf_m = {"15m": "15", "1h": "60", "1D": "D"}

        # ── 1. Исторические данные из локального parquet ─────────────────────
        df_hist = _load_crypto_parquet(symbol, tf)

        # ── 2. Свежие данные из Bybit API ────────────────────────────────────
        # Если parquet есть → 1 страница (≈10 дней, только свежак)
        # Если parquet нет (VPS без файлов) → 3 страницы (≈30 дней, полная история)
        api_pages = 1 if not df_hist.empty else max(1, limit // 1000)
        all_data, end_time = [], None
        for _ in range(api_pages):
            try:
                params = {"category": "linear", "symbol": symbol,
                          "interval": tf_m[tf], "limit": 1000}
                if end_time:
                    params["end"] = end_time
                res = st.session_state.session.get_kline(**params)\
                        .get('result', {}).get('list', [])
                if not res: break
                all_data.extend(res)
                end_time = int(res[-1][0]) - 1
            except: break

        df_live = pd.DataFrame()
        if all_data:
            df_live = pd.DataFrame(all_data, columns=['ts','o','h','l','c','v','t'])
            df_live['ts'] = pd.to_datetime(pd.to_numeric(df_live['ts']), unit='ms')
            df_live = df_live.sort_values('ts').reset_index(drop=True)
            for c in ['o','h','l','c','v','t']: df_live[c] = df_live[c].astype(float)
            df_live.columns = ['timestamp','open','high','low','close','volume','turnover']

        # ── 3. Склейка: parquet (история) + API (свежак) ─────────────────────
        if not df_hist.empty and not df_live.empty:
            # Берём из parquet только то, что раньше первой API-свечи
            api_start = df_live['timestamp'].min()
            df_old = df_hist[df_hist['timestamp'] < api_start]
            df = pd.concat([df_old, df_live], ignore_index=True)
            df = df.sort_values('timestamp').reset_index(drop=True)
        elif not df_hist.empty:
            df = df_hist
        elif not df_live.empty:
            df = df_live
        else:
            return pd.DataFrame()

        # Возвращаем последние limit баров
        return df.tail(limit).reset_index(drop=True)

    else:
        # ── Форекс: parquet из cache_forex или yfinance fallback ─────────────
        from pathlib import Path as _Path
        # Формат имени файла: EUR_USD_15m.parquet
        sym_key = symbol.replace('/', '_').replace('-', '_')
        fx_path = _Path(__file__).parent / 'backtest_data' / 'cache_forex' / f'{sym_key}_{tf}.parquet'
        if fx_path.exists():
            try:
                df = pd.read_parquet(fx_path)
                if df['timestamp'].dt.tz is not None:
                    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
                df = df.sort_values('timestamp').reset_index(drop=True)
                return df.tail(limit).reset_index(drop=True)
            except Exception as _e:
                print(f"[forex parquet] {fx_path.name}: {_e}")

        # Fallback: yfinance
        tf_y = {"15m": "15m", "1h": "1h", "1D": "1d"}
        try:
            df = yf.download(symbol, interval=tf_y[tf], period="max", progress=False)
            df = df.reset_index()
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df = df.rename(columns={'Datetime':'timestamp','Date':'timestamp',
                                    'Open':'open','High':'high','Low':'low',
                                    'Close':'close','Volume':'volume'})
            if 'timestamp' in df.columns:
                ts = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
                df['timestamp'] = ts
            df = df.reset_index(drop=True)
            return df.tail(limit).reset_index(drop=True)
        except: return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_daily_levels(symbol, src):
    if src != "crypto": return None
    try:
        d = st.session_state.session.get_kline(
            category="linear", symbol=symbol, interval="D", limit=2
        ).get('result', {}).get('list', [])
        return {'pdh': float(d[1][2]), 'pdl': float(d[1][3])} if len(d) >= 2 else None
    except: return None

# ─────────────────────────────────────────────────────────────────────────────
# 2б. TELEGRAM + ФОНОВЫЙ СКАНЕР
# ─────────────────────────────────────────────────────────────────────────────

def tg_send(text):
    """Отправляет сообщение в Telegram. Логирует ошибки."""
    if not TG_TOKEN or not TG_CHAT_ID:
        print(f"[TG] Токен или chat_id не настроен — сообщение не отправлено")
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        resp = requests.post(url, json={
            "chat_id":    TG_CHAT_ID,
            "text":       text,
            "parse_mode": "HTML"
        }, timeout=10)
        if not resp.ok:
            print(f"[TG] Ошибка отправки: {resp.status_code} {resp.text[:200]}")
        else:
            print(f"[TG] Сообщение отправлено: {text[:60]}...")
    except Exception as e:
        print(f"[TG] Exception: {e}")


def format_signal_message(r):
    """Форматирует сигнал скринера в красивое TG сообщение."""

    # ── PRE-SIGNAL (радар) — отдельный формат
    if r.get('strategy') == "Pre-Signal":
        sigs     = "\n".join(f"  👁 {s}" for s in r['signals'])
        amt_hint = f"\n💡 <i>{r.get('amt_hint','')}</i>" if r.get('amt_hint') else ""
        return (
            f"🟡 <b>PRE-SIGNAL: {r['symbol']}</b>\n"
            f"Цена: <code>{r['price']:.5f}</code>  |  {r.get('context','')}\n"
            f"─────────────────\n"
            f"<b>Ситуации для наблюдения:</b>\n{sigs}\n"
            f"─────────────────\n"
            f"<i>👁 Без подтверждений — решай сам</i>"
            f"{amt_hint}\n"
            f"<i>⏱ {r['timestamp']}</i>"
        )

    # ── ENTRY SIGNAL — полный формат
    dir_emoji   = "🟢 LONG" if r['trade_dir'] == 1 else "🔴 SHORT"
    stars       = "⭐" * min(r['score'] // 2, 5)
    tvx         = " + ".join(r.get('tvx', []))
    sigs        = "\n".join(f"  ✅ {s}" for s in r['signals'])
    is_clean     = r.get('strategy') == "Clean Retest"
    is_trans     = r.get('phase_trans', False)
    type_label   = "⚠️ Clean Retest" if is_clean else f"✅ {r.get('strategy','')}"
    trans_label  = "\n🔄 <b>Phase Transition!</b>" if is_trans else ""
    rr_warn      = "\n⚠️ <i>Clean Retest — используй меньший объём</i>" if is_clean else ""
    amt_hint     = f"\n💡 <i>{r.get('amt_hint','')}</i>" if r.get('amt_hint') else ""

    msg = (
        f"<b>{'⚠️' if is_clean else '🚨'} СИГНАЛ: {r['symbol']}</b>\n"
        f"{dir_emoji}  {stars}  {type_label}  [score: {r['score']}]\n"
        f"─────────────────\n"
        f"<b>Контекст:</b> {r.get('context','—')}\n"
        f"<b>ТВХ:</b> {tvx}\n"
        f"─────────────────\n"
        f"<b>Entry:</b>  <code>{r['price']:.5f}</code>\n"
        f"<b>Stop:</b>   <code>{r.get('suggested_sl', '—')}</code>\n"
        f"<b>Take:</b>   <code>{r.get('suggested_tp', '—')}</code>\n"
        f"<b>R/R:</b>    1:{r.get('rr', '—')}\n"
        f"─────────────────\n"
        f"<b>Сигналы:</b>\n{sigs}\n"
        f"─────────────────\n"
        f"<i>⏱ {r['timestamp']}</i>"
        f"{rr_warn}"
        f"{trans_label}"
        f"{amt_hint}"
    )
    return msg


# Хранилище уже отправленных сигналов
_sent_signals: dict = {}

# Кулдаун Balance Reversal: {(symbol, tf): timestamp последнего сигнала}
_bzone_cooldown: dict = {}
# Кулдаун в секундах по TF (из бэктеста v3.2: 1h→6 баров, 15m→6 баров, 1D→3 бара)
_BZONE_COOLDOWN_SEC = {'15m': 5400, '1h': 21600, '1D': 259200}

# Активные Balance Reversal сигналы для мониторинга безубытка
# {key: {'symbol','tf','entry','sl','tp','direction','be_level','be_sent','ts'}}
_active_bzone_signals: dict = {}

def _collect_rf_features_live(df, symbol, tf, direction,
                               phase, key_high, key_low,
                               lc, vah, val, poc,
                               setup_name=None):
    """
    Собирает признаки для RF фильтра из live данных скринера.
    Синхронизировано с RF модель v2.2 (train_rf_smc.py, обучена на v3.6 бэктесте).
    86 признаков: 61 rf_* + 22 pat_* + tf_hours + btc_trend_1d + btc_atr_ratio_1h.
    """
    feats = {}
    try:
        # Базовые ценовые признаки
        feats['price_position'] = float((lc - key_low) / (key_high - key_low))                                    if key_high > key_low else 0.5
        feats['is_long']   = 1 if direction == 'LONG' else 0
        feats['is_short']  = 1 if direction == 'SHORT' else 0
        feats['is_trend_up']   = 1 if phase == 'TREND_UP' else 0
        feats['is_trend_down'] = 1 if phase == 'TREND_DOWN' else 0
        feats['is_balance']    = 1 if phase == 'BALANCE' else 0

        # Volume Profile (v3.6: ATR-нормализованные расстояния)
        # ATR вычисляется ниже, поэтому используем временный расчёт
        _atr_vp = float((df['high'] - df['low']).rolling(14, min_periods=1).mean().iloc[-1]) \
                  if len(df) >= 14 else 0.0
        if val and vah and val > 0 and _atr_vp > 0:
            feats['dist_to_val_atr'] = (lc - val) / _atr_vp
            feats['dist_to_vah_atr'] = (lc - vah) / _atr_vp
            feats['va_width_atr']    = (vah - val) / _atr_vp
        else:
            feats['dist_to_val_atr'] = feats['dist_to_vah_atr'] = feats['va_width_atr'] = 0.0
        if poc and poc > 0 and _atr_vp > 0:
            feats['dist_to_poc_atr'] = (lc - poc) / _atr_vp
            feats['poc_value_vp']    = poc
        else:
            feats['dist_to_poc_atr'] = 0.0
        feats['val_value'] = val if val else 0
        feats['vah_value'] = vah if vah else 0

        # OTE зона (0.62-0.79 от диапазона)
        if key_high > key_low:
            ote_low  = key_low + (key_high - key_low) * 0.62
            ote_high = key_low + (key_high - key_low) * 0.79
            feats['is_ote_zone'] = 1 if ote_low <= lc <= ote_high else 0
        else:
            feats['is_ote_zone'] = 0

        # ATR и свечные признаки из последнего бара
        if len(df) >= 14:
            atr = float((df['high'] - df['low']).rolling(14, min_periods=1).mean().iloc[-1])
            feats['atr_current']    = atr
            feats['atr_at_entry']   = atr
            feats['atr_ratio']      = atr / lc if lc > 0 else 0

            last = df.iloc[-1]
            body      = abs(float(last['close']) - float(last['open']))
            candle_rng = float(last['high']) - float(last['low'])
            feats['range_vs_atr']     = candle_rng / atr if atr > 0 else 0
            feats['lower_wick_ratio'] = (min(float(last['open']), float(last['close'])) - float(last['low'])) / candle_rng if candle_rng > 0 else 0
            feats['upper_wick_ratio'] = (float(last['high']) - max(float(last['open']), float(last['close']))) / candle_rng if candle_rng > 0 else 0

        # CVD / Order Flow
        if 'delta' in df.columns:
            feats['delta_at_entry']  = float(df['delta'].iloc[-1])
            feats['delta_pressure']  = float(df['delta'].rolling(3).sum().iloc[-1])
        if 'cvd' in df.columns:
            cvd_vals = df['cvd'].dropna()
            if len(cvd_vals) >= 5:
                feats['cvd_slope'] = float(cvd_vals.iloc[-1] - cvd_vals.iloc[-5]) / 5

        # Объём
        if 'volume' in df.columns and len(df) >= 20:
            avg_vol = float(df['volume'].rolling(20).mean().iloc[-1])
            feats['volume_vs_avg'] = float(df['volume'].iloc[-1]) / avg_vol if avg_vol > 0 else 1.0

        # Renko кирпичи (из существующей функции если есть)
        try:
            renko_df = build_renko(df, symbol)
            feats['renko_bricks'] = len(renko_df) if renko_df is not None else 0
        except:
            feats['renko_bricks'] = 0

        # Сессионное время
        import datetime as dt
        now_utc = dt.datetime.utcnow()
        h = now_utc.hour + now_utc.minute / 60
        feats['is_asian']       = 1 if 0 <= h < 7 else 0
        feats['is_london']      = 1 if 7 <= h < 16 else 0
        feats['is_ny']          = 1 if 13 <= h < 21 else 0
        feats['is_dead_zone']   = 1 if 16.5 <= h < 18.5 else 0
        feats['is_london_open'] = 1 if 7 <= h < 10 else 0
        feats['is_ny_open']     = 1 if 13 <= h < 15 else 0
        feats['is_overlap']     = 1 if 13 <= h < 16 else 0

        # SL distance (ключевой признак #1, важность 39.6%)
        # Рассчитываем из реального SL (key_high/low), нормализуем на ATR
        _atr_sl = feats.get('atr_current', _atr_vp) or _atr_vp
        if _atr_sl > 0 and key_high > key_low:
            if direction == 'LONG':
                feats['sl_distance_atr'] = (lc - key_low) / _atr_sl
            else:
                feats['sl_distance_atr'] = (key_high - lc) / _atr_sl
            feats['sl_distance_atr'] = max(0.1, min(feats['sl_distance_atr'], 10.0))
        else:
            feats['sl_distance_atr'] = 1.5  # медиана по всем паттернам

        # dpoc (ATR-нормализация v3.6)
        feats['dpoc_value'] = poc if poc else 0
        _atr_dpoc = feats.get('atr_current', _atr_vp) or _atr_vp
        if poc and poc > 0 and _atr_dpoc > 0:
            feats['dpoc_vs_price_atr'] = (lc - poc) / _atr_dpoc
        else:
            feats['dpoc_vs_price_atr'] = 0.0

        # ZigZag market phase (Дядя Миша: pivot_compression + vol_ratio)
        try:
            _zz_pv = calc_zigzag(df)
            _mp    = calc_market_phase(df, _zz_pv, tf)
            feats['vol_ratio']         = _mp['vol_ratio']
            feats['pivot_compression'] = 1 if _mp['pivot_compression'] else 0
        except Exception:
            feats['vol_ratio']         = 1.0
            feats['pivot_compression'] = 0

        # MTF Alignment (v3.6): HTF данные не передаются в live-скринере → 0
        feats['htf_trend_1h'] = 0.0
        feats['htf_bos_1h']   = 0.0

        # Approach Momentum (v3.6)
        try:
            _hi_lo = df['high'] - df['low']
            _atr3  = float(_hi_lo.iloc[-3:].mean())  if len(df) >= 3  else 1.0
            _atr50 = float(_hi_lo.iloc[-50:].mean()) if len(df) >= 50 else _atr3
            feats['approach_momentum'] = _atr3 / _atr50 if _atr50 > 0 else 1.0
        except Exception:
            feats['approach_momentum'] = 1.0

        # Delta vs Volume (v3.6)
        try:
            if 'delta' in df.columns and 'volume' in df.columns:
                _vol = float(df['volume'].iloc[-1])
                _del = float(df['delta'].iloc[-1])
                feats['delta_vs_volume'] = _del / _vol if _vol > 0 else 0.0
            else:
                feats['delta_vs_volume'] = 0.0
        except Exception:
            feats['delta_vs_volume'] = 0.0

        # BOS/FVG Age (v3.6) — нейтральные в live (нет кэша _bos_ob_fvg)
        feats['bos_age_bars'] = 10.0  # умеренно свежий
        feats['fvg_age_bars'] = 10.0
        # Altcoin-BTC correlation (v3.6) — нейтральное значение в live
        feats['altcoin_btc_corr'] = 0.5

        # ── tf_hours (признак важности #6) ───────────────────────────────────
        _tf_map = {'15m': 0.25, '1h': 1.0, '1D': 24.0}
        feats['tf_hours'] = _tf_map.get(tf, 1.0)

        # ── pat_* one-hot encoding (признаки #2-4 по важности) ───────────────
        # Маппинг app.py setup_name → backtest pattern name
        _SETUP_TO_PATTERN = {
            'Breakout Retest':          ('Breakout Retest Long', 'Breakout Retest Short'),
            'Continuation (FTR Demand)':('Continuation FTR Long', None),
            'Continuation (FTR Supply)':(None, 'Continuation FTR Short'),
            'Liq.Grab (Key High Sweep)':(None, 'Liq.Grab Key High'),
            'Liq.Grab (Key Low Sweep)': ('Liq.Grab Key Low', None),
            'Balance Reversal':         ('Balance Zone Breakout Long', 'Balance Zone Breakout Short'),
        }
        _ALL_PATTERNS = [
            'Absorption Reversal', 'Balance Zone Breakout Long', 'Balance Zone Breakout Short',
            'Breakout Retest Long', 'Breakout Retest Short', 'CVD Absorption Long',
            'CVD Absorption Short', 'CVD Exhaustion Long', 'CVD Exhaustion Short',
            'Continuation FTR Long', 'Continuation FTR Short', 'Failed Auction',
            'FVG Retest Long', 'FVG Retest Short', 'Institutional Reversal Long',
            'Institutional Reversal Short', 'Liq.Grab Key High', 'Liq.Grab Key Low',
            'VA Rejection Long', 'VA Rejection Short', 'dPOC Divergence Long', 'dPOC Divergence Short',
        ]
        for _pat in _ALL_PATTERNS:
            feats[f'pat_{_pat}'] = 0.0
        if setup_name and setup_name in _SETUP_TO_PATTERN:
            _long_pat, _short_pat = _SETUP_TO_PATTERN[setup_name]
            _matched = _long_pat if direction == 'LONG' else _short_pat
            if _matched:
                feats[f'pat_{_matched}'] = 1.0

        # ── BTC Regime Features (v2.1) — загружаем из готового parquet ───────
        try:
            import pandas as _pd
            _btc_1d = _pd.read_parquet('D:/MyScreener/BTCUSDT_1D.parquet',
                                        columns=['timestamp','close'])
            _btc_1d = _btc_1d.sort_values('timestamp').tail(100)
            _ema50  = _btc_1d['close'].ewm(span=50, adjust=False).mean().iloc[-1]
            feats['btc_trend_1d'] = 1.0 if float(_btc_1d['close'].iloc[-1]) > _ema50 else -1.0
        except Exception:
            feats['btc_trend_1d'] = 0.0

        try:
            import pandas as _pd
            _btc_1h = _pd.read_parquet('D:/MyScreener/BTCUSDT_1h.parquet',
                                        columns=['timestamp','high','low','close'])
            _btc_1h = _btc_1h.sort_values('timestamp').tail(200)
            _prev   = _btc_1h['close'].shift(1)
            _tr     = (_btc_1h['high'] - _btc_1h['low']).combine(
                          (_btc_1h['high'] - _prev).abs(), max).combine(
                          (_btc_1h['low']  - _prev).abs(), max)
            _atr14  = _tr.ewm(span=14, adjust=False).mean()
            _atr100 = _atr14.rolling(100, min_periods=20).mean().iloc[-1]
            feats['btc_atr_ratio_1h'] = float(np.clip(_atr14.iloc[-1] / _atr100, 0.1, 5.0)) \
                                         if _atr100 > 0 else 1.0
        except Exception:
            feats['btc_atr_ratio_1h'] = 1.0

    except Exception as e:
        print(f"[RF] _collect_rf_features_live ошибка: {e}")

    return feats


def _collect_rf_features_bzone(df, direction, key_high, key_low, lc, vah, val, poc, tf):
    """
    8 признаков для rf_model_balance_zone (синхронизировано с train_balance_zone_rf.py):
    sweep_atr, poc_r, sl_dist, va_width_r, poc_pos, is_long, tf_15m, tf_1h.
    """
    try:
        import numpy as np
        atr14 = float((df['high'] - df['low']).rolling(14, min_periods=1).mean().iloc[-1]) \
                if len(df) >= 2 else abs(lc * 0.01)
        if atr14 <= 0:
            atr14 = abs(lc * 0.01)

        is_long  = 1 if direction == 'LONG' else 0
        sl_level = key_low if is_long else key_high
        sl_dist  = abs(lc - sl_level) if (sl_level and not np.isnan(sl_level)) else atr14
        if sl_dist <= 0:
            sl_dist = atr14

        # Глубина свипа: насколько wick пробил границу зоны
        if is_long:
            sweep_depth = max(0.0, float(key_low) - float(df['low'].iloc[-1])) \
                          if (key_low and not np.isnan(key_low)) else 0.0
        else:
            sweep_depth = max(0.0, float(df['high'].iloc[-1]) - float(key_high)) \
                          if (key_high and not np.isnan(key_high)) else 0.0
        sweep_atr = sweep_depth / atr14

        poc_r = abs(poc - lc) / sl_dist \
                if (poc and not np.isnan(poc) and sl_dist > 0) else 0.5
        va_width_r = (vah - val) / sl_dist \
                     if (vah and val and not np.isnan(vah) and not np.isnan(val) and sl_dist > 0) else 1.0
        if vah and val and not np.isnan(vah) and not np.isnan(val) and (vah - val) > 0:
            poc_pos = (poc - val) / (vah - val) if (poc and not np.isnan(poc)) else 0.5
        else:
            poc_pos = 0.5

        return {
            'sweep_atr':  float(np.clip(sweep_atr,  0.0, 10.0)),
            'poc_r':      float(np.clip(poc_r,       0.0, 10.0)),
            'sl_dist':    float(sl_dist),
            'va_width_r': float(np.clip(va_width_r, 0.0, 20.0)),
            'poc_pos':    float(np.clip(poc_pos,     0.0,  1.0)),
            'is_long':    float(is_long),
            'tf_15m':     1.0 if tf == '15m' else 0.0,
            'tf_1h':      1.0 if tf == '1h'  else 0.0,
        }
    except Exception as e:
        print(f"[RF_BZONE] Ошибка сбора признаков: {e}")
        return {}


def tg_route_and_send(alert_data):
    """
    Маршрутизатор Telegram сообщений.
    VALID_SETUP → торговый сигнал ⚡️
    RADAR       → радар пре-сигнал 📡
    """
    symbol    = alert_data.get('symbol', '')
    tf        = alert_data.get('tf', '')
    direction = alert_data.get('dir', '')
    icon      = "🟢" if direction == "LONG" else "🔴"

    if alert_data.get('group') == 'VALID_SETUP':
        setup  = alert_data.get('setup', '')
        phase  = alert_data.get('phase', '')
        price  = alert_data.get('price', 0)
        kh     = alert_data.get('key_high', float('nan'))
        kl     = alert_data.get('key_low',  float('nan'))
        vah    = alert_data.get('vah', float('nan'))
        val    = alert_data.get('val', float('nan'))
        poc    = alert_data.get('poc', float('nan'))
        desc   = '\n'.join(alert_data.get('desc', []))
        ex     = alert_data.get('extra_levels', {})

        # ── КУЛДАУН Balance Reversal (строго по бэктесту v3.2) ───────────────
        if setup == 'Balance Reversal':
            _cd_key  = (symbol, tf)
            _cd_secs = _BZONE_COOLDOWN_SEC.get(tf, 21600)
            _last_bz = _bzone_cooldown.get(_cd_key, 0)
            if time.time() - _last_bz < _cd_secs:
                _mins_left = int((_cd_secs - (time.time() - _last_bz)) / 60)
                print(f"[BZONE] КУЛДАУН {symbol} {tf}: ещё {_mins_left} мин до следующего сигнала")
                return

        # ── ПАТТЕРН-СПЕЦИФИЧНАЯ СТРОКА УРОВНЕЙ ───────────────────────────────
        def _fmt(v): return f"{v:.5f}" if v and not np.isnan(v) else "—"

        if 'zone_low' in ex and 'zone_high' in ex:
            # Зонные паттерны: FTR Demand/Supply, Balance Reversal
            sl_hint = ex.get('sl_hint')
            levels_line = (
                f"📦 <b>{ex.get('label','Зона')}:</b> "
                f"<code>{_fmt(ex['zone_low'])}</code> – <code>{_fmt(ex['zone_high'])}</code>\n"
                f"🛑 <b>SL за зону:</b> <code>{_fmt(sl_hint)}</code>"
            )
        elif 'level' in ex:
            # Уровневые паттерны: Breakout Retest, Liq.Grab
            sl_hint = ex.get('sl_hint')
            levels_line = (
                f"🔑 <b>{ex.get('label','Уровень')}:</b> <code>{_fmt(ex['level'])}</code>\n"
                f"🛑 <b>SL за уровень:</b> <code>{_fmt(sl_hint)}</code>"
            )
        else:
            # Fallback: Key High / Key Low
            levels_line = f"🔑 Key High: <code>{_fmt(kh)}</code> | Key Low: <code>{_fmt(kl)}</code>"

        # VP подсказка (POC / VAH-VAL) если есть
        vp_line = ""
        if poc and not np.isnan(poc):
            vp_line = f"\n💹 <b>POC:</b> <code>{_fmt(poc)}</code>"
            if vah and not np.isnan(vah) and val and not np.isnan(val):
                vp_line += f"  VAH: <code>{_fmt(vah)}</code>  VAL: <code>{_fmt(val)}</code>"

        # ── RF ФИЛЬТР ────────────────────────────────────────────────────────
        bzone_feats = alert_data.get('bzone_rf_features', {})
        rf_feats    = alert_data.get('rf_features', {})
        if bzone_feats and setup in ('Balance Reversal', 'Balance Zone Breakout'):
            rf_prob   = _rf_score_bzone(bzone_feats)
            rf_thresh = _rf_bzone_threshold if _rf_bzone_model is not None else 0.0
        else:
            rf_prob   = _rf_score(rf_feats, is_renko=False) if rf_feats else -1.0
            rf_thresh = _RF_THRESHOLD
        print(f"[RF] {symbol} {tf} {setup}: prob={rf_prob:.3f} threshold={rf_thresh} feats={len(bzone_feats or rf_feats)}")
        if rf_prob >= 0 and rf_thresh > 0:
            if rf_prob < rf_thresh:
                print(f"[RF] ПРОПУЩЕН {symbol} {tf} {setup}: prob={rf_prob:.3f} < {rf_thresh}")
                return

        # ── СТАТИСТИКА БЭКТЕСТА v3.5 для TG ──────────────────────────────────
        def _bt_line(pat, dir_):
            """Формирует строку с историческим WR из бэктеста v3.5."""
            oos_key = (pat, dir_, 'OOS')
            is_key  = (pat, dir_, 'IS')
            oos = _BT_STATS.get(oos_key)
            iss = _BT_STATS.get(is_key)
            if not oos and not iss:
                return ""
            parts = []
            if iss:
                parts.append(f"IS WR <b>{iss['wr']}%</b> N={iss['n']:,}")
            if oos:
                pnl_sign = '+' if oos['pnl'] > 0 else ''
                parts.append(f"OOS WR <b>{oos['wr']}%</b> P&amp;L {pnl_sign}{oos['pnl']:.0f}% N={oos['n']:,}")
            return "📊 <b>Бэктест v3.5 (RR1:3):</b> " + " · ".join(parts)

        bt_stat_line = _bt_line(setup, direction)

        # Рекомендация по RR на основе статистики
        def _rr_hint(dir_):
            if dir_ == 'SHORT':
                return "💡 <b>Рек. RR:</b> 1:3 – 1:5 (SHORT прибылен без фильтра)"
            else:
                return "⚠️ <b>LONG:</b> применяй RF ≥0.55 · рек. RR 1:4+"

        rr_hint_line = _rr_hint(direction)

        # RF строка
        rf_label = ""
        if rf_prob >= 0:
            rf_bar = '█' * int(rf_prob * 10) + '░' * (10 - int(rf_prob * 10))
            rf_color = "✅" if rf_prob >= rf_thresh else ("⚠️" if rf_prob >= rf_thresh - 0.1 else "❌")
            rf_label = f"\n🤖 <b>RF вероятность:</b> {rf_prob*100:.0f}% {rf_color} [{rf_bar}]"

        # ── БЛОК TP/RR/БЕЗУБЫТОК (только для Balance Reversal) ───────────────
        tp_block = ""
        _sig_sl   = None
        _sig_tp   = None
        _sig_be   = None
        _sig_rr   = None
        if setup == 'Balance Reversal':
            _sl_price = ex.get('sl_hint')
            _tp_price = poc if (poc and not np.isnan(poc)) else None
            if _sl_price and price and _sl_price != price:
                _r_dist  = abs(price - _sl_price)
                _sl_pct  = _r_dist / price * 100
                _sig_sl  = _sl_price
                if _tp_price:
                    _rr_val  = abs(_tp_price - price) / _r_dist
                    _sig_rr  = _rr_val
                    _sig_tp  = _tp_price
                    _be_dist = 1.5 * _r_dist
                    _be_price = (price + _be_dist) if direction == 'LONG' else (price - _be_dist)
                    _sig_be   = _be_price
                    _tp_sign  = "+" if direction == 'LONG' else "-"
                    tp_block  = (
                        f"\n─────────────────\n"
                        f"💰 <b>TP (POC):</b> <code>{_fmt(_tp_price)}</code>  <b>{_tp_sign}{_rr_val:.1f}R</b>\n"
                        f"🛑 <b>SL:</b> <code>{_fmt(_sl_price)}</code>  "
                        f"│  R = <code>{_sl_pct:.2f}%</code> от цены\n"
                        f"🔐 <b>Безубыток при:</b> <code>{_fmt(_be_price)}</code> (+1.5R)\n"
                        f"📐 <b>Размер позиции:</b> <code>1% депо / {_sl_pct:.2f}%</code>"
                    )
            # Регистрируем сигнал для мониторинга безубытка
            if _sig_tp and _sig_sl and _sig_be and (_sig_rr or 0) >= 2.0:
                _bz_key = f"{symbol}_{tf}_{round(price, 5)}"
                _active_bzone_signals[_bz_key] = {
                    'symbol':    symbol,
                    'tf':        tf,
                    'entry':     price,
                    'sl':        _sig_sl,
                    'tp':        _sig_tp,
                    'direction': direction,
                    'be_level':  _sig_be,
                    'be_sent':   False,
                    'ts':        time.time(),
                }
            # Обновляем кулдаун
            _bzone_cooldown[(symbol, tf)] = time.time()

        # ── ИТОГОВОЕ СООБЩЕНИЕ ────────────────────────────────────────────────
        bt_block = f"\n{bt_stat_line}" if bt_stat_line else ""
        msg = (
            f"⚡️ <b>СЕТАП | {symbol} | {tf}</b> ⚡️\n\n"
            f"{icon} <b>Направление:</b> {direction}\n"
            f"🎯 <b>Паттерн:</b> {setup}\n"
            f"📊 <b>Фаза рынка:</b> {phase}\n"
            f"💵 <b>Цена входа:</b> <code>{price:.5f}</code>\n"
            f"─────────────────\n"
            f"{levels_line}"
            f"{vp_line}"
            f"{tp_block}"
            f"\n─────────────────\n"
            f"📋 {desc}"
            f"{bt_block}\n"
            f"{rr_hint_line}"
            f"{rf_label}"
        )
        tg_send(msg)

    elif alert_data.get('group') == 'RADAR':
        radar_type = alert_data.get('type', '')
        desc       = alert_data.get('desc', '')
        key_high   = alert_data.get('key_high', float('nan'))
        key_low    = alert_data.get('key_low',  float('nan'))
        phase      = alert_data.get('phase', '')
        score      = alert_data.get('score', 0)
        is_combo   = alert_data.get('combo', False)

        # Скоринг силы сигнала
        score_stars = '⭐' * min(score, 5) if score > 0 else ''

        tips = {
            'Volume Test':      '<i>Ищите CHoCH на младшем ТФ для входа</i>',
            'CVD Divergence':   '<i>Крупный игрок ставит лимиты. Ждите реакцию цены</i>',
            'Structure Sweep':  '<i>Ликвидность собрана. Возможен резкий откат</i>',
        }
        tip = tips.get(radar_type, '')

        kh_str = f"{key_high:.5f}" if not (isinstance(key_high, float) and (key_high != key_high)) else "—"
        kl_str = f"{key_low:.5f}"  if not (isinstance(key_low,  float) and (key_low  != key_low))  else "—"

        combo_header = "🔥 <b>КОМБО-СИГНАЛ</b> | " if is_combo else ""
        msg = (
            f"📡 {combo_header}<b>РАДАР | {symbol} | {tf}</b> 📡\n\n"
            f"{icon} <b>Тип:</b> {radar_type} {score_stars}\n"
            f"🔍 <b>Событие:</b> {desc}\n"
            f"📊 <b>Фаза:</b> {phase}\n"
            f"🔑 KEY HIGH: {kh_str} | KEY LOW: {kl_str}\n\n"
            f"{tip}"
        )
        tg_send(msg)


# ── ИСПРАВЛЕНИЕ 1: отдельный клиент Bybit для фонового потока
# Не используем st.session_state (недоступен вне сессии браузера)
_bg_bybit_client = None

def _get_bg_client():
    global _bg_bybit_client
    if _bg_bybit_client is None:
        _bg_bybit_client = HTTP(testnet=False)
    return _bg_bybit_client


# ── ИСПРАВЛЕНИЕ 1: отдельная функция загрузки БЕЗ @st.cache_data
def get_optimal_limit(tf_string):
    """
    Адаптивная глубина истории по таймфрейму.
    Для структурного анализа нужно достаточно данных чтобы
    построить Key High/Low и FTR зоны.
    """
    if tf_string in ("1", "3", "5"):
        return 1000   # скальпинг — максимум истории
    elif tf_string in ("15", "15m"):
        return 1000   # ~10 дней — хватит для Key High/Low
    elif tf_string in ("30", "60", "1h"):
        return 500    # ~20 дней для 1H
    elif tf_string in ("240", "D", "1D"):
        return 300    # 300 дней для дневки
    return 500


def fetch_data_bg(symbol, tf):
    """
    Загрузка данных для сканирования.
    Использует адаптивный лимит баров через get_optimal_limit.
    """
    tf_m   = {"15m": "15", "1h": "60", "1D": "D"}
    tf_key = tf_m.get(tf, tf)
    limit  = get_optimal_limit(tf_key)
    client = _get_bg_client()
    try:
        params = {"category": "linear", "symbol": symbol,
                  "interval": tf_key, "limit": limit}
        res = client.get_kline(**params).get('result', {}).get('list', [])
        if not res:
            return pd.DataFrame()
        df = pd.DataFrame(res, columns=['ts','o','h','l','c','v','t'])
        df['ts'] = pd.to_datetime(pd.to_numeric(df['ts']), unit='ms')
        df = df.sort_values('ts').reset_index(drop=True)
        for c in ['o','h','l','c','v','t']:
            df[c] = df[c].astype(float)
        df.columns = ['timestamp','open','high','low','close','volume','turnover']
        df['timestamp_ms'] = df['timestamp'].astype(np.int64) // 10 ** 6
        return df
    except Exception as e:
        print(f"[BG] Ошибка загрузки {symbol} {tf}: {e}")
        return pd.DataFrame()


def _get_last_price(symbol: str) -> float:
    """Возвращает текущую последнюю цену символа через bybit ticker."""
    try:
        client = _get_bg_client()
        res = client.get_tickers(category="linear", symbol=symbol)
        lst = res.get('result', {}).get('list', [])
        if lst:
            return float(lst[0].get('lastPrice', 0))
    except Exception as e:
        print(f"[PRICE] {symbol}: {e}")
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# СКАНЕР ВЕРНЫХ СЕТАПОВ — Группа 1 Сканирования
# Работает строго по 1 ТФ, чистая структурная логика SMC
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# СКАНЕР ВЕРНЫХ СЕТАПОВ — 4 паттерна, строго 1 ТФ, чистая SMC логика
# ─────────────────────────────────────────────────────────────────────────────
def fetch_historical_data(symbol, tf, end_timestamp_ms=None, limit=1000):
    """
    "Машина времени" — загружает данные строго ДО указанного момента времени.
    end_timestamp_ms: UNIX ms (момент входа в сделку).
    Если None — загружает текущие данные.

    По чек-поинту Группы 3: применяет полный анализ (CVD, dPOC, структура)
    чтобы журнал видел тот же анализ что и реальный сканер на момент входа.
    """
    tf_m   = {"15m": "15", "1h": "60", "1D": "D"}
    tf_key = tf_m.get(tf, "15")

    try:
        client_obj = st.session_state.get('session') or _get_bg_client()
    except:
        client_obj = _get_bg_client()

    all_data = []
    cur_end  = end_timestamp_ms

    for _ in range(2):
        try:
            params = {"category": "linear", "symbol": symbol,
                      "interval": tf_key, "limit": 1000}
            if cur_end:
                params["end"] = int(cur_end)
            if hasattr(client_obj, 'get_kline'):
                res = client_obj.get_kline(**params).get('result', {}).get('list', [])
            else:
                break
            if not res:
                break
            all_data.extend(res)
            cur_end = int(res[-1][0]) - 1
        except Exception as e:
            print(f"[HISTORY] {symbol} {tf}: {e}")
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=['ts','o','h','l','c','v','t'])
    df['ts'] = pd.to_datetime(pd.to_numeric(df['ts']), unit='ms')
    df = df.sort_values('ts').reset_index(drop=True)
    for c in ['o','h','l','c','v','t']:
        df[c] = df[c].astype(float)
    df.columns = ['timestamp','open','high','low','close','volume','turnover']
    df['timestamp_ms'] = df['timestamp'].astype(np.int64) // 10 ** 6

    if end_timestamp_ms:
        end_dt = pd.Timestamp(end_timestamp_ms, unit='ms')
        df = df[df['timestamp'] <= end_dt].reset_index(drop=True)

    if df.empty or len(df) < 30:
        return df

    # ── Полный аналитический пайплайн (чек-поинт Группы 3)
    # Тот же анализ что в реальном сканере — для корректного снимка на момент входа
    try:
        # 1. CVD (Дельта Гимельфарба) + Order Flow
        df = apply_order_flow(df)

        # 2. Якорный dPOC
        dpoc_arr = calc_dynamic_poc(df, tf=tf)
        df['dPOC'] = dpoc_arr

        # 3. Williams Fractal-based Key High / Key Low
        kh, kl, trend, struct_lines = _smc_structure_fractal(df)
        df['key_high'] = kh
        df['key_low']  = kl
        df['smc_phase'] = ('TREND_UP'   if trend == 1
                           else 'TREND_DOWN' if trend == -1
                           else 'BALANCE')
        # Кэшируем struct_lines для рендера графика
        try:
            df.attrs['_struct_lines'] = struct_lines
        except:
            pass
    except Exception as e:
        print(f"[HISTORY] Анализ {symbol} {tf}: {e}")

    return df


def scan_valid_setups(symbol, tf):
    """
    Сканер Верных Сетапов.
    Строго 1 таймфрейм, чистая структурная логика HH/HL/LL/LH.
    4 паттерна: Breakout Retest, Continuation FTR, Liq.Grab, Balance Reversal.
    """
    try:
        df = fetch_data_bg(symbol, tf)
        if df is None or df.empty or len(df) < 60:
            return None

        df = apply_order_flow(df)

        (in_bal, t_dir, key_high, key_low,
         vah, val, poc, bko, structure, kh_list, kl_list) = get_struct_levels(df, tf=tf, sw=3)

        phase = 'BALANCE' if in_bal else ('TREND_UP' if t_dir == 1 else 'TREND_DOWN')
        zns   = calc_ftr_zones(df, **get_ftr_params(symbol))
        lc    = float(df['close'].iloc[-1])

        setup_name = None
        direction  = None
        desc_list  = []

        # ── ФИЛЬТР: минимальный диапазон Key High/Low
        # Диапазон < 0.5% = структура слишком сжата, нет смысла в сигнале
        # Диапазон > 15%  = аномальный выброс, пропускаем
        if not (np.isnan(key_high) or np.isnan(key_low)):
            kh_kl_range_pct = abs(key_high - key_low) / key_low * 100
            if kh_kl_range_pct < 0.5:
                return None  # диапазон слишком мал
            if kh_kl_range_pct > 15.0:
                return None  # аномальный диапазон

            # Доп.фильтр: цена должна быть ВНУТРИ диапазона Key H/L
            # (не выше Key High и не ниже Key Low — это уже пробой)
            if lc > key_high * 1.005 or lc < key_low * 0.995:
                return None  # цена вышла за структуру

        # extra_levels — паттерн-специфичные уровни для Telegram
        extra_levels = {}

        # ── ПАТТЕРН: Balance Reversal (единственная активная стратегия)
        # Паттерны 1-3 (Breakout Retest, FTR Continuation, Liq.Grab) отключены —
        # бот торгует строго по верифицированному бэктесту v3.2 Balance Zone.
        if phase == 'BALANCE':
            if not np.isnan(key_low) and _sweep_confirmed(df, key_low, 'bullish'):
                setup_name = "Balance Reversal"
                direction  = "LONG"
                desc_list  = [f"Свип Range Low {key_low:.4f} в балансе"]
                extra_levels = {'label': 'Баланс', 'zone_low': key_low, 'zone_high': key_high, 'sl_hint': key_low}
            elif not np.isnan(key_high) and _sweep_confirmed(df, key_high, 'bearish'):
                setup_name = "Balance Reversal"
                direction  = "SHORT"
                desc_list  = [f"Свип Range High {key_high:.4f} в балансе"]
                extra_levels = {'label': 'Баланс', 'zone_low': key_low, 'zone_high': key_high, 'sl_hint': key_high}

        if not setup_name:
            return None

        # ── Собираем признаки для RF фильтра ────────────────────────────
        rf_feats    = {}
        bzone_feats = {}
        try:
            rf_feats = _collect_rf_features_live(
                df=df, symbol=symbol, tf=tf,
                direction=direction, phase=phase,
                key_high=key_high, key_low=key_low,
                lc=lc, vah=vah, val=val, poc=poc,
                setup_name=setup_name,
            )
        except Exception as e:
            print(f"[RF] Сбор признаков {symbol}: {e}")
        if setup_name == "Balance Reversal":
            try:
                bzone_feats = _collect_rf_features_bzone(
                    df=df, direction=direction,
                    key_high=key_high, key_low=key_low,
                    lc=lc, vah=vah, val=val, poc=poc, tf=tf,
                )
            except Exception as e:
                print(f"[RF_BZONE] Сбор признаков {symbol}: {e}")
        # ─────────────────────────────────────────────────────────────────

        return {
            'group':             'VALID_SETUP',
            'symbol':            symbol,
            'tf':                tf,
            'setup':             setup_name,
            'dir':               direction,
            'phase':             phase,
            'price':             lc,
            'key_high':          key_high,
            'key_low':           key_low,
            'vah': vah, 'val': val, 'poc': poc,
            'desc':              desc_list,
            'rf_features':       rf_feats,
            'bzone_rf_features': bzone_feats,
            'extra_levels':      extra_levels,
        }
    except Exception as e:
        print(f"[SETUP] {symbol} {tf}: {e}")
        return None


    """
    "Машина времени" — загружает данные ДО конкретного момента времени.

    end_timestamp_ms: UNIX timestamp в миллисекундах (момент входа в сделку).
    Если None — загружает текущие данные.

    Автоматически применяет calc_gimelfarb_delta + calc_dynamic_poc
    чтобы график строился с полным анализом на исторический момент.
    """
    tf_m   = {"15m": "15", "1h": "60", "1D": "D"}
    tf_key = tf_m.get(tf, "15")

    # Используем сессионный клиент если доступен, иначе фоновый
    try:
        client_obj = st.session_state.get('session') or _get_bg_client()
    except:
        client_obj = _get_bg_client()

    all_data = []
    cur_end  = end_timestamp_ms

    for _ in range(2):
        try:
            params = {"category": "linear", "symbol": symbol,
                      "interval": tf_key, "limit": 1000}
            if cur_end:
                params["end"] = int(cur_end)

            if hasattr(client_obj, 'get_kline'):
                res = client_obj.get_kline(**params).get('result', {}).get('list', [])
            else:
                break

            if not res:
                break
            all_data.extend(res)
            cur_end = int(res[-1][0]) - 1
        except Exception as e:
            print(f"[HISTORY] {symbol} {tf}: {e}")
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=['ts','o','h','l','c','v','t'])
    df['ts'] = pd.to_datetime(pd.to_numeric(df['ts']), unit='ms')
    df = df.sort_values('ts').reset_index(drop=True)
    for c in ['o','h','l','c','v','t']:
        df[c] = df[c].astype(float)
    df.columns = ['timestamp','open','high','low','close','volume','turnover']

    # Обрезаем строго до момента входа
    if end_timestamp_ms:
        end_dt = pd.Timestamp(end_timestamp_ms, unit='ms')
        df = df[df['timestamp'] <= end_dt].reset_index(drop=True)

    return df


def build_journal_chart(symbol, tf, end_timestamp_ms=None):
    """
    Строит полный Plotly график для скриншота журнала.
    Строго на момент end_timestamp_ms — "машина времени".
    Включает: свечи, CVD, FTR зоны, dPOC, BOS.
    """
    try:
        df = fetch_historical_data(symbol, tf, end_timestamp_ms, limit=1000)
        if df.empty or len(df) < 30:
            return None

        df = apply_order_flow(df)

        # FTR зоны
        zones = calc_ftr_zones(df, **get_ftr_params(symbol))
        active_zones = [z for z in zones if z.get('active')]

        # dPOC
        dpoc_arr = calc_dynamic_poc(df, tf=tf)

        # Структура и уровни
        (in_bal, t_dir, kh, kl, vah, val, poc, _, _, _, _) = get_struct_levels(df, tf=tf, sw=3)

        # BOS
        bos_list, _, _ = find_bos_ob_fvg(df)

        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.75, 0.25],
            shared_xaxes=True,
            vertical_spacing=0.03
        )

        # Свечи
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'], high=df['high'],
            low=df['low'],   close=df['close'],
            increasing_fillcolor='#26a69a', decreasing_fillcolor='#ef5350',
            increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
            name='Price'
        ), row=1, col=1)

        # FTR зоны (только активные)
        for z in active_zones[-6:]:
            color = 'rgba(0,180,60,0.20)' if z['dir'] == 1 else 'rgba(220,30,30,0.20)'
            border = 'rgba(0,200,80,0.9)'  if z['dir'] == 1 else 'rgba(220,30,30,0.9)'
            x0 = df['timestamp'].iloc[min(z['i'], len(df)-1)]
            x1 = df['timestamp'].iloc[-1]
            fig.add_shape(type='rect', x0=x0, x1=x1,
                          y0=z['zl'], y1=z['zh'],
                          fillcolor=color, line=dict(color=border, width=2),
                          row=1, col=1)
            fig.add_annotation(
                x=x1, y=z['zh'],
                text='FTR',
                showarrow=False,
                xanchor='right', yanchor='bottom',
                font=dict(color=border, size=11, family='Arial Black'),
                bgcolor='rgba(0,0,0,0)',
                row=1, col=1)

        # dPOC ступенчатая линия
        dpoc_valid = [(df['timestamp'].iloc[i], float(dpoc_arr[i]))
                      for i in range(len(df)) if not np.isnan(dpoc_arr[i])]
        if dpoc_valid:
            fig.add_trace(go.Scatter(
                x=[v[0] for v in dpoc_valid],
                y=[v[1] for v in dpoc_valid],
                mode='lines',
                line=dict(color='rgba(255,200,0,0.9)', width=1, shape='hv'),
                name='dPOC'
            ), row=1, col=1)

        # VAH/VAL/POC
        for level, name, color in [
            (vah, 'VAH', 'rgba(100,140,255,0.7)'),
            (val, 'VAL', 'rgba(100,140,255,0.7)'),
            (poc, 'POC', 'yellow')
        ]:
            if not np.isnan(level):
                fig.add_hline(y=level,
                              line=dict(color=color, width=1, dash='dot'),
                              annotation_text=name,
                              row=1, col=1)

        # BOS линии
        for b in bos_list[-10:]:
            c = 'rgba(0,220,80,0.8)' if b['dir'] == 1 else 'rgba(220,50,50,0.8)'
            fig.add_shape(type='line',
                          x0=b['x0'], x1=b['x1'], y0=b['y'], y1=b['y'],
                          line=dict(color=c, width=1),
                          row=1, col=1)

        # ZigZag
        try:
            _zz = calc_zigzag(df)
            for _tr in _zigzag_traces(_zz):
                fig.add_trace(_tr, row=1, col=1)
        except Exception:
            pass

        # CVD
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['cvd'],
            line=dict(color='#00E676', width=1),
            name='CVD'
        ), row=2, col=1)

        # Дивергенции
        divs = find_divergences(df, lookback=5, min_dist=8)
        for dv in divs[-5:]:
            c = '#00E5FF' if dv['type'] == 'bull' else '#FF6D00'
            fig.add_shape(type='line',
                          x0=dv['x0'], x1=dv['x1'],
                          y0=dv['yp0'], y1=dv['yp1'],
                          line=dict(color=c, width=2, dash='dot'),
                          row=1, col=1)
            fig.add_shape(type='line',
                          x0=dv['x0'], x1=dv['x1'],
                          y0=dv['yc0'], y1=dv['yc1'],
                          line=dict(color=c, width=2, dash='dot'),
                          row=2, col=1)

        phase = 'BALANCE' if in_bal else ('TREND ↑' if t_dir == 1 else 'TREND ↓')
        fig.update_layout(
            template='plotly_dark',
            title=f"{symbol} | {tf} | {phase} | dPOC={dpoc_valid[-1][1]:.4f}" if dpoc_valid else f"{symbol} | {tf}",
            xaxis_rangeslider_visible=False,
            height=700,
            showlegend=False,
            margin=dict(l=40, r=80, t=50, b=20),
        )

        return fig
    except Exception as e:
        print(f"[CHART] Ошибка построения для журнала: {e}")
        return None


# Алиас — render_full_chart совместим с именем из документации
def render_full_chart(df, symbol='', tf='15m'):
    """
    Алиас build_journal_chart для совместимости.
    Принимает готовый датафрейм вместо загрузки данных.
    """
    try:
        if df is None or df.empty or len(df) < 30:
            return None
        if 'cvd' not in df.columns:
            df = apply_order_flow(df)

        zones      = calc_ftr_zones(df, **get_ftr_params(symbol)) if symbol else []
        active_z   = [z for z in zones if z.get('active')]
        dpoc_arr   = calc_dynamic_poc(df, tf=tf)
        bos_list, _, _ = find_bos_ob_fvg(df)
        (in_bal, t_dir, kh, kl, vah, val, poc, _, _, _, _) = get_struct_levels(df, tf=tf, sw=3)
        divs       = find_divergences(df, lookback=5, min_dist=8)

        fig = make_subplots(rows=2, cols=1, row_heights=[0.8, 0.2],
                            shared_xaxes=True, vertical_spacing=0.03)

        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'], high=df['high'],
            low=df['low'],   close=df['close'],
            increasing_fillcolor='#26a69a', decreasing_fillcolor='#ef5350',
            name='Price'
        ), row=1, col=1)

        for z in active_z[-6:]:
            color  = 'rgba(0,180,60,0.20)'  if z['dir']==1 else 'rgba(220,30,30,0.20)'
            border = 'rgba(0,200,80,0.9)'   if z['dir']==1 else 'rgba(220,30,30,0.9)'
            x0 = df['timestamp'].iloc[min(z['i'], len(df)-1)]
            x1 = df['timestamp'].iloc[-1]
            fig.add_shape(type='rect', x0=x0, x1=x1, y0=z['zl'], y1=z['zh'],
                          fillcolor=color, line=dict(color=border, width=2), row=1, col=1)
            fig.add_annotation(
                x=x1, y=z['zh'],
                text='FTR',
                showarrow=False,
                xanchor='right', yanchor='bottom',
                font=dict(color=border, size=11, family='Arial Black'),
                bgcolor='rgba(0,0,0,0)',
                row=1, col=1)

        dpoc_v = [(df['timestamp'].iloc[i], float(dpoc_arr[i]))
                  for i in range(len(df)) if not np.isnan(dpoc_arr[i])]
        if dpoc_v:
            fig.add_trace(go.Scatter(
                x=[v[0] for v in dpoc_v], y=[v[1] for v in dpoc_v],
                mode='lines', line=dict(color='rgba(255,200,0,0.9)', width=1, shape='hv'),
                name='dPOC'
            ), row=1, col=1)

        for level, name, color in [(vah,'VAH','rgba(100,140,255,0.7)'),
                                    (val,'VAL','rgba(100,140,255,0.7)'),
                                    (poc,'POC','yellow')]:
            if not np.isnan(level):
                fig.add_hline(y=level, line=dict(color=color, width=1, dash='dot'),
                              annotation_text=name, row=1, col=1)

        for b in bos_list[-8:]:
            c = 'rgba(0,220,80,0.7)' if b['dir']==1 else 'rgba(220,50,50,0.7)'
            fig.add_shape(type='line', x0=b['x0'], x1=b['x1'],
                          y0=b['y'], y1=b['y'],
                          line=dict(color=c, width=1), row=1, col=1)

        # ZigZag
        try:
            _zz = calc_zigzag(df)
            for _tr in _zigzag_traces(_zz):
                fig.add_trace(_tr, row=1, col=1)
        except Exception:
            pass

        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['cvd'],
            line=dict(color='#00E676', width=1), name='CVD'
        ), row=2, col=1)

        for dv in divs[-5:]:
            c = '#00E5FF' if dv['type']=='bull' else '#FF6D00'
            for row, y0, y1 in [(1, dv['yp0'], dv['yp1']), (2, dv['yc0'], dv['yc1'])]:
                fig.add_shape(type='line', x0=dv['x0'], x1=dv['x1'], y0=y0, y1=y1,
                              line=dict(color=c, width=2, dash='dot'), row=row, col=1)

        phase = 'BALANCE' if in_bal else ('TREND↑' if t_dir==1 else 'TREND↓')
        fig.update_layout(
            template='plotly_dark',
            title=f"{symbol} | {tf} | {phase}",
            xaxis_rangeslider_visible=False,
            height=700, showlegend=False,
        )
        return fig
    except Exception as e:
        print(f"[RENDER] render_full_chart: {e}")
        return None


    """
    Сканер Верных Сетапов.
    Строго 1 таймфрейм, чистая структурная логика HH/HL/LL/LH.
    4 паттерна: Breakout Retest, Continuation FTR, Liq.Grab, Balance Reversal.
    """
    try:
        df = fetch_data_bg(symbol, tf)
        if df is None or df.empty or len(df) < 60:
            return None

        df = apply_order_flow(df)

        # Структурный анализ (Группа 2)
        (in_bal, t_dir, key_high, key_low,
         vah, val, poc, bko, structure, kh_list, kl_list) = get_struct_levels(df, tf=tf, sw=3)

        phase = 'BALANCE' if in_bal else ('TREND_UP' if t_dir == 1 else 'TREND_DOWN')

        # FTR зоны
        zns = calc_ftr_zones(df, **get_ftr_params(symbol))

        lc       = float(df['close'].iloc[-1])
        atr_val  = float((df['high'] - df['low']).rolling(14).mean().iloc[-1])

        setup_name = None
        direction  = None
        desc_list  = []

        # ── ПАТТЕРН: Balance Reversal (единственная активная стратегия)
        # Паттерны 1-3 (Breakout Retest, FTR Continuation, Liq.Grab) отключены —
        # бот торгует строго по верифицированному бэктесту v3.2 Balance Zone.
        if phase == 'BALANCE':
            if not np.isnan(key_low) and _sweep_confirmed(df, key_low, 'bullish'):
                setup_name = "Balance Reversal"
                direction  = "LONG"
                desc_list  = [f"Свип Range Low {key_low:.4f} в балансе"]
            elif not np.isnan(key_high) and _sweep_confirmed(df, key_high, 'bearish'):
                setup_name = "Balance Reversal"
                direction  = "SHORT"
                desc_list  = [f"Свип Range High {key_high:.4f} в балансе"]

        if not setup_name:
            return None

        return {
            'group':    'VALID_SETUP',
            'symbol':   symbol,
            'tf':       tf,
            'setup':    setup_name,
            'dir':      direction,
            'phase':    phase,
            'price':    lc,
            'key_high': key_high,
            'key_low':  key_low,
            'vah':      vah, 'val': val, 'poc': poc,
            'desc':     desc_list,
        }
    except Exception as e:
        print(f"[SETUP] {symbol} {tf}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# РАДАР ПРЕ-СИГНАЛОВ — Группа 2 Сканирования
# Ищет аномалии: тесты POC/VAL, дивергенции, свипы
# ─────────────────────────────────────────────────────────────────────────────

def scan_radar(symbol, tf):
    """
    Радар Пре-Сигналов.
    Сигналы типа "Внимание — потенциальная ситуация, проверь и прими решение".
    Ищет: тесты объёмных уровней (VAL/VAH/POC),
          дивергенции CVD (свечи + Renko),
          свипы структурных уровней.
    """
    try:
        df = fetch_data_bg(symbol, tf)
        if df is None or df.empty or len(df) < 60:
            return []

        df = apply_order_flow(df)

        (in_bal, t_dir, key_high, key_low,
         vah, val, poc, bko, structure, kh_list, kl_list) = get_struct_levels(df, tf=tf, sw=3)

        lc      = float(df['close'].iloc[-1])
        prev_c  = float(df['close'].iloc[-2])
        signals = []

        # ── 1. Тесты объёмных уровней (VAL / VAH / POC)
        vp = get_current_vp_levels(df, tf=tf)
        tol = 0.002  # 0.2% proximity

        if vp:
            _vah, _val, _poc = vp['vah'], vp['val'], vp['poc']

            # Возврат к VAL сверху
            if prev_c > _val and lc <= _val * (1 + tol):
                signals.append({'type': 'Volume Test',
                                 'desc': f'Возврат к VAL сверху ({_val:.5f})',
                                 'dir':  'LONG'})
            # Тест VAL снизу
            elif prev_c < _val and lc >= _val * (1 - tol):
                signals.append({'type': 'Volume Test',
                                 'desc': f'Тест VAL снизу ({_val:.5f})',
                                 'dir':  'SHORT'})

            # Подход к VAH снизу
            if prev_c < _vah and lc >= _vah * (1 - tol):
                signals.append({'type': 'Volume Test',
                                 'desc': f'Подход к VAH снизу ({_vah:.5f})',
                                 'dir':  'SHORT'})

            # POC после выхода за VAH
            if was_above_vah(df, _vah) and abs(lc - _poc) / lc < tol:
                signals.append({'type': 'Volume Test',
                                 'desc': f'Возврат к POC после выхода за VAH ({_poc:.5f})',
                                 'dir':  'LONG'})
            # POC после выхода под VAL
            elif was_below_val(df, _val) and abs(lc - _poc) / lc < tol:
                signals.append({'type': 'Volume Test',
                                 'desc': f'Возврат к POC после выхода под VAL ({_poc:.5f})',
                                 'dir':  'SHORT'})

        # ── 2. Дивергенции CVD на свечах
        try:
            div_type = check_cvd_divergence(df)
            if div_type == 'BULL_ABSORPTION':
                signals.append({'type': 'CVD Divergence',
                                 'desc': 'Бычье поглощение CVD — лимитник покупает',
                                 'dir':  'LONG'})
            elif div_type == 'BULL_EXHAUSTION':
                signals.append({'type': 'CVD Divergence',
                                 'desc': 'Бычье истощение CVD — продавцы выдохлись',
                                 'dir':  'LONG'})
            elif div_type == 'BEAR_ABSORPTION':
                signals.append({'type': 'CVD Divergence',
                                 'desc': 'Медвежье поглощение CVD — лимитник продаёт',
                                 'dir':  'SHORT'})
            elif div_type == 'BEAR_EXHAUSTION':
                signals.append({'type': 'CVD Divergence',
                                 'desc': 'Медвежье истощение CVD — покупатели выдохлись',
                                 'dir':  'SHORT'})
            elif div_type == 'BULL_HIDDEN':
                signals.append({'type': 'CVD Divergence',
                                 'desc': 'Скрытая бычья дивергенция CVD — продолжение роста',
                                 'dir':  'LONG'})
            elif div_type == 'BEAR_HIDDEN':
                signals.append({'type': 'CVD Divergence',
                                 'desc': 'Скрытая медвежья дивергенция CVD — продолжение падения',
                                 'dir':  'SHORT'})
        except:
            pass

        # ── 4. Свипы структурных уровней
        _sweep_kh = not np.isnan(key_high) and _sweep_confirmed(df, key_high, 'bearish')
        _sweep_kl = not np.isnan(key_low)  and _check_sweep(df, key_low,  'bullish')
        if _sweep_kh:
            signals.append({
                'type': 'Structure Sweep',
                'desc': f'Свип Key High {key_high:.4f} → шорт-реверсия',
                'dir':  'SHORT'
            })
        if _sweep_kl:
            signals.append({
                'type': 'Structure Sweep',
                'desc': f'Свип Key Low {key_low:.4f} → лонг-реверсия',
                'dir':  'LONG'
            })

        # ── 5. Institutional Reversal (Сетап 1: Liq Grab + CVD Absorption)
        try:
            _last_delta = float(df['delta'].iloc[-1]) if 'delta' in df.columns else 0.0
            _last_cl    = float(df['close'].iloc[-1])
            _last_body  = abs(float(df['close'].iloc[-1]) - float(df['open'].iloc[-1]))
            _last_sprd  = float(df['high'].iloc[-1]) - float(df['low'].iloc[-1])
            _effort     = _last_body / _last_sprd if _last_sprd > 0 else 0
            if _sweep_kh and _effort > 0.3 and _last_delta > 0 and _last_cl < key_high:
                signals.append({'type': 'Institutional Reversal',
                                 'desc': f'Институциональный разворот: свип KH + CVD Absorption → шорт',
                                 'dir':  'SHORT'})
            if _sweep_kl and _effort > 0.3 and _last_delta < 0 and _last_cl > key_low:
                signals.append({'type': 'Institutional Reversal',
                                 'desc': f'Институциональный разворот: свип KL + CVD Absorption → лонг',
                                 'dir':  'LONG'})
        except Exception:
            pass

        # ── 6. Balance Zone Breakout (пробой VAH/VAL с дельтой)
        try:
            if len(df) >= 20:
                _delta_l = float(df['delta'].iloc[-1]) if 'delta' in df.columns else 0.0
                if not np.isnan(key_high) and lc > key_high and _delta_l > 0:
                    signals.append({'type': 'Balance Zone Breakout',
                                     'desc': f'Пробой баланса вверх через {key_high:.4f} + delta+ → лонг',
                                     'dir':  'LONG'})
                if not np.isnan(key_low) and lc < key_low and _delta_l < 0:
                    signals.append({'type': 'Balance Zone Breakout',
                                     'desc': f'Пробой баланса вниз через {key_low:.4f} + delta- → шорт',
                                     'dir':  'SHORT'})
        except Exception:
            pass

        if not signals:
            return []

        # ── Скоринг (1-5 звёзд)
        score_map = {
            'Institutional Reversal':   5,
            'Structure Sweep':          5,
            'Balance Zone Breakout':    5,
            'CVD Divergence':           4,
            'Balance Zone Fade':        4,
            'Volume Test':              3,
        }
        price_near_key = False
        if not np.isnan(key_high) and not np.isnan(key_low):
            kh_dist = abs(lc - key_high) / lc * 100
            kl_dist = abs(lc - key_low)  / lc * 100
            price_near_key = min(kh_dist, kl_dist) < 0.5

        phase = 'BALANCE' if in_bal else ('TREND_UP' if t_dir == 1 else 'TREND_DOWN')

        scored = []
        for s in signals:
            base_score = score_map.get(s['type'], 2)
            if price_near_key:
                base_score = min(base_score + 1, 5)
            if phase == 'TREND_DOWN' and s['dir'] == 'LONG' and s['type'] != 'Structure Sweep':
                base_score = max(base_score - 1, 1)
            elif phase == 'TREND_UP' and s['dir'] == 'SHORT' and s['type'] != 'Structure Sweep':
                base_score = max(base_score - 1, 1)
            scored.append({**s, 'score': base_score, 'phase': phase,
                           'key_high': key_high, 'key_low': key_low})

        if len(scored) >= 2:
            combo_desc  = ' + '.join(s['desc'] for s in scored[:2])
            combo_score = min(sum(s['score'] for s in scored[:2]), 5)
            combo = {**scored[0], 'desc': combo_desc, 'score': combo_score, 'combo': True}
            return [{'group': 'RADAR', 'symbol': symbol, 'tf': tf, **combo}]

        return [{'group': 'RADAR', 'symbol': symbol, 'tf': tf, **s}
                for s in scored]
    except Exception as e:
        print(f"[RADAR] {symbol} {tf}: {e}")
        return []

def scan_signal_bg(symbol, tf_scan):
    """
    Фоновый вариант scan_signal.
    Использует fetch_data_bg вместо fetch_main_data (без st.cache_data).
    Логика идентична scan_signal v7.
    """
    try:
        d = fetch_data_bg(symbol, tf_scan)
        if d.empty or len(d) < 60:
            return None
        d = apply_order_flow(d)

        htf_tf = "1h" if tf_scan == "15m" else "1D"
        d_htf  = fetch_data_bg(symbol, htf_tf)
        if d_htf.empty or len(d_htf) < 30:
            return None
        d_htf = apply_order_flow(d_htf)
        # ── Структурный анализ HTF (Key High / Key Low → Баланс → Profile)
        (in_bal_htf, t_dir_htf,
         key_high_htf, key_low_htf,
         vah_htf, val_htf, poc_htf,
         breakout_htf, structure_htf,
         kh_htf, kl_htf) = get_struct_levels(d_htf, tf=htf_tf, sw=3)

        # Единые параметры FTR — та же логика что на графиках
        zns = calc_ftr_zones(d, **get_ftr_params(symbol))

        last_close = float(d['close'].iloc[-1])
        last_open  = float(d['open'].iloc[-1])
        atr_val    = float((d['high'] - d['low']).rolling(14).mean().iloc[-1])

        # ── Структурный анализ LTF
        (in_bal, t_dir,
         key_high_ltf, key_low_ltf,
         vah, val, cur_poc,
         mkt_mode_struct, structure_ltf,
         kh_ltf, kl_ltf) = get_struct_levels(d, tf=tf_scan, sw=3)

        # mkt_mode: структурный breakout вместо profile-based
        mkt_mode = mkt_mode_struct  # "UP"/"DOWN"/None → используем ниже
        bal_high = key_high_ltf if not np.isnan(key_high_ltf) else np.nan
        bal_low  = key_low_ltf  if not np.isnan(key_low_ltf)  else np.nan

        _all_prof = build_balance_profiles(d, tf_scan, max_balances=10)
        hist_pocs = [{'x0': p['x0'], 'x1': p['x1'], 'poc': p['poc']}
                     for p in _all_prof if not p.get('is_current', False)]

        # ── AMT полный анализ (CVD / absorption — остаётся для триггеров)
        amt = analyze_auction_context(d, d_htf)

        # ── HTF контекст через СТРУКТУРУ (не VAH/VAL!)
        if in_bal_htf:
            htf_context = "BALANCE"
            # Позиция цены относительно Key High/Low (не VAH/VAL)
            kh = key_high_htf if not np.isnan(key_high_htf) else vah_htf
            kl = key_low_htf  if not np.isnan(key_low_htf)  else val_htf
            range_size = kh - kl if (not np.isnan(kh) and not np.isnan(kl)) else 0
            price_pos  = (last_close - kl) / range_size if range_size > 0 else 0.5
            if price_pos > 0.75:   trade_dir = -1
            elif price_pos < 0.25: trade_dir = 1
            else:                  trade_dir = None
            # CVD фаза как дополнительный фильтр направления в середине баланса
            if trade_dir is None:
                if amt.get('balance_phase') == "ACCUMULATION":   trade_dir = 1
                elif amt.get('balance_phase') == "DISTRIBUTION": trade_dir = -1
        else:
            htf_context = "TREND"
            trade_dir   = t_dir_htf

        abs_bull_r  = d['abs_bull'].iloc[-5:].any()
        abs_bear_r  = d['abs_bear'].iloc[-5:].any()
        dshift_bull = d['delta_shift_bull'].iloc[-3:].any()
        dshift_bear = d['delta_shift_bear'].iloc[-3:].any()
        dp          = float(d['delta_pressure'].iloc[-1])
        # Свипы: 2-свечное подтверждение (обе закрыты)
        # Свип-свеча = iloc[-3], подтверждение = iloc[-2], вход = iloc[-1]
        # Референс — 20-баровый минимум ДО свипной свечи (iloc[-4])
        _roll20_low_ref  = float(d['low'].rolling(20).min().iloc[-4])
        _roll20_high_ref = float(d['high'].rolling(20).max().iloc[-4])
        _sw_close   = float(d['close'].iloc[-3])   # close свип-свечи
        _cn_close   = float(d['close'].iloc[-2])   # close подтверждающей свечи
        sweep_low   = (float(d['low'].iloc[-3])  < _roll20_low_ref  and
                       _sw_close > _roll20_low_ref  and _cn_close > _sw_close)
        sweep_high  = (float(d['high'].iloc[-3]) > _roll20_high_ref and
                       _sw_close < _roll20_high_ref and _cn_close < _sw_close)
        divs        = find_divergences(d)
        has_bull_div = any(dv['type']=='bull' and
                           len(d)-np.searchsorted(d['timestamp'].values,dv['x1'])<=10
                           for dv in divs)
        has_bear_div = any(dv['type']=='bear' and
                           len(d)-np.searchsorted(d['timestamp'].values,dv['x1'])<=10
                           for dv in divs)
        bos_list, _, _ = find_bos_ob_fvg(d)
        has_bos_bull = any(b['dir']==1  and
                           len(d)-np.searchsorted(d['timestamp'].values,b['x1'])<=5
                           for b in bos_list)
        has_bos_bear = any(b['dir']==-1 and
                           len(d)-np.searchsorted(d['timestamp'].values,b['x1'])<=5
                           for b in bos_list)

        # ══════════════════════════════════════════════════════
        # ИСПРАВЛЕНИЕ 2: единый контекст через AMT (убираем конфликт фаз)
        # ══════════════════════════════════════════════════════
        # Контекст = только структура (не переопределяем через AMT)
        final_context = htf_context

        # ИСПРАВЛЕНИЕ 3: phase transition (деньги именно здесь)
        # Переход между фазами = только Reversal и Liq.Grab, не Continuation
        phase_transition = (
            amt['sot'] or
            amt['deep_pullback'] or
            amt.get('balance_phase') in ["ACCUMULATION", "DISTRIBUTION"]
        )

        # ИСПРАВЛЕНИЕ 1: AMT как жёсткий фильтр
        # Не торгуем Continuation в умирающем тренде
        # Не торгуем лонг в Distribution и шорт в Accumulation
        amt_block_continuation = amt['sot'] or amt['deep_pullback']
        amt_block_long  = amt.get('balance_phase') == "DISTRIBUTION"
        amt_block_short = amt.get('balance_phase') == "ACCUMULATION"

        setup_type = None; signal_dir = None; signals = []; ftr_tc = 0

        # Тип 1 — Reversal
        for z in zns:
            if not z['active']: continue
            if final_context == "TREND" and z['dir'] != trade_dir: continue
            if final_context == "BALANCE" and trade_dir and z['dir'] != trade_dir: continue
            # AMT фильтр направления
            if z['dir'] == 1  and amt_block_long:  continue
            if z['dir'] == -1 and amt_block_short: continue
            inside = z['zl'] <= last_close <= z['zh']
            near   = min(abs(last_close-z['zl']),abs(last_close-z['zh']))/last_close*100 < 0.8
            if not (inside or near): continue
            tc = z.get('touch_count', 0)
            if tc >= 3: continue

            # ── ФИЛЬТР: зона должна быть зрелой (цена уходила из неё после формирования)
            # "Зона без выхода = не зона, а просто свеча" (с)
            bars_since_zone = len(d) - z['i']
            if bars_since_zone < 5:
                continue  # зона только что сформировалась — пропускаем

            # Проверяем: была ли цена ВНЕ зоны хотя бы 1 бар после формирования
            z_slice = d['close'].iloc[z['i']+1:-1]
            if len(z_slice) > 0:
                if z['dir'] == 1:
                    was_outside = (z_slice > z['zh']).any()  # для demand: цена уходила выше
                else:
                    was_outside = (z_slice < z['zl']).any()  # для supply: цена уходила ниже
                if not was_outside:
                    continue  # цена никогда не покидала зону — это рождение, не ретест

            triggers = get_fp_triggers(z['dir'])

            # Минимум 2 триггера — нет подтверждения = нет сигнала
            if len(triggers) < 2:
                continue

            setup_type = "Reversal"; signal_dir = z['dir']; ftr_tc = tc
            dist = min(abs(last_close-z['zl']),abs(last_close-z['zh']))/last_close*100
            touch_str = "1-й тест ⭐" if tc==0 else ("2-й тест 🔥" if tc==1 else "3-й тест")
            signals = [f"HTF:{final_context} {'↑' if t_dir_htf==1 else '↓'}",
                       f"{'Demand' if z['dir']==1 else 'Supply'} FTR {touch_str} ({dist:.2f}%)"] + triggers
            break

        # Тип 2 — Continuation: тренд + откат к уровню ИЛИ shallow pullback
        # В BREAKOUT режиме запрещён — сначала нужен ретест (AMT: acceptance first)
        _cont_allowed = (
            final_context == "TREND" and
            not amt_block_continuation and
            not phase_transition and
            mkt_mode != "BREAKOUT"
        )
        if setup_type is None and _cont_allowed:
            cont_dir  = t_dir_htf
            poc_dist  = abs(last_close-cur_poc)/last_close*100 if not np.isnan(cur_poc) else 999
            val_d     = abs(last_close-val)/last_close*100     if not np.isnan(val)     else 999
            vah_d     = abs(last_close-vah)/last_close*100     if not np.isnan(vah)     else 999
            vhtf_d    = abs(last_close-val_htf)/last_close*100 if not np.isnan(val_htf) else 999
            vahtf_d   = abs(last_close-vah_htf)/last_close*100 if not np.isnan(vah_htf) else 999
            at_level  = False; level_name = ""

            if cont_dir == 1:
                if val_d < 0.8 and not np.isnan(val):
                    valid, _ = is_valid_level_interaction('VAL', last_close, val, d, 1)
                    if valid: at_level=True; level_name="LTF VAL"
                elif vhtf_d < 1.2 and not np.isnan(val_htf):
                    valid, _ = is_valid_level_interaction('VAL', last_close, val_htf, d, 1)
                    if valid: at_level=True; level_name="HTF VAL"
                elif poc_dist < 0.4 and not np.isnan(cur_poc):
                    # POC — только при наличии подтверждения (не случайный вход)
                    if abs_bull_r or dshift_bull:
                        at_level=True; level_name="POC"
            else:
                if vah_d < 0.8 and not np.isnan(vah):
                    valid, _ = is_valid_level_interaction('VAH', last_close, vah, d, -1)
                    if valid: at_level=True; level_name="LTF VAH"
                elif vahtf_d < 1.2 and not np.isnan(vah_htf):
                    valid, _ = is_valid_level_interaction('VAH', last_close, vah_htf, d, -1)
                    if valid: at_level=True; level_name="HTF VAH"
                elif poc_dist < 0.4 and not np.isnan(cur_poc):
                    at_level=True; level_name="POC"

            # Shallow pullback — откат 0.2–0.7 ATR
            if not at_level:
                pullback = abs(float(d['close'].iloc[-1]) - float(d['close'].iloc[-3]))
                if 0.2 * atr_val < pullback < 0.7 * atr_val:
                    at_level=True; level_name="Shallow Pullback"
            if at_level:
                triggers = get_fp_triggers(cont_dir)
                if triggers:
                    setup_type="Continuation"; signal_dir=cont_dir
                    signals=[f"HTF:{final_context} {'↑' if t_dir_htf==1 else '↓'}", f"Откат к {level_name}"]+triggers

        # Тип 3 — Liquidity Grab: sweep ИЛИ касание экстремума + возврат
        # При phase_transition — Liq.Grab получает приоритет
        if setup_type is None:
            grab_dir    = None
            # sweep_low/sweep_high уже включают полное подтверждение (2 свечи)
            # touch: подтверждающая свеча (iloc[-2]) касается уровня
            _cn_low  = float(d['low'].iloc[-2])
            _cn_high = float(d['high'].iloc[-2])
            touch_low  = (not sweep_low  and abs(_cn_low  - _roll20_low_ref)  / last_close * 100 < 0.3
                          and _cn_close > _roll20_low_ref)
            touch_high = (not sweep_high and abs(_cn_high - _roll20_high_ref) / last_close * 100 < 0.3
                          and _cn_close < _roll20_high_ref)
            if (sweep_low  or touch_low)  and (trade_dir==1  or trade_dir is None): grab_dir = 1
            elif (sweep_high or touch_high) and (trade_dir==-1 or trade_dir is None): grab_dir = -1
            if grab_dir is not None:
                returned = True  # sweep_*_full уже проверяет return, touch проверяет выше
                if returned:
                    liq_label = "Sweep" if (sweep_low if grab_dir==1 else sweep_high) else "Касание экстремума"
                    # Sweep + возврат достаточен — без триггеров
                    setup_type = "Liq.Grab"
                    signal_dir = grab_dir
                    signals = [
                        f"HTF:{final_context} {'↑' if t_dir_htf==1 else '↓'}",
                        f"{liq_label} {'Low' if grab_dir==1 else 'High'} ⚡ + возврат",
                    ]

        # ── BREAKOUT RETEST в scan_signal_bg (без триггеров)
        # Структурный breakout: цена вышла за Key High или Key Low
        _struct_bko = mkt_mode  # "UP"/"DOWN"/None
        if setup_type is None and _struct_bko is not None:
            if not np.isnan(vah) and not np.isnan(val):
                vah_d_br = abs(last_close - vah) / last_close * 100
                val_d_br = abs(last_close - val) / last_close * 100
                if last_close > vah and vah_d_br < 0.6:
                    if (d['close'].iloc[-5:-1] > vah).any():
                        setup_type = "Breakout Retest"; signal_dir = -1
                        signals = [f"HTF:{htf_context} {'↑' if t_dir_htf==1 else '↓'}", "🔄 Retest VAH сверху"]
                elif last_close < val and val_d_br < 0.6:
                    if (d['close'].iloc[-5:-1] < val).any():
                        setup_type = "Breakout Retest"; signal_dir = 1
                        signals = [f"HTF:{htf_context} {'↑' if t_dir_htf==1 else '↓'}", "🔄 Retest VAL снизу"]
                elif not np.isnan(cur_poc):
                    poc_d_br = abs(last_close - cur_poc) / last_close * 100
                    if poc_d_br < 0.4:
                        br_dir = 1 if (abs_bull_r or dshift_bull) else (-1 if (abs_bear_r or dshift_bear) else t_dir_htf)
                        setup_type = "Breakout Retest"; signal_dir = br_dir
                        signals = [f"HTF:{htf_context} {'↑' if t_dir_htf==1 else '↓'}", "🔄 Retest POC"]

        # ══════════════════════════════════════════════════
        # PRE-SIGNAL: радар без подтверждений
        # Цель: "дай мне ситуации — я сам решу"
        # ══════════════════════════════════════════════════
        pre_signals = []

        # PRE: подход к структурным уровням
        d_vah = abs(last_close-vah)/last_close*100 if not np.isnan(vah) else 999
        d_val = abs(last_close-val)/last_close*100 if not np.isnan(val) else 999
        d_poc = abs(last_close-cur_poc)/last_close*100 if not np.isnan(cur_poc) else 999

        if d_vah < 0.5:
            pre_signals.append(f"PRE: подход к VAH ({d_vah:.2f}%) [{mkt_mode or '?'}]")
        if d_val < 0.5:
            pre_signals.append(f"PRE: подход к VAL ({d_val:.2f}%) [{mkt_mode or '?'}]")
        if d_poc < 0.4:
            pre_signals.append(f"PRE: у POC ({d_poc:.2f}%)")

        # PRE: подход к FTR зонам
        for z in zns:
            if not z['active']: continue
            z_dist = min(abs(last_close-z['zl']),abs(last_close-z['zh']))/last_close*100
            if z_dist < 0.5:
                pre_signals.append(
                    f"PRE: {'Demand' if z['dir']==1 else 'Supply'} FTR ({z_dist:.2f}%)"
                )
                break

        # PRE: Breakout возврат (без триггеров)
        if mkt_mode == "BREAKOUT" and not np.isnan(vah) and not np.isnan(val):
            if last_close > vah and d_vah < 0.6:
                was_above = (d['close'].iloc[-5:-1] > vah).any()
                if was_above:
                    pre_signals.append(f"PRE: Retest VAH сверху ({d_vah:.2f}%)")
            elif last_close < val and d_val < 0.6:
                was_below = (d['close'].iloc[-5:-1] < val).any()
                if was_below:
                    pre_signals.append(f"PRE: Retest VAL снизу ({d_val:.2f}%)")

        # PRE: Liquidity Sweep + подтверждение (2 закрытые свечи)
        # Свип-свеча = iloc[-3], подтверждение = iloc[-2], реф = iloc[-4]
        _pre_low_ref   = float(d['low'].rolling(20).min().iloc[-4])
        _pre_high_ref  = float(d['high'].rolling(20).max().iloc[-4])
        _pre_sw_cl     = float(d['close'].iloc[-3])
        _pre_cn_cl     = float(d['close'].iloc[-2])
        sweep_low_pre  = (float(d['low'].iloc[-3])  < _pre_low_ref  and
                          _pre_sw_cl > _pre_low_ref  and _pre_cn_cl > _pre_sw_cl)
        sweep_high_pre = (float(d['high'].iloc[-3]) > _pre_high_ref and
                          _pre_sw_cl < _pre_high_ref and _pre_cn_cl < _pre_sw_cl)
        if sweep_low_pre:
            pre_signals.append("PRE: Sweep Low + подтверждение ⚡")
        elif sweep_high_pre:
            pre_signals.append("PRE: Sweep High + подтверждение ⚡")

        # ── Нет ENTRY сигнала — возвращаем PRE если есть
        if setup_type is None:
            if pre_signals:
                return {
                    'symbol':    symbol,
                    'price':     last_close,
                    'score':     len(pre_signals),
                    'signals':   pre_signals,
                    'trade_dir': 0,
                    'context':   f"PRE | {htf_context}",
                    'strategy':  "Pre-Signal",
                    'ftr_test':  0, 'tvx': [],
                    'suggested_sl': np.nan,
                    'suggested_tp': np.nan,
                    'rr': 0,
                    'timestamp': str(d['timestamp'].iloc[-1])[:16],
                }
            return None

        if signal_dir is None:
            return None

        stop_dist    = atr_val * 1.0
        suggested_sl = last_close - stop_dist if signal_dir==1 else last_close + stop_dist
        tp_candidates = []
        if signal_dir == 1:
            if not np.isnan(vah)     and vah>last_close:     tp_candidates.append(vah)
            if not np.isnan(poc_htf) and poc_htf>last_close: tp_candidates.append(poc_htf)
            if not np.isnan(vah_htf) and vah_htf>last_close: tp_candidates.append(vah_htf)
            for z in zns:
                if z['active'] and z['dir']==-1 and z['zl']>last_close: tp_candidates.append(z['zl']); break
            SH,_ = find_swings(d)
            ab = [s['price'] for s in SH if s['price']>last_close]
            if ab: tp_candidates.append(min(ab))
        else:
            if not np.isnan(val)     and val<last_close:     tp_candidates.append(val)
            if not np.isnan(poc_htf) and poc_htf<last_close: tp_candidates.append(poc_htf)
            if not np.isnan(val_htf) and val_htf<last_close: tp_candidates.append(val_htf)
            for z in zns:
                if z['active'] and z['dir']==1 and z['zh']<last_close: tp_candidates.append(z['zh']); break
            _,SLsw = find_swings(d)
            bl = [s['price'] for s in SLsw if s['price']<last_close]
            if bl: tp_candidates.append(max(bl))

        best_tp=None; best_rr=0.0
        for tp in tp_candidates:
            tpd = abs(tp-last_close)
            if tpd < atr_val*0.7: continue
            rr_tmp = tpd/stop_dist if stop_dist>0 else 0
            if rr_tmp > best_rr: best_rr=rr_tmp; best_tp=tp
        # R/R фильтр убран
        if best_tp is None:
            best_tp = last_close + stop_dist*2 if signal_dir==1 else last_close - stop_dist*2
            best_rr = 2.0

        return {
            'symbol': symbol, 'price': last_close, 'score': len(signals),
            'signals': signals, 'trade_dir': signal_dir,
            'context': f"HTF:{htf_context}", 'strategy': setup_type,
            'ftr_test': ftr_tc, 'tvx': triggers if 'triggers' in dir() else [],
            'suggested_sl': round(suggested_sl,6),
            'suggested_tp': round(best_tp,6),
            'rr': round(best_rr,1),
            'timestamp': str(d['timestamp'].iloc[-1])[:16],
        }
    except Exception as e:
        print(f"[BG] scan_signal_bg {symbol}: {e}")
        return None



_sent_pre: dict = {}

# Статус сканирований (для отображения в сайдбаре)
_scan_status = {
    'entry_last':   None,   # время последнего ENTRY цикла
    'entry_found':  0,
    'pre_last':     None,   # время последнего 15м цикла
    'pre_found':    0,
    'ctx_found':    0,
    'entry_total':  0,      # всего ENTRY за сессию
    'pre_total':    0,
}

def _background_scan():
    """
    ПОТОК 1 — каждые 60 секунд.
    Ищет ТОЛЬКО сигналы ENTRY:
      Reversal / Continuation / Liq.Grab / Breakout Retest
    Pre-Signals и Renko — в отдельном 15-минутном потоке.
    """
    ALL_BG = list(dict.fromkeys(
        ["BTCUSDT"] + LIST_ALTS_MAIN + LIST_ALTS_2 + LIST_ALTS_3
    ))
    print(f"[ENTRY] Сканер запущен. Активов: {len(ALL_BG)}")

    while True:
        cycle_start = time.time()
        found = 0

        for sym_idx, sym in enumerate(ALL_BG):
            # Обновляем статус каждые 10 символов для UI
            if sym_idx % 10 == 0:
                _scan_status['entry_last'] = time.time()

            try:
                result = scan_signal_bg(sym, "15m")
                if result is None:
                    time.sleep(0.05)
                    continue

                # Пропускаем Pre-Signals — они в 15-минутном потоке
                if result.get('strategy') == 'Pre-Signal':
                    continue

                key = (f"{sym}_{result['trade_dir']}_"
                       f"{round(result.get('suggested_sl',0),3)}_"
                       f"{round(result.get('suggested_tp',0),3)}")
                last_sent = _sent_signals.get(key)
                if last_sent and (time.time() - last_sent) < 7200:
                    continue

                _sent_signals[key] = time.time()
                tg_send(format_signal_message(result))
                found += 1
                print(f"[ENTRY] {sym} [{result['strategy']}] score={result['score']}")
                time.sleep(0.1)
            except Exception as e:
                import traceback
                print(f"[ENTRY] Ошибка {sym}: {e}")
                print(traceback.format_exc()[:300])

        cycle_time = time.time() - cycle_start
        _scan_status['entry_last']  = time.time()
        _scan_status['entry_found'] = found
        _scan_status['entry_total'] += found
        print(f"[ENTRY] Цикл завершён за {cycle_time:.1f}с | Сигналов: {found}")
        time.sleep(max(5, 60 - cycle_time))


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT SCANNER — MTF анализ приближения к зонам (1D → 1H → 15m)
# ─────────────────────────────────────────────────────────────────────────────

def detect_ftr_absorption_signal(d, zns, atr_val):
    """
    Детектирует паттерн со скриншота:
    Свеча закрылась медвежьей (close < open) у зоны FTR
    НО дельта положительная (покупатели поглощены) = медвежий absorption.

    Это сигнал "крупняк защищает зону сверху" — классический шорт у Supply FTR.
    Зеркально: бычья свеча + отрицательная дельта у Demand FTR = лонг.
    """
    signals = []
    last    = d.iloc[-1]
    lc      = float(last['close'])
    lo_p    = float(last['open'])
    ld      = float(last['delta']) if 'delta' in last.index else 0
    lv      = float(last['volume'])

    vm30 = float(d['volume'].rolling(30).mean().iloc[-1])

    for z in zns:
        if not z['active']: continue
        dist = min(abs(lc - z['zl']), abs(lc - z['zh'])) / lc * 100
        if dist > 0.8: continue  # слишком далеко

        if z['dir'] == -1:  # Supply зона — ждём медвежий исход
            # Свеча медвежья + дельта положительная = absorption продавцом
            bear_candle   = lc < lo_p
            positive_delta = ld > 0
            if bear_candle and positive_delta:
                signals.append({
                    'type':  'Bear Absorption at Supply',
                    'emoji': '🔴💥',
                    'zone':  z,
                    'dist':  round(dist, 3),
                    'delta': round(ld, 0),
                })

        elif z['dir'] == 1:  # Demand зона — ждём бычий исход
            # Свеча бычья + дельта отрицательная = absorption покупателем
            bull_candle    = lc > lo_p
            negative_delta = ld < 0
            if bull_candle and negative_delta:
                signals.append({
                    'type':  'Bull Absorption at Demand',
                    'emoji': '🟢💥',
                    'zone':  z,
                    'dist':  round(dist, 3),
                    'delta': round(ld, 0),
                })

    return signals


def is_approaching_zone(price, zl, zh, threshold=1.0):
    """
    Проверяет ПРИБЛИЖЕНИЕ цены к зоне — только снаружи.
    Если цена уже внутри зоны — НЕ считается приближением.
    """
    dist = min(abs(price - zl), abs(price - zh)) / price * 100
    inside = zl <= price <= zh
    if inside:
        return False, 0.0
    return dist < threshold, round(dist, 3)


def is_retest(price, level, df, direction):
    """
    Проверяет ретест уровня — цена была по другую сторону и вернулась.
    VAH retest: была выше → вернулась ниже (шорт)
    VAL retest: была ниже → вернулась выше (лонг)
    """
    try:
        recent = df['close'].iloc[-10:-1]
        if direction == -1:  # VAH ретест (шорт)
            was_above = (recent > level).any()
            now_below = price < level
            return was_above and now_below
        else:  # VAL ретест (лонг)
            was_below = (recent < level).any()
            now_above = price > level
            return was_below and now_above
    except:
        return False


def is_valid_level_interaction(zone_type, price, level, df, direction=None):
    """
    Универсальная проверка: цена взаимодействует с уровнем правильно.

    VAH (шорт):
      ✅ подход снизу (price < VAH)
      ✅ ретест сверху (была выше → вернулась)
      ❌ цена уже выше и идёт дальше

    VAL (лонг):
      ✅ подход сверху (price > VAL)
      ✅ ретест снизу (была ниже → вернулась)
      ❌ цена уже ниже и идёт дальше

    FTR: без ограничений по стороне (зона работает с обеих)
    POC: требует отдельной проверки через is_strong_poc
    """
    if zone_type in ('FTR', 'Demand FTR', 'Supply FTR'):
        return True, "FTR"  # FTR проверяется через was_outside

    if zone_type == 'VAH':
        approach = price < level  # подход снизу к сопротивлению
        retest   = is_retest(price, level, df, direction=-1)
        if approach:
            return True, "Подход к VAH снизу"
        if retest:
            return True, "Ретест VAH ↓"
        return False, "Цена уже выше VAH — нет сетапа"

    if zone_type == 'VAL':
        approach = price > level  # подход сверху к поддержке
        retest   = is_retest(price, level, df, direction=1)
        if approach:
            return True, "Подход к VAL сверху"
        if retest:
            return True, "Ретест VAL ↑"
        return False, "Цена уже ниже VAL — нет сетапа"

    if zone_type == 'POC':
        # POC работает как магнит — с обеих сторон
        return True, "POC"

    return True, zone_type


# ─────────────────────────────────────────────────────────────────────────────
# DECISION POINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def has_space(price, direction, d, atr):
    """Есть ли куда идти после зоны (минимум 1.5 ATR пространства)."""
    try:
        if direction == 1:
            next_resistance = float(d['high'].rolling(50).max().iloc[-2])
            space = next_resistance - price
        else:
            next_support = float(d['low'].rolling(50).min().iloc[-2])
            space = price - next_support
        return space > atr * 1.5
    except:
        return True  # если не можем посчитать — не блокируем


def is_strong_poc(z, d):
    """
    POC сильный только если:
    1. Зона не свежая (> 10 баров)
    2. После формирования был импульс (> 1.5 ATR)
    3. Цена покидала зону
    """
    try:
        bars_since = len(d) - z['i']
        if bars_since < 10:
            return False
        after = d.iloc[z['i']:]
        if len(after) < 3:
            return False
        atr = float((d['high']-d['low']).rolling(14).mean().iloc[-1])
        impulse = abs(float(after['close'].iloc[-1]) - float(after['close'].iloc[0]))
        if impulse < atr * 1.5:
            return False
        was_outside = (
            (after['close'] > z['zh']).any() or
            (after['close'] < z['zl']).any()
        )
        return was_outside
    except:
        return False


def poc_has_ftr_overlap(poc_zl, poc_zh, ftr_zones, tolerance_pct=0.5):
    """
    Ищет совпадение POC с FTR зоной.
    Confluence FTR + POC = A+ сетап (Wyckoff + Dalton + Order Flow).
    """
    poc_mid = (poc_zl + poc_zh) / 2
    for ftr in ftr_zones:
        if not ftr.get('active', True):
            continue
        ftr_mid = (ftr['zl'] + ftr['zh']) / 2
        dist_pct = abs(ftr_mid - poc_mid) / poc_mid * 100
        if dist_pct < tolerance_pct:
            return True, ftr
        # Или перекрытие зон
        overlap = not (ftr['zh'] < poc_zl or ftr['zl'] > poc_zh)
        if overlap:
            return True, ftr
    return False, None


def calc_decision_score(z, amt_h1, amt_d1, price, d, ftr_zones_h1, atr):
    """
    Decision Score (0-10) — насколько зона является точкой принятия решения.
    Сигнал только если score >= 5.
    """
    score = 0
    reasons = []

    # 1. Тип зоны (FTR сильнее VAH/VAL)
    zt = z.get('zone_type','')
    if 'FTR' in zt:
        score += 3; reasons.append("FTR зона")
    elif zt in ('VAH','VAL'):
        score += 2; reasons.append(f"{zt} край диапазона")
    elif zt == 'POC':
        # POC только если сильный
        score += 1; reasons.append("POC")

    # 2. Confluence FTR + POC
    if zt == 'POC':
        has_conf, _ = poc_has_ftr_overlap(z['zl'], z['zh'], ftr_zones_h1)
        if has_conf:
            score += 3; reasons.append("🔥 Confluence FTR+POC")
        else:
            score -= 2  # POC без FTR = слабо

    # 3. Есть пространство для движения
    if z.get('dir', 0) != 0 and has_space(price, z['dir'], d, atr):
        score += 2; reasons.append("✅ Space OK")
    else:
        score -= 1

    # 4. Phase transition (конец тренда / накопление / распределение)
    is_trans = (
        amt_h1.get('sot') or
        amt_d1.get('sot') or
        amt_h1.get('balance_phase') in ["ACCUMULATION","DISTRIBUTION"] or
        amt_d1.get('balance_phase') in ["ACCUMULATION","DISTRIBUTION"]
    )
    if is_trans:
        score += 2; reasons.append("🔄 Phase Transition")

    # 5. Зрелость зоны
    zone_age = len(d) - z.get('i', 0) if z.get('i') is not None else 999
    if zone_age > 20:
        score += 1; reasons.append("Зрелая зона")

    return score, reasons


def scan_context(symbol):
    """
    MTF Context Scanner (1D → 1H → 15m).
    Ищет ситуации где цена приближается к зоне — без триггера входа.
    Отправляется как ранний сигнал 🟡 CONTEXT.
    """
    try:
        # ── Загрузка 3 таймфреймов
        d1  = fetch_data_bg(symbol, "1D")
        h1  = fetch_data_bg(symbol, "1h")
        m15 = fetch_data_bg(symbol, "15m")
        if d1.empty or h1.empty or m15.empty: return None
        if len(d1) < 20 or len(h1) < 30 or len(m15) < 30: return None

        d1  = apply_order_flow(d1)
        h1  = apply_order_flow(h1)
        m15 = apply_order_flow(m15)

        price = float(m15['close'].iloc[-1])

        # ── Структурный анализ через машину состояний (единая истина)
        (in_bal_d1, t_dir_d1, kh_d1, kl_d1, vah_d1_ctx, val_d1_ctx, poc_d1,
         _, _, _, _) = get_struct_levels(d1, tf="1D", sw=3)
        (in_bal_h1, t_dir_h1, kh_h1, kl_h1, vah_h1_ctx, val_h1_ctx, poc_h1,
         _, _, _, _) = get_struct_levels(h1, tf="1h", sw=3)
        (in_bal_m15, t_dir_m15, kh_m15, kl_m15, vah_m15, val_m15, poc_m15,
         _, _, _, _) = get_struct_levels(m15, tf="15m", sw=3)

        phase_d1  = 'BALANCE' if in_bal_d1  else ('TREND_UP' if t_dir_d1  == 1 else 'TREND_DOWN')
        dir_d1    = t_dir_d1
        phase_h1  = 'BALANCE' if in_bal_h1  else ('TREND_UP' if t_dir_h1  == 1 else 'TREND_DOWN')
        dir_h1    = t_dir_h1
        phase_m15 = 'BALANCE' if in_bal_m15 else ('TREND_UP' if t_dir_m15 == 1 else 'TREND_DOWN')
        dir_m15   = t_dir_m15

        # AMT для баланс-фазы (только для подтипа баланса — накопление/распределение)
        amt_d1  = analyze_auction_context(d1,  tf="1D")
        amt_h1  = analyze_auction_context(h1,  tf="1h")
        amt_m15 = analyze_auction_context(m15, tf="15m")

        # ── Зоны FTR на D1 и H1
        _p     = get_ftr_params(symbol)
        zns_d1 = calc_ftr_zones(d1, **_p)
        zns_h1 = calc_ftr_zones(h1, **_p)

        def phase_label(phase, direction):
            labels = {'TREND_UP': 'TREND ↑', 'TREND_DOWN': 'TREND ↓', 'BALANCE': 'BALANCE'}
            return labels.get(phase, phase)

        # FTR absorption сигналы на 1H и 15m
        ftr_abs_h1  = detect_ftr_absorption_signal(h1,  zns_h1, float((h1['high']-h1['low']).rolling(14).mean().iloc[-1]))
        ftr_abs_m15 = detect_ftr_absorption_signal(m15, zns_h1, float((m15['high']-m15['low']).rolling(14).mean().iloc[-1]))

        # ── Собираем приближения с фильтрацией по качеству

        # ── Собираем приближения с фильтрацией по качеству
        approaching = []

        def add_zone(tf, zone_type, zdir, zl, zh, dist, strength, z_raw=None):
            """Добавляет зону только если она является Decision Point."""
            # Размер зоны — слишком широкие отсекаем
            zone_size = (zh - zl) / price * 100
            if zone_size > 2.0:
                return

            # ── Фильтр по SMC структуре: зона должна быть внутри Key High/Low диапазона
            # Demand FTR должна быть выше KEY LOW (иначе — за пределами структуры)
            # Supply FTR должна быть ниже KEY HIGH
            kh_ref = kh_d1 if tf == '1D' else kh_h1
            kl_ref = kl_d1 if tf == '1D' else kl_h1
            if zdir == 1 and not np.isnan(kl_ref):
                if zh < kl_ref:
                    return  # Demand зона ниже KEY LOW — вне структуры
            if zdir == -1 and not np.isnan(kh_ref):
                if zl > kh_ref:
                    return  # Supply зона выше KEY HIGH — вне структуры

            # Фильтр по SMC фазе (используем фазы из get_struct_levels)
            if tf == '1H':
                if phase_h1 in ('TREND_UP','TREND_DOWN') and zdir != 0 and zdir != dir_h1:
                    return  # не по тренду
            if tf == '1D':
                if phase_d1 in ('TREND_UP','TREND_DOWN') and zdir != 0 and zdir != dir_d1:
                    return

            # Только края диапазона (не середина) — для VAH/VAL/POC
            if zone_type in ('VAH','VAL','POC'):
                rs_vah = vah_d1_ctx if tf == '1D' else vah_h1_ctx
                rs_val = val_d1_ctx if tf == '1D' else val_h1_ctx
                range_size = (rs_vah - rs_val) if (not np.isnan(rs_vah) and not np.isnan(rs_val)) else 0
                if range_size > 0:
                    pos = (price - rs_val) / range_size
                    if 0.3 < pos < 0.7:
                        return  # середина диапазона — нет движения

            # ── Проверка правильной стороны уровня (VAH/VAL/POC)
            if zone_type in ('VAH', 'VAL', 'POC'):
                # Определяем уровень для проверки
                level_mid = (zl + zh) / 2
                valid_interact, interact_reason = is_valid_level_interaction(
                    zone_type, price, level_mid, m15, zdir
                )
                if not valid_interact:
                    return  # неправильная сторона — пропускаем
                # Добавляем reason к strength
                strength = strength + f" [{interact_reason}]"

            # POC — только сильный
            if zone_type == 'POC':
                if z_raw is None or not is_strong_poc(z_raw, m15):
                    return

            # Space фильтр — есть куда идти
            atr_m15 = float((m15['high']-m15['low']).rolling(14).mean().iloc[-1])
            if zdir != 0 and not has_space(price, zdir, m15, atr_m15):
                return

            # Decision Score
            z_for_score = z_raw or {'zone_type': zone_type, 'dir': zdir, 'zl': zl, 'zh': zh, 'i': 0}
            dscore, dreasons = calc_decision_score(
                {**z_for_score, 'zone_type': zone_type},
                amt_h1, amt_d1, price, m15, zns_h1, atr_m15
            )
            if dscore < 2:
                return  # недостаточно сильная точка (снижен с 4 до 2)

            approaching.append({
                'tf': tf, 'zone_type': zone_type, 'dir': zdir,
                'zl': round(zl, 6), 'zh': round(zh, 6),
                'dist': dist, 'strength': strength,
                'dscore': dscore, 'reasons': dreasons,
            })

        # FIX 1: сниженные пороги — D1=0.6%, H1=0.4%
        # D1 FTR зоны
        for z in zns_d1:
            if not z['active']: continue
            near, dist = is_approaching_zone(price, z['zl'], z['zh'], threshold=0.6)
            if near:
                add_zone('1D',
                    f"{'Demand' if z['dir']==1 else 'Supply'} FTR",
                    z['dir'], z['zl'], z['zh'], dist,
                    '🔴🔴🔴' if dist < 0.3 else '🔴🔴')

        # H1 FTR зоны
        for z in zns_h1:
            if not z['active']: continue
            near, dist = is_approaching_zone(price, z['zl'], z['zh'], threshold=0.4)
            if near:
                add_zone('1H',
                    f"{'Demand' if z['dir']==1 else 'Supply'} FTR",
                    z['dir'], z['zl'], z['zh'], dist,
                    '🟡🟡🟡' if dist < 0.2 else '🟡🟡')

        # VAH/VAL/POC через get_levels — единственный источник правды
        vah_d1_ctx, val_d1_ctx, poc_d1_ctx, mode_d1_ctx, bh_d1, bl_d1 = get_levels(d1, "1D")
        vah_h1_ctx, val_h1_ctx, poc_h1_ctx, mode_h1_ctx, bh_h1, bl_h1 = get_levels(h1, "1h")

        for tf_label, lvl_vah, lvl_val, lvl_poc, thr in [
            ('1D', vah_d1_ctx, val_d1_ctx, poc_d1_ctx, 0.5),
            ('1H', vah_h1_ctx, val_h1_ctx, poc_h1_ctx, 0.3),
        ]:
            levels = []
            if not np.isnan(lvl_vah): levels.append(('VAH', lvl_vah, -1))
            if not np.isnan(lvl_val): levels.append(('VAL', lvl_val,  1))
            if tf_label == '1D' and not np.isnan(lvl_poc):
                levels.append(('POC', lvl_poc, 0))
            for level_name, level_val, ldir in levels:
                near, dist = is_approaching_zone(price, level_val*0.999, level_val*1.001, threshold=thr)
                if near:
                    add_zone(tf_label, level_name, ldir,
                             level_val*0.999, level_val*1.001, dist,
                             '🔵🔵' if dist < 0.15 else '🔵', z_raw=None)

        if not approaching: return None

        # FIX 2: берём только 1 лучшую зону — ближайшую D1, иначе H1
        approaching.sort(key=lambda x: (0 if x['tf']=='1D' else 1, x['dist']))
        # Оставляем максимум 2: лучшую D1 и лучшую H1
        best = []
        seen_tf = set()
        for z in approaching:
            if z['tf'] not in seen_tf:
                best.append(z)
                seen_tf.add(z['tf'])
        approaching = best[:2]

        # Определяем общее направление (совпадение трендов = сильнее)
        if phase_d1 == "TREND" and phase_h1 == "TREND" and dir_d1 == dir_h1:
            alignment_str = "✅ Тренды совпадают"
        elif phase_d1 == "BALANCE" and phase_h1 == "BALANCE":
            alignment_str = "⚖️ Оба в балансе"
        else:
            alignment_str = "⚠️ Разные фазы"

        # Режимы рынка для контекста
        mode_str_d1 = f" [{mode_d1_ctx}]" if mode_d1_ctx else ""
        mode_str_h1 = f" [{mode_h1_ctx}]" if mode_h1_ctx else ""

        # Breakout retest зоны в Context (дополнительно к FTR зонам)
        br_zones = []
        for tf_lbl, lvl_vah, lvl_val, lvl_poc, mode_ctx in [
            ('1D', vah_d1_ctx, val_d1_ctx, poc_d1_ctx, mode_d1_ctx),
            ('1H', vah_h1_ctx, val_h1_ctx, poc_h1_ctx, mode_h1_ctx),
        ]:
            if mode_ctx != "BREAKOUT": continue
            thr_br = 0.6 if tf_lbl == '1D' else 0.4
            for lv_name, lv_val, lv_dir in [
                ('VAH', lvl_vah, -1), ('VAL', lvl_val, 1), ('POC', lvl_poc, 0)
            ]:
                if np.isnan(lv_val): continue
                dist_br = abs(price - lv_val) / price * 100
                if dist_br < thr_br:
                    # Цена снаружи уровня (breakout) = ретест
                    outside_vah = price > lv_val and lv_name == 'VAH'
                    outside_val = price < lv_val and lv_name == 'VAL'
                    near_poc    = lv_name == 'POC'
                    if outside_vah or outside_val or near_poc:
                        br_zones.append({
                            'tf': tf_lbl, 'zone_type': f"🔄 Retest {lv_name}",
                            'dir': lv_dir, 'zl': round(lv_val*0.999, 6),
                            'zh': round(lv_val*1.001, 6), 'dist': round(dist_br, 3),
                            'strength': '🟣🟣🟣' if dist_br < 0.2 else '🟣🟣',
                            'dscore': 7, 'reasons': [f"BREAKOUT Retest {lv_name}"]
                        })

        all_approaching = approaching[:2] + br_zones[:2]

        return {
            'symbol':       symbol,
            'price':        price,
            'phase_d1':     phase_label(phase_d1, dir_d1) + mode_str_d1,
            'phase_h1':     phase_label(phase_h1, dir_h1) + mode_str_h1,
            'phase_m15':    phase_label(phase_m15, dir_m15),
            'sot_d1':       amt_d1.get('sot', False),
            'sot_h1':       amt_h1.get('sot', False),
            'bal_phase_h1': amt_h1.get('balance_phase'),
            'approaching':  all_approaching,
            'ftr_abs':      ftr_abs_h1 + ftr_abs_m15,
            'alignment':    alignment_str,
            'hint_d1':      amt_d1.get('strategy_hint',''),
            'timestamp':    str(m15['timestamp'].iloc[-1])[:16],
        }
    except Exception as e:
        print(f"[CTX] Ошибка {symbol}: {e}")
        return None


def format_context_message(r):
    """Форматирует CONTEXT сигнал для Telegram."""
    zones_txt = ""
    for z in r['approaching']:
        arrow   = "🟢" if z['dir']==1 else ("🔴" if z['dir']==-1 else "🔵")
        dscore  = z.get('dscore', 0)
        stars   = "⭐" * min(dscore // 2, 5)
        reasons = " | ".join(z.get('reasons', []))
        zones_txt += (
            f"\n{z['strength']} <b>{z['tf']}</b> {arrow} {z['zone_type']} {stars}"
            f"\n   Зона: <code>{z['zl']:.4f} \u2013 {z['zh']:.4f}</code>"
            f"\n   Дистанция: <code>{z['dist']:.2f}%</code>"
            f"\n   <i>{reasons}</i>\n"
        )

    sot_warn = ""
    if r.get('sot_d1'): sot_warn += "\n\u26a0\ufe0f SOT на D1 \u2014 тренд слабеет"
    if r.get('sot_h1'): sot_warn += "\n\u26a0\ufe0f SOT на H1 \u2014 тренд слабеет"

    bp_h1  = r.get('bal_phase_h1','')
    bp_str = f"\n\U0001f4e6 H1 баланс: {bp_h1}" if bp_h1 and bp_h1 != "NEUTRAL" else ""

    # Блок FTR Absorption (паттерн: свеча против зоны + дельта за зону)
    abs_txt = ""
    for a in r.get('ftr_abs', []):
        z = a['zone']
        abs_txt += (
            f"\n{a['emoji']} <b>{a['type']}</b>"
            f"\n   Зона FTR: <code>{z['zl']:.4f} \u2013 {z['zh']:.4f}</code>"
            f"  |  Дельта: <code>{a['delta']:+.0f}</code>\n"
        )
    abs_block = (
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\U0001f525 <b>FTR Absorption (войди и проверь!):</b>{abs_txt}"
    ) if abs_txt else ""

    header_emoji = "\U0001f534" if r.get('ftr_abs') else "\U0001f7e1"
    zones_block  = (
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\U0001f4cd <b>Приближение к зонам:</b>{zones_txt}"
    ) if zones_txt.strip() else ""

    msg = (
        f"{header_emoji} <b>CONTEXT: {r['symbol']}</b>\n"
        f"Цена: <code>{r['price']:.5f}</code>\n"
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\U0001f4ca <b>Фазы рынка:</b>\n"
        f"  1D \u2192 {r['phase_d1']}\n"
        f"  1H \u2192 {r['phase_h1']}\n"
        f"  15m \u2192 {r['phase_m15']}\n"
        f"  {r['alignment']}"
        f"{bp_str}"
        f"{sot_warn}\n"
        f"{abs_block}"
        f"{zones_block}"
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\U0001f4a1 <i>{r.get('hint_d1','')}</i>\n"
        f"<i>\u23f1 {r['timestamp']}</i>"
    )
    return msg


# Хранилище отправленных CONTEXT сигналов (не повторять 4 часа)
_sent_context: dict = {}


def _context_scan_loop():
    """
    ПОТОК 2 — каждые 15 минут.
    Ищет три группы ситуаций:

    A) PRE-SIGNALS (свечные графики):
       Цена подходит к важным уровням без подтверждения.
       Кулдаун: 1 час.

    B) RENKO ДИВЕРГЕНЦИИ:
       Расхождение CVD и цены на Renko.
       Кулдаун: 2 часа.

    C) CONTEXT SCANNER (MTF 1D→1H→15m):
       Фазы рынка + приближение к зонам + FTR Absorption.
       Кулдаун: 6 часов.
    """
    ALL_BG = list(dict.fromkeys(
        ["BTCUSDT"] + LIST_ALTS_MAIN + LIST_ALTS_2 + LIST_ALTS_3
    ))
    print(f"[15m] Сканер запущен. Активов: {len(ALL_BG)}")

    while True:
        cycle_start = time.time()
        pre_found   = 0
        ctx_found   = 0

        for sym in ALL_BG:

            # ── A) Pre-Signals (свечные)
            try:
                result = scan_signal_bg(sym, "15m")
                if result is not None and result.get('strategy') == 'Pre-Signal':
                    pre_key = f"pre_{sym}_{result.get('context','')}"
                    last_pre = _sent_pre.get(pre_key)
                    if not last_pre or (time.time() - last_pre) > 3600:
                        _sent_pre[pre_key] = time.time()
                        tg_send(format_signal_message(result))
                        pre_found += 1
                        print(f"[PRE] {sym}: {result['signals'][0] if result['signals'] else ''}")
                        time.sleep(0.1)
            except Exception as e:
                print(f"[PRE] Ошибка {sym}: {e}")

            # ── A2) Дивергенции на свечных графиках (каждые 15м)
            try:
                d_div = fetch_data_bg(sym, "15m")
                if d_div is not None and len(d_div) >= 50:
                    d_div = apply_order_flow(d_div)
                    candle_divs = calc_candle_divergences(d_div, lookback=5, min_dist=8)
                    if candle_divs:
                        last_ts = d_div['timestamp'].iloc[-1]
                        n_d = len(d_div)
                        fresh = [dv for dv in candle_divs
                                 if n_d - np.searchsorted(d_div['timestamp'].values, dv['x1']) <= 5]
                        if fresh:
                            div_key = f"candle_div_{sym}_{fresh[-1]['type']}"
                            last_d  = _sent_pre.get(div_key)
                            if not last_d or (time.time() - last_d) > 7200:
                                _sent_pre[div_key] = time.time()
                                dv     = fresh[-1]
                                lc_d   = float(d_div['close'].iloc[-1])
                                d_emoji = "🟢" if dv['type']=='bull' else "🔴"
                                d_label = "Бычья" if dv['type']=='bull' else "Медвежья"
                                msg = (
                                    f"📊 <b>PRE: Дивергенция CVD (свечи)</b>\n"
                                    f"<b>{sym}</b> | 15m\n"
                                    f"{d_emoji} {d_label} дивергенция CVD vs Price\n"
                                    f"Цена: <code>{lc_d:.5f}</code>"
                                )
                                tg_send(msg)
                                pre_found += 1
                                print(f"[DIV] {sym}: {d_label} дивергенция на свечах")
                                time.sleep(0.1)
            except Exception as e:
                print(f"[DIV] Ошибка {sym}: {e}")

            # ── C) Context Scanner (MTF)
            try:
                ctx = scan_context(sym)
                if ctx is not None:
                    has_zones = bool(ctx.get('approaching'))
                    has_abs   = bool(ctx.get('ftr_abs'))
                    if has_zones or has_abs:
                        z0       = ctx['approaching'][0] if has_zones else {'tf':'abs','zl':0,'zh':0}
                        abs_key  = ctx['ftr_abs'][0]['type'] if has_abs else "none"
                        zone_key = f"{z0['tf']}_{round(z0['zl'],4)}" if has_zones else "no_zone"
                        key      = f"ctx_{sym}_{zone_key}_{abs_key}"
                        last_ctx = _sent_context.get(key)
                        if not last_ctx or (time.time() - last_ctx) > 21600:
                            _sent_context[key] = time.time()
                            tg_send(format_context_message(ctx))
                            ctx_found += 1
                            print(f"[CTX] {sym}: {zone_key}")
                            time.sleep(0.2)
            except Exception as e:
                print(f"[CTX] Ошибка {sym}: {e}")

        cycle_time = time.time() - cycle_start
        _scan_status['pre_last']    = time.time()
        _scan_status['pre_found']   = pre_found
        _scan_status['ctx_found']   = ctx_found
        _scan_status['pre_total']  += pre_found
        print(f"[15m] Цикл за {cycle_time:.1f}с | PRE:{pre_found} CTX:{ctx_found}")
        time.sleep(max(30, 900 - cycle_time))


def _is_thread_alive(name):
    """Проверяет жив ли поток по имени."""
    for t in threading.enumerate():
        if t.name == name:
            return True
    return False


def start_background_scanner():
    """Оставлено для совместимости — не используется (ручной запуск)."""
    pass


def _run_entry_scan_once(tf="15m"):
    """
    Сканер Верных Сетапов — ручной запуск.
    Использует scan_valid_setups: строго 1 ТФ, чистая SMC структура.
    Паттерны: Breakout Retest, Continuation FTR, Liq.Grab, Balance Reversal.
    """
    ALL_BG = list(dict.fromkeys(
        ["BTCUSDT"] + LIST_ALTS_MAIN + LIST_ALTS_2 + LIST_ALTS_3
    ))
    found = 0
    _scan_status['entry_last']  = time.time()
    _scan_status['entry_found'] = 0

    for sym in ALL_BG:
        try:
            result = scan_valid_setups(sym, tf)
            if result is None:
                continue
            tg_route_and_send(result)
            found += 1
            print(f"[SETUP] {sym} {tf}: {result['setup']} {result['dir']}")
        except Exception as e:
            print(f"[SETUP] Ошибка {sym}: {e}")

    _scan_status['entry_found']  = found
    _scan_status['entry_total'] += found
    _scan_status['entry_last']   = time.time()
    print(f"[SETUP] Завершено. Найдено сетапов: {found}")


def _run_pre_scan_once(tf="15m"):
    """
    Радар Пре-Сигналов — ручной запуск.
    Использует scan_radar: тесты VOL уровней, дивергенции CVD, свипы структуры.
    """
    ALL_BG = list(dict.fromkeys(
        ["BTCUSDT"] + LIST_ALTS_MAIN + LIST_ALTS_2 + LIST_ALTS_3
    ))
    total = 0
    _scan_status['pre_last']  = time.time()
    _scan_status['pre_found'] = 0

    for sym in ALL_BG:
        try:
            signals = scan_radar(sym, tf)
            for sig in signals:
                tg_route_and_send(sig)
                total += 1
                print(f"[RADAR] {sym} {tf}: {sig['type']} — {sig['desc'][:50]}")
        except Exception as e:
            print(f"[RADAR] Ошибка {sym}: {e}")

    _scan_status['pre_found']  = total
    _scan_status['pre_total'] += total
    _scan_status['pre_last']   = time.time()
    print(f"[RADAR] Завершено. Найдено событий: {total}")


# ─────────────────────────────────────────────────────────────────────────────
# МОНИТОРИНГ БЕЗУБЫТКА — Background поток для Balance Reversal сигналов
# Когда сделка достигает +1.5R → отправляет TG алерт "переводи SL в безубыток"
# ─────────────────────────────────────────────────────────────────────────────

def _run_breakeven_monitor():
    """
    Фоновый поток: раз в 60 сек проверяет активные Balance Reversal сигналы.
    При достижении +1.5R отправляет Telegram алерт.
    Удаляет сигналы достигшие TP или SL.
    """
    print("[BE_MONITOR] Поток мониторинга безубытка запущен")
    while True:
        try:
            time.sleep(60)
            if not _active_bzone_signals:
                continue
            expired = []
            for key, sig in list(_active_bzone_signals.items()):
                try:
                    cur = _get_last_price(sig['symbol'])
                    if cur <= 0:
                        continue
                    entry = sig['entry']
                    sl    = sig['sl']
                    tp    = sig['tp']
                    be    = sig['be_level']
                    dirn  = sig['direction']
                    sym   = sig['symbol']
                    tf    = sig['tf']

                    hit_be = (cur >= be) if dirn == 'LONG' else (cur <= be)
                    hit_tp = (cur >= tp) if dirn == 'LONG' else (cur <= tp)
                    hit_sl = (cur <= sl) if dirn == 'LONG' else (cur >= sl)

                    if hit_tp or hit_sl:
                        result = "✅ TP достигнут" if hit_tp else "❌ SL выбит"
                        print(f"[BE_MONITOR] {sym} {tf}: {result}, удаляем из активных")
                        expired.append(key)
                        continue

                    if hit_be and not sig['be_sent']:
                        r_dist = abs(entry - sl)
                        icon   = "🟢" if dirn == 'LONG' else "🔴"
                        msg = (
                            f"🔐 <b>БЕЗУБЫТОК | {sym} | {tf}</b>\n\n"
                            f"{icon} {dirn} достиг <b>+1.5R</b> — переводи SL!\n\n"
                            f"Цена входа: <code>{entry:.5f}</code>\n"
                            f"SL → безубыток: <code>{entry:.5f}</code>\n"
                            f"TP (POC): <code>{tp:.5f}</code>\n"
                            f"Текущая цена: <code>{cur:.5f}</code>"
                        )
                        tg_send(msg)
                        _active_bzone_signals[key]['be_sent'] = True
                        print(f"[BE_MONITOR] {sym} {tf}: безубыток отправлен")

                    # Удаляем сигналы старше 7 дней
                    if time.time() - sig['ts'] > 604800:
                        expired.append(key)

                except Exception as e:
                    print(f"[BE_MONITOR] Ошибка {key}: {e}")

            for k in expired:
                _active_bzone_signals.pop(k, None)

        except Exception as e:
            print(f"[BE_MONITOR] Поток: {e}")


def _start_breakeven_monitor_if_needed():
    """Запускает поток мониторинга безубытка если ещё не запущен."""
    if not _is_thread_alive('bzone_be_monitor'):
        t = threading.Thread(target=_run_breakeven_monitor,
                             name='bzone_be_monitor', daemon=True)
        t.start()
        print("[BE_MONITOR] Поток запущен")


# Автозапуск убран — сканирование только вручную через кнопки в сайдбаре


# ─────────────────────────────────────────────────────────────────────────────
# 3а. ЗАГРУЗКА 1М СВЕЧЕЙ И DELTA CANDLES
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_1m_data(symbol, src):
    if src != "crypto":
        return pd.DataFrame()
    all_data, end_time = [], None
    for _ in range(3):
        try:
            params = {"category": "linear", "symbol": symbol, "interval": "1", "limit": 1000}
            if end_time:
                params["end"] = end_time
            res = st.session_state.session.get_kline(**params).get('result', {}).get('list', [])
            if not res: break
            all_data.extend(res)
            end_time = int(res[-1][0]) - 1
        except: break
    if not all_data:
        return pd.DataFrame()
    d = pd.DataFrame(all_data, columns=['ts','o','h','l','c','v','t'])
    d['ts'] = pd.to_datetime(pd.to_numeric(d['ts']), unit='ms')
    d = d.sort_values('ts').reset_index(drop=True)
    for col in ['o','h','l','c','v','t']: d[col] = d[col].astype(float)
    d.columns = ['timestamp','open','high','low','close','volume','turnover']
    return d


def build_delta_candles(df_main, df_1m, gimelfarb_fn):
    if df_1m.empty:
        return pd.DataFrame()
    d1 = df_1m.copy()
    d1['delta'] = gimelfarb_fn(d1)
    records = []
    ts_arr = df_main['timestamp'].values
    for i in range(len(df_main)):
        t0 = ts_arr[i]
        t1 = ts_arr[i+1] if i+1 < len(df_main) else t0 + pd.Timedelta(minutes=1)
        mask = (d1['timestamp'] >= t0) & (d1['timestamp'] < t1)
        bars = d1[mask]
        if len(bars) == 0:
            records.append({'timestamp': t0, 'o': 0.0, 'h': 0.0, 'l': 0.0, 'c': 0.0})
            continue
        cum = np.cumsum(bars['delta'].values)
        records.append({'timestamp': t0, 'o': float(cum[0]),
                        'h': float(cum.max()), 'l': float(cum.min()),
                        'c': float(cum[-1])})
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 3. ИНДИКАТОРЫ (Order Flow)
# ─────────────────────────────────────────────────────────────────────────────
def calc_gimelfarb_delta(df):
    """
    Точная формула CVD по Вадиму Гимельфарбу (S&C Magazine, October 2003).
    Разделяет давление быков и медведей через 6 условий с учётом close[1].
    Принципиально точнее простой формулы (close-open)/(high-low)*volume.
    """
    o  = df['open'].values
    h  = df['high'].values
    l  = df['low'].values
    c  = df['close'].values
    v  = df['volume'].values
    c1 = np.roll(c, 1)
    c1[0] = o[0]

    n    = len(df)
    bull = np.zeros(n)
    bear = np.zeros(n)

    for i in range(n):
        hi, lo, op, cl, cl1 = h[i], l[i], o[i], c[i], c1[i]

        # BullPower
        if cl < op:
            bull[i] = max(hi - cl1, cl - lo) if cl1 < op else max(hi - op, cl - lo)
        elif cl > op:
            bull[i] = hi - lo if cl1 > op else max(op - cl1, hi - lo)
        else:
            if hi - cl > cl - lo:
                bull[i] = max(hi - cl1, cl - lo) if cl1 < op else hi - op
            elif hi - cl < cl - lo:
                bull[i] = hi - lo if cl1 > op else max(op - cl1, hi - lo)
            else:
                if   cl1 > op: bull[i] = max(hi - op,  cl - lo)
                elif cl1 < op: bull[i] = max(op - cl1, hi - lo)
                else:          bull[i] = hi - lo

        # BearPower
        if cl < op:
            bear[i] = max(cl1 - op, hi - lo) if cl1 > op else hi - lo
        elif cl > op:
            bear[i] = max(cl1 - lo, hi - cl) if cl1 > op else max(op - lo, hi - cl)
        else:
            if hi - cl > cl - lo:
                bear[i] = max(cl1 - op, hi - lo) if cl1 > op else hi - lo
            elif hi - cl < cl - lo:
                bear[i] = max(cl1 - lo, hi - cl) if cl1 > op else op - lo
            else:
                if   cl1 > op: bear[i] = max(cl1 - op, hi - lo)
                elif cl1 < op: bear[i] = max(op - lo,  hi - cl)
                else:          bear[i] = hi - lo

    total    = bull + bear
    safe     = np.where(total > 0, total, 1)
    bull_vol = (bull / safe) * v
    bear_vol = (bear / safe) * v
    return bull_vol - bear_vol


def _zigzag_traces(pivots: list, row: int = 1, col: int = 1) -> list:
    """
    Строит Plotly-трассы для ZigZag по списку пивотов из calc_zigzag().

    Визуальная градация пивотов (методология Дяди Миши):
    ┌─────────────────────────────────────────────────────────┐
    │  ТВЁРДЫЙ (hard) — ZigZag + Williams Fractal совпали    │
    │    Хай  → треугольник вниз, фиолетовый  #9C27B0, 14px  │
    │    Лой  → треугольник вверх, голубой    #00ACC1, 14px  │
    │                                                         │
    │  МЯГКИЙ (soft) — только ZigZag                         │
    │    Хай  → круг, серый  #757575, 7px                    │
    │    Лой  → круг, серый  #757575, 7px                    │
    │                                                         │
    │  НЕЗАФИКСИРОВАННЫЙ — последний пивот (пунктир)         │
    │    Серый, открытый кружок, 8px                         │
    └─────────────────────────────────────────────────────────┘
    """
    if not pivots:
        return []

    traces = []
    conf   = [p for p in pivots if p.get('confirmed')]
    unconf = [p for p in pivots if not p.get('confirmed')]

    if len(conf) >= 2:
        xs            = [pd.Timestamp(p['timestamp']) for p in conf]
        ys            = [p['price'] for p in conf]
        marker_colors = []
        marker_sizes  = []
        marker_syms   = []
        marker_lines  = []
        hover_texts   = []

        for p in conf:
            is_hard = p.get('quality') == 'hard'
            is_high = p['type'] == 'H'
            if is_hard:
                color = '#9C27B0' if is_high else '#00ACC1'   # яркий фиолет / голубой
                size  = 14
                sym   = 'triangle-down' if is_high else 'triangle-up'
                lw    = 2
                lc    = 'white'
            else:
                color = '#616161'     # тёмно-серый для мягких
                size  = 7
                sym   = 'circle'
                lw    = 1
                lc    = '#9E9E9E'
            marker_colors.append(color)
            marker_sizes.append(size)
            marker_syms.append(sym)
            marker_lines.append({'width': lw, 'color': lc})
            q = '★ ТВЁРДЫЙ' if is_hard else '○ мягкий'
            marker_type = 'Хай' if is_high else 'Лой'
            hover_texts.append(f"{q} {marker_type}: {p['price']:.5f}")

        # Единая линия через все confirmed пивоты
        traces.append(go.Scatter(
            x=xs, y=ys,
            mode='lines+markers',
            line=dict(color='rgba(179,157,219,0.6)', width=1.5),
            marker=dict(
                color=marker_colors,
                size=marker_sizes,
                symbol=marker_syms,
                line=dict(
                    width=[ml['width'] for ml in marker_lines],
                    color=[ml['color'] for ml in marker_lines],
                ),
            ),
            text=hover_texts,
            hovertemplate='%{text}<extra></extra>',
            showlegend=False,
        ))

    # Незафиксированный сегмент: последний confirmed → unconfirmed (пунктир)
    if unconf and conf:
        last_c = conf[-1]
        u      = unconf[-1]
        traces.append(go.Scatter(
            x=[pd.Timestamp(last_c['timestamp']), pd.Timestamp(u['timestamp'])],
            y=[last_c['price'], u['price']],
            mode='lines+markers',
            line=dict(color='rgba(120,144,156,0.7)', width=1.2, dash='dot'),
            marker=dict(
                color=['rgba(0,0,0,0)', 'rgba(120,144,156,0.8)'],
                size=[0, 8],
                symbol=['circle', 'circle-open'],
                line=dict(width=2, color='rgba(120,144,156,0.8)'),
            ),
            text=['', '? незафиксированный'],
            hovertemplate='%{text}: %{y:.5f}<extra></extra>',
            showlegend=False,
        ))

    return traces

# ─────────────────────────────────────────────────────────────────────────────
# 4. FTR ZONES  (AlgoPoint: ATR-импульс + база до + инвалидация)
# ─────────────────────────────────────────────────────────────────────────────


def ftr_shapes(df, zones, max_show=8):
    """
    Рисует АКТИВНЫЕ FTR зоны + последние 4 деактивированных (ghost).
    Активные — полная непрозрачность.
    Ghost (отработавшие) — показываем для истории, рисуются с пониженной opacity.
    """
    out, x1 = [], df['timestamp'].iloc[-1]

    active_zones   = [z for z in zones if z.get('active', False)]
    inactive_zones = [z for z in zones if not z.get('active', False)]

    show_active = active_zones[-max_show:] if len(active_zones) > max_show else active_zones
    show_ghost  = inactive_zones[-4:]  # последние 4 деактивированных

    for z in show_active:
        idx = min(z['i'], len(df) - 1)
        x0  = df['timestamp'].iloc[idx]
        out.append({'dir': z['dir'], 'x0': x0, 'x1': x1,
                    'y0': z['zl'], 'y1': z['zh'], 'active': True})

    for z in show_ghost:
        idx = min(z['i'], len(df) - 1)
        x0  = df['timestamp'].iloc[idx]
        # x1 для ghost = момент деактивации (последний бар у нас нет, используем x1)
        out.append({'dir': z['dir'], 'x0': x0, 'x1': x1,
                    'y0': z['zl'], 'y1': z['zh'], 'active': False})

    return out

# ─────────────────────────────────────────────────────────────────────────────
# 5. MARKET DECISION ENGINE  (PRO V14 — быстрая numpy версия)
# ─────────────────────────────────────────────────────────────────────────────




# ──────────────────────────────────────────────────────────────────────────────
# ЛОГИКА БАЛАНСОВ — точная копия Pine Script индикатора
# Принцип: Value Area → баланс (цена внутри), вышла → конец баланса → POC
# ──────────────────────────────────────────────────────────────────────────────












# ──────────────────────────────────────────────────────────────────────────────
# СТРУКТУРНЫЙ АНАЛИЗ РЫНКА
# BOS по телу свечи, Key High/Low, CHoCH, Balance Detection
# ──────────────────────────────────────────────────────────────────────────────


def get_true_smc_structure(df):
    """Williams Fractal-based Key High/Key Low (все ТФ)."""
    n = len(df)
    if n < 9:
        return np.nan, np.nan, 0, []
    return _smc_structure_fractal(df)


def get_market_phase(structure, df_len=0):
    """TREND_UP / TREND_DOWN / BALANCE из структуры."""
    if len(structure) < 4:
        return 'BALANCE', 0, None, None
    recent = structure[-12:] if len(structure) >= 12 else structure
    labels = [s[2] for s in recent]
    last4  = labels[-4:] if len(labels) >= 4 else labels
    if all(l in ('HH','HL') for l in last4):
        return 'TREND_UP', 1, None, None
    if all(l in ('LL','LH') for l in last4):
        return 'TREND_DOWN', -1, None, None
    range_high = range_low = None
    for s in reversed(structure):
        if s[2] in ('LH','HH') and range_high is None:
            range_high = s
        if s[2] in ('HL','LL') and range_low is None:
            range_low = s
        if range_high and range_low:
            break
    return 'BALANCE', 0, range_high, range_low


def detect_balance_from_structure(structure, key_highs, key_lows):
    phase, _, _, _ = get_market_phase(structure, 0)
    return phase == 'BALANCE'


def get_trend_from_structure(structure):
    _, t_dir, _, _ = get_market_phase(structure, 0)
    return t_dir


def is_deviation(df, level, direction, lookback=5):
    """Ложный пробой: тень за уровень, тело вернулось."""
    try:
        last = df.iloc[-1]
        body_top = max(float(last['close']), float(last['open']))
        body_bot = min(float(last['close']), float(last['open']))
        if direction == 1:
            return float(last['high']) > level and body_top <= level * 1.002
        else:
            return float(last['low']) < level and body_bot >= level * 0.998
    except:
        return False


def struct_phase(df, sw=3):
    in_bal, t_dir, *_ = get_struct_levels(df, sw=sw)
    return in_bal, t_dir


def get_levels(df, tf):
    """
    Единственный источник VAH/VAL/POC.
    Структурные границы → Volume Profile внутри.
    """
    profiles = build_balance_profiles(df, tf, max_balances=1)
    if not profiles:
        return np.nan, np.nan, np.nan, None, np.nan, np.nan
    last = profiles[-1]
    return last['vah'], last['val'], last['poc'], last['mode'], last['high'], last['low']


def get_phase(df, vah, val):
    """BALANCE если цена внутри VAH-VAL."""
    lc = float(df['close'].iloc[-1])
    if np.isnan(vah) or np.isnan(val):
        mid = float(df['close'].rolling(20).mean().iloc[-1])
        return False, 1 if lc > mid else -1
    return val <= lc <= vah, 1 if lc > vah else -1


def find_all_balances(df, tf='15m', max_balances=5):
    """Публичный API для отрисовки на графике."""
    return build_balance_profiles(df, tf, max_balances)


# ── Алиасы для совместимости ────────────────────────────────────────────────
def get_profile_for_tf(df, tf):
    p = build_balance_profiles(df, tf, max_balances=1)
    if not p: return None, None, None, None, None, None
    l = p[-1]; return l['vah'], l['val'], l['poc'], l['mode'], l['high'], l['low']

def find_balance_range(df, **kw):                return None
def build_balances(df, tf='15m', max_balances=5): return build_balance_profiles(df, tf, max_balances)
def compute_profiles(df, balances=None):          return build_balance_profiles(df)
def detect_balance_ranges(df, **kw):              return build_balance_profiles(df)
def get_profiles_from_balances(df, tf='15m', max_balances=5): return build_balance_profiles(df, tf, max_balances)



def detect_market_state(df, lookback=50, tf='15m'):
    """
    Правильная логика по Далтону:
    1. VAH/VAL/POC = структурный Volume Profile (через баланс)
    2. in_balance / trend_dir = на основе этих уровней
    3. hist_pocs = память рынка (завершённые балансы)

    Профиль строится через get_profile_for_tf() — баланс → объём → уровни.
    НЕ использует произвольное окно lookback для уровней.
    """
    n = len(df)

    # ── ШАГ 1: структурный VAH/VAL/POC
    vah, val, current_poc, _mode, _bh, _bl = get_profile_for_tf(df, tf)

    # Последняя цена
    last_close = float(df['close'].iloc[-1])

    # ── ШАГ 2: определяем фазу — защита от None если баланс не найден
    if vah is None or val is None:
        # Нет структурного баланса — fallback на простой расчёт
        sl = df.iloc[max(0, n-50):]
        try:
            vah, val, current_poc = calc_value_area(
                sl['high'].values, sl['low'].values, sl['volume'].values
            )
        except:
            vah = float(df['high'].iloc[-20:].max())
            val = float(df['low'].iloc[-20:].min())
            current_poc = (vah + val) / 2

    vah = float(vah); val = float(val)
    current_poc = float(current_poc) if current_poc is not None else (vah + val) / 2

    in_balance = (last_close <= vah) and (last_close >= val)
    trend_dir  = 1 if last_close > vah else -1

    # ── ШАГ 3: исторические POC из State Machine (те же балансы что на графике)
    all_profiles = build_balance_profiles(df, tf, max_balances=10)
    hist_pocs = []
    for p in all_profiles:
        if not p.get('is_current', False) and not np.isnan(p['poc']):
            hist_pocs.append({
                'x0':  p['x0'],
                'x1':  p['x1'],
                'poc': p['poc'],
            })

    return vah, val, in_balance, trend_dir, current_poc, hist_pocs


# ─────────────────────────────────────────────────────────────────────────────
# 6. ДИВЕРГЕНЦИИ CVD / ЦЕНА
# ─────────────────────────────────────────────────────────────────────────────



def check_sweep(df, level, direction='bearish'):
    """
    Свип ликвидности (ложный пробой) заданного уровня.
    Bearish: тень пробила уровень вверх, тело закрылось ниже → шорт
    Bullish: тень пробила уровень вниз, тело закрылось выше → лонг
    """
    if level is None or (isinstance(level, float) and np.isnan(level)):
        return False
    last = df.iloc[-1]
    if direction == 'bearish':
        return float(last['high']) > level and float(last['close']) < level
    else:
        return float(last['low']) < level and float(last['close']) > level


def check_pullback_to_broken_key_high(df, key_high):
    """
    Breakout Retest: BOS вверх + откат к пробитому Key High.
    Цена находится в диапазоне [key_high, key_high * 1.003].
    """
    if key_high is None or (isinstance(key_high, float) and np.isnan(key_high)):
        return False
    lc = float(df['close'].iloc[-1])
    # Цена должна была быть выше и теперь вернулась к уровню
    was_above = float(df['close'].iloc[-5:-1].max()) > key_high * 1.002
    near_now  = key_high <= lc <= key_high * 1.005
    return was_above and near_now


def check_pullback_to_broken_key_low(df, key_low):
    """
    Breakout Retest вниз: BOS вниз + откат к пробитому Key Low.
    """
    if key_low is None or (isinstance(key_low, float) and np.isnan(key_low)):
        return False
    lc = float(df['close'].iloc[-1])
    was_below = float(df['close'].iloc[-5:-1].min()) < key_low * 0.998
    near_now  = key_low * 0.995 <= lc <= key_low
    return was_below and near_now


def check_price_in_demand_ftr(df, active_ftr_zones, key_low=None):
    """
    Цена вошла в активную Demand FTR зону (First Time Back).
    Требует: зона выше Key Low + цена уже уходила выше зоны (FTB).
    """
    lc = float(df['close'].iloc[-1])
    for zone in active_ftr_zones:
        if not zone.get('active', False):
            continue
        if zone.get('dir') == 1 and zone['zl'] <= lc <= zone['zh']:
            if key_low is not None and not np.isnan(key_low) and zone['zl'] <= key_low:
                continue  # зона ниже Key Low — вне структуры
            if _ftb_check(df, zone):  # FTB обязателен
                return True
    return False


def check_price_in_supply_ftr(df, active_ftr_zones, key_high=None):
    """
    Цена вошла в активную Supply FTR зону (First Time Back).
    Требует: зона ниже Key High + цена уже уходила ниже зоны (FTB).
    """
    lc = float(df['close'].iloc[-1])
    for zone in active_ftr_zones:
        if not zone.get('active', False):
            continue
        if zone.get('dir') == -1 and zone['zl'] <= lc <= zone['zh']:
            if key_high is not None and not np.isnan(key_high) and zone['zh'] >= key_high:
                continue  # зона выше Key High — вне структуры
            if _ftb_check(df, zone):  # FTB обязателен
                return True
    return False


def get_current_vp_levels(df, tf='15m'):
    """
    Возвращает текущие VAH/VAL/POC из последнего структурного баланса.
    Используется Радаром для поиска тестов объёмных уровней.
    """
    try:
        _, _, kh, kl, vah, val, poc, _, _, _, _ = get_struct_levels(df, tf=tf, sw=3)
        if not np.isnan(vah):
            return {'vah': float(vah), 'val': float(val), 'poc': float(poc)}
    except:
        pass
    return None


def was_above_vah(df, vah, lookback=10):
    """
    Была ли цена недавно выше VAH (для поиска ретеста POC после выхода из баланса).
    """
    if vah is None or np.isnan(vah):
        return False
    return any(float(h) > vah for h in df['high'].iloc[-lookback:])


def was_below_val(df, val, lookback=10):
    """Была ли цена недавно ниже VAL."""
    if val is None or np.isnan(val):
        return False
    return any(float(l) < val for l in df['low'].iloc[-lookback:])


def check_cvd_divergence(df):
    """
    Быстрая проверка дивергенции CVD для Радара.
    Возвращает тип дивергенции:
      'BULL_ABSORPTION', 'BEAR_ABSORPTION'  — поглощение (сильнейший сигнал)
      'BULL_EXHAUSTION', 'BEAR_EXHAUSTION'  — классическая разворотная
      'BULL_HIDDEN',     'BEAR_HIDDEN'      — скрытая (подтверждение тренда)
    или None.
    """
    if 'cvd' not in df.columns or len(df) < 30:
        return None
    try:
        divs = find_divergences(df, lookback=5, min_dist=8)
        if not divs:
            return None
        # Берём самую свежую дивергенцию
        n = len(df)
        for dv in reversed(divs):
            ts_x1 = dv['x1']
            dist  = n - np.searchsorted(df['timestamp'].values, ts_x1)
            if dist <= 5:  # только последние 5 баров
                subtype = dv.get('subtype', 'exhaustion')
                if dv['type'] == 'bull':
                    if subtype == 'absorption': return 'BULL_ABSORPTION'
                    if subtype == 'hidden':     return 'BULL_HIDDEN'
                    return 'BULL_EXHAUSTION'
                else:
                    if subtype == 'absorption': return 'BEAR_ABSORPTION'
                    if subtype == 'hidden':     return 'BEAR_HIDDEN'
                    return 'BEAR_EXHAUSTION'
    except:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
def build_renko(df, asset):
    # Используем последнюю актуальную цену для расчёта размера кирпича.
    # Ранее использовалась avg_p (среднее за всю историю) — это давало
    # заниженный кирпич если текущая цена выше исторической средней.
    last_p = float(df['close'].iloc[-1])
    # Размер кирпича адаптирован под инструмент
    if "BTC" in asset:
        k = 0.0025
    elif "ETH" in asset:
        k = 0.0040
    elif asset in ("EURUSD=X", "GBPUSD=X", "EURUSD", "GBPUSD"):
        # Forex: пары торгуются около 1.0-1.3, нужен очень маленький кирпич
        k = 0.0008   # ~0.08% = ~8 пипсов для EUR/USD
    elif "GC=F" in asset or "GOLD" in asset or "XAU" in asset:
        k = 0.0020   # Золото ~2000$, кирпич ~4$
    else:
        k = 0.0035

    brick = last_p * k
    if brick <= 0:
        brick = last_p * 0.003

    closes = df['close'].values
    times  = df['timestamp'].values
    vols   = df['volume'].values

    # Forex (Yahoo Finance) даёт тиковый объём — почти константу.
    # Gimelfarb на таком объёме даёт мусорный CVD.
    # Используем Money Flow формулу: ((C-L)-(H-C))/(H-L)*V
    # Она оценивает где закрылась цена относительно диапазона свечи —
    # это намного информативнее чем просто направление (C-O)*V.
    # При H==L (дожи) результат = 0 чтобы избежать деления на ноль.
    is_forex = asset in ("EURUSD=X", "GBPUSD=X", "EURUSD", "GBPUSD")
    if is_forex:
        hi  = df['high'].values
        lo  = df['low'].values
        cl  = df['close'].values
        vol = df['volume'].values
        hl  = hi - lo
        mf  = np.where(hl > 0, ((cl - lo) - (hi - cl)) / hl * vol, 0.0)
        deltas = mf
    else:
        deltas = calc_gimelfarb_delta(df)

    # Накопление дельты по свечам внутри кирпича.
    #
    # Равномерное распределение: если одна (или несколько) свечей
    # формируют N кирпичей, накопленная дельта делится поровну на N.
    # Это исключает раздувание CVD когда волатильная свеча создаёт
    # 3-5 кирпичей и каждый получал бы полную дельту (ошибка x3-x5).
    #
    # Алгоритм:
    # 1. Накапливаем дельты свечей в pending_deltas пока не форм. кирпич
    # 2. Считаем сколько кирпичей сформирует текущая цена (bricks_up/dn)
    # 3. Делим total_delta / N — каждый кирпич получает равную долю
    # 4. Сбрасываем pending после распределения
    pending_deltas = []
    pending_vol    = 0.0

    bricks, cur = [], closes[0]
    for i in range(1, len(closes)):
        p = closes[i]
        pending_deltas.append(float(deltas[i]))
        pending_vol += float(vols[i])

        # Считаем сколько кирпичей вверх сформирует эта цена
        bricks_up = 0
        temp_cur = cur
        while p >= temp_cur + brick:
            bricks_up += 1
            temp_cur  += brick

        # Считаем сколько кирпичей вниз
        bricks_dn = 0
        temp_cur = cur
        while p <= temp_cur - brick:
            bricks_dn += 1
            temp_cur  -= brick

        n_bricks = bricks_up + bricks_dn
        if n_bricks == 0:
            continue  # кирпич ещё не сформировался — накапливаем дальше

        # Равномерное распределение дельты и объёма по всем кирпичам
        d_total       = sum(pending_deltas)
        delta_per_b   = d_total / n_bricks
        vol_per_b     = pending_vol / n_bricks
        d_max         = max(pending_deltas) if pending_deltas else 0.0
        d_min         = min(pending_deltas) if pending_deltas else 0.0
        d_first       = pending_deltas[0]   if pending_deltas else 0.0
        d_last        = pending_deltas[-1]  if pending_deltas else 0.0

        for _ in range(bricks_up):
            bricks.append({
                'open': cur, 'close': cur + brick,
                'high': cur + brick, 'low': cur,
                'time': times[i], 'vol': vol_per_b,
                'delta': delta_per_b, 'bull': True,
                'delta_max': d_max, 'delta_min': d_min,
                'delta_first': d_first, 'delta_last': d_last,
            })
            cur += brick

        for _ in range(bricks_dn):
            bricks.append({
                'open': cur, 'close': cur - brick,
                'high': cur, 'low': cur - brick,
                'time': times[i], 'vol': vol_per_b,
                'delta': delta_per_b, 'bull': False,
                'delta_max': d_max, 'delta_min': d_min,
                'delta_first': d_first, 'delta_last': d_last,
            })
            cur -= brick

        # Сбрасываем после распределения по всем кирпичам
        pending_deltas = []
        pending_vol    = 0.0

    if not bricks: return pd.DataFrame()
    rb = pd.DataFrame(bricks).reset_index(drop=True)

    vm   = rb['vol'].rolling(30).mean()
    vstd = rb['vol'].rolling(30).std()
    rb['v_col'] = '#DFDFDF'
    rb.loc[rb['vol'] >= vm + vstd, 'v_col'] = '#CF0909'
    rb.loc[rb['vol'] <= vm - vstd, 'v_col'] = '#FACA2E'

    # CVD со сбросом по торговым сессиям (каждый новый день).
    # Без сброса CVD "дрейфует" в бесконечность и теряет читаемость.
    # Определяем дату каждого кирпича через поле time (timestamp свечи).
    try:
        brick_dates = pd.to_datetime(rb['time'], unit='ms', utc=True).dt.date.values
        cvd_session = np.zeros(len(rb))
        session_acc = 0.0
        cur_date    = brick_dates[0]
        for idx_b in range(len(rb)):
            if brick_dates[idx_b] != cur_date:
                session_acc = 0.0          # сброс при смене дня
                cur_date    = brick_dates[idx_b]
            session_acc      += float(rb['delta'].iloc[idx_b])
            cvd_session[idx_b] = session_acc
        rb['cvd'] = cvd_session
    except Exception:
        # Fallback — обычный cumsum если timestamp недоступен
        rb['cvd'] = rb['delta'].cumsum()

    rb['brick_size'] = brick
    return rb


def _is_line_unbroken(series_values, x1, x2, is_bullish):
    """
    Геометрический фильтр чистоты дивергенции.

    Проверяет что прямая линия между точками x1 и x2 не пробивается
    графиком между ними:
    - Медвежья (is_bullish=False): ни одна точка не должна подняться ВЫШЕ линии
    - Бычья    (is_bullish=True):  ни одна точка не должна опуститься НИЖЕ линии

    Без этого фильтра линия может "прошивать" CVD насквозь если между
    двумя пиками/впадинами есть всплеск объёма.
    """
    if x2 - x1 <= 1:
        return True
    y1    = series_values[x1]
    y2    = series_values[x2]
    slope = (y2 - y1) / (x2 - x1)
    for x in range(x1 + 1, x2):
        line_y   = y1 + slope * (x - x1)
        actual_y = series_values[x]
        if is_bullish and actual_y < line_y:
            return False
        if not is_bullish and actual_y > line_y:
            return False
    return True


def find_structural_divergences(rb, reversal_bricks=4):
    """
    Структурные дивергенции CVD vs Цена на Renko.

    Алгоритм ZigZag с подтверждением экстремума + геометрический фильтр:
    - Пик подтверждён когда цена откатила на reversal_bricks кирпичей вниз.
    - Впадина подтверждена когда цена выросла на reversal_bricks кирпичей.
    - После нахождения дивергенции применяется _is_line_unbroken:
      линия не должна "прошивать" ни ценовой график ни CVD между точками.

    Бычья:   цена LL + CVD HL → накопление (🟢 Спринг)
    Медвежья: цена HH + CVD LH → дистрибуция (🔴 Аптраст)
    """
    if rb is None or len(rb) < 10:
        return [], []

    brick_size = float(rb['brick_size'].iloc[0]) if 'brick_size' in rb.columns else 1.0
    threshold  = reversal_bricks * brick_size
    closes     = rb['close'].values
    cvd_vals   = rb['cvd'].values

    bullish_divs = []
    bearish_divs = []

    mode = 'up' if closes[1] > closes[0] else 'down'
    current_extreme_idx   = 0
    current_extreme_price = closes[0]
    last_confirmed_top_idx    = None
    last_confirmed_bottom_idx = None

    for i in range(1, len(rb)):
        price = closes[i]

        if mode == 'up':
            if price >= current_extreme_price:
                current_extreme_price = price
                current_extreme_idx   = i
            elif price <= current_extreme_price - threshold:
                confirmed_top_idx = current_extreme_idx

                if last_confirmed_top_idx is not None:
                    p_prev = closes[last_confirmed_top_idx]
                    c_prev = cvd_vals[last_confirmed_top_idx]
                    p_curr = closes[confirmed_top_idx]
                    c_curr = cvd_vals[confirmed_top_idx]
                    if p_curr > p_prev and c_curr < c_prev:
                        # Геометрический фильтр: линия не должна пробиваться
                        price_clean = _is_line_unbroken(
                            closes, last_confirmed_top_idx, confirmed_top_idx, False)
                        cvd_clean   = _is_line_unbroken(
                            cvd_vals, last_confirmed_top_idx, confirmed_top_idx, False)
                        if price_clean and cvd_clean:
                            bearish_divs.append((last_confirmed_top_idx, confirmed_top_idx))

                last_confirmed_top_idx = confirmed_top_idx
                mode                   = 'down'
                current_extreme_price  = price
                current_extreme_idx    = i

        else:
            if price <= current_extreme_price:
                current_extreme_price = price
                current_extreme_idx   = i
            elif price >= current_extreme_price + threshold:
                confirmed_bottom_idx = current_extreme_idx

                if last_confirmed_bottom_idx is not None:
                    p_prev = closes[last_confirmed_bottom_idx]
                    c_prev = cvd_vals[last_confirmed_bottom_idx]
                    p_curr = closes[confirmed_bottom_idx]
                    c_curr = cvd_vals[confirmed_bottom_idx]
                    if p_curr < p_prev and c_curr > c_prev:
                        # Геометрический фильтр
                        price_clean = _is_line_unbroken(
                            closes, last_confirmed_bottom_idx, confirmed_bottom_idx, True)
                        cvd_clean   = _is_line_unbroken(
                            cvd_vals, last_confirmed_bottom_idx, confirmed_bottom_idx, True)
                        if price_clean and cvd_clean:
                            bullish_divs.append((last_confirmed_bottom_idx, confirmed_bottom_idx))

                last_confirmed_bottom_idx = confirmed_bottom_idx
                mode                      = 'up'
                current_extreme_price     = price
                current_extreme_idx       = i

    return bullish_divs, bearish_divs



# ═══════════════════════════════════════════════════════════════════════════════
# ЛАБОРАТОРИЯ 2026 — ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (Группа 2)
# Изолированы от основного кода. Используются только вкладками 18-21.
# ═══════════════════════════════════════════════════════════════════════════════

def lab_calc_fvg(df, atr_mult=1.0):
    """
    Fair Value Gap (FVG) детектор для Лаборатории.

    Три-свечной паттерн дисбаланса:
    - Бычий FVG:   low[i] > high[i-2]  → пустота цены снизу вверх
    - Медвежий FVG: high[i] < low[i-2] → пустота цены сверху вниз

    Зона митигирована если после её появления цена (high или low)
    пересекла середину зоны — тогда не рисуем.
    Активные зоны рисуются от момента появления до правого края графика.
    """
    if df is None or len(df) < 3:
        return []

    hi   = df['high'].values
    lo   = df['low'].values
    ts   = df['timestamp'].values
    n    = len(df)
    atr  = (df['high'] - df['low']).rolling(14).mean().values
    fvgs = []

    for i in range(2, n):
        av = float(atr[i]) if not np.isnan(atr[i]) else 0
        if av == 0:
            continue

        # Бычий FVG: low[i] выше high[i-2]
        gap_up = float(lo[i]) - float(hi[i-2])
        if gap_up > av * atr_mult:
            y0, y1 = float(hi[i-2]), float(lo[i])
            mid    = (y0 + y1) / 2
            # Митигация: цена касалась mid после появления зоны
            mitigated = any(lo[j] <= mid for j in range(i+1, n))
            fvgs.append({
                'dir': 1, 'y0': y0, 'y1': y1,
                'ts_start': ts[i-2],   # начало зоны (левый край)
                'ts_end':   ts[-1],    # правый край = последняя свеча
                'mitigated': mitigated
            })

        # Медвежий FVG: high[i] ниже low[i-2]
        gap_dn = float(lo[i-2]) - float(hi[i])
        if gap_dn > av * atr_mult:
            y0, y1 = float(hi[i]), float(lo[i-2])
            mid    = (y0 + y1) / 2
            mitigated = any(hi[j] >= mid for j in range(i+1, n))
            fvgs.append({
                'dir': -1, 'y0': y0, 'y1': y1,
                'ts_start': ts[i-2],
                'ts_end':   ts[-1],
                'mitigated': mitigated
            })

    # Только немитигированные, последние 6
    active = [z for z in fvgs if not z['mitigated']]
    return active[-6:]


def lab_calc_atr_stops(df, mult=1.5, period=14):
    """
    Динамические уровни стоп-лосса на основе ATR.
    Возвращает (atr_value, stop_long, stop_short) для последней свечи.
    """
    if df is None or len(df) < period:
        return None, None, None
    atr   = (df['high'] - df['low']).rolling(period).mean().iloc[-1]
    close = float(df['close'].iloc[-1])
    return float(atr), close - mult * float(atr), close + mult * float(atr)


def lab_calc_ob(df, sw=3):
    """
    Order Block детектор для Лаборатории.
    OB = последняя противоположная свеча перед импульсным BOS.
    Бычий OB:  последняя медвежья свеча перед сильным ростом.
    Медвежий OB: последняя бычья свеча перед сильным падением.

    Митигация: если цена вернулась в зону OB (пересекла середину)
    после его формирования — OB считается отработанным.
    Возвращает список: {dir, y0, y1, ts_start, ts_end, mitigated}
    """
    if df is None or len(df) < sw * 2 + 5:
        return []

    n   = len(df)
    obs = []
    hi  = df['high'].values
    lo  = df['low'].values
    cl  = df['close'].values
    op  = df['open'].values
    ts  = df['timestamp'].values
    ts_last = ts[-1]

    for i in range(sw + 2, n - sw):
        # Swing High → медвежий OB (последняя бычья свеча перед пиком)
        if hi[i] == max(hi[i-sw:i+sw+1]):
            for k in range(i-1, max(i-10, 0), -1):
                if cl[k] > op[k]:
                    y0, y1 = float(lo[k]), float(hi[k])
                    mid = (y0 + y1) / 2
                    # Митигация: high пересёк середину зоны после формирования
                    mitigated = any(hi[j] >= mid for j in range(i+1, n))
                    obs.append({
                        'dir': -1, 'y0': y0, 'y1': y1,
                        'ts_start': ts[k], 'ts_end': ts_last,
                        'mitigated': mitigated
                    })
                    break
        # Swing Low → бычий OB (последняя медвежья свеча перед дном)
        if lo[i] == min(lo[i-sw:i+sw+1]):
            for k in range(i-1, max(i-10, 0), -1):
                if cl[k] < op[k]:
                    y0, y1 = float(lo[k]), float(hi[k])
                    mid = (y0 + y1) / 2
                    # Митигация: low пробил середину зоны после формирования
                    mitigated = any(lo[j] <= mid for j in range(i+1, n))
                    obs.append({
                        'dir': 1, 'y0': y0, 'y1': y1,
                        'ts_start': ts[k], 'ts_end': ts_last,
                        'mitigated': mitigated
                    })
                    break

    # Дедупликация по ts_start, только немитигированные, последние 6
    seen = set()
    result = []
    for ob in obs:
        key = ob['ts_start']
        if key not in seen and not ob['mitigated']:
            seen.add(key)
            result.append(ob)

    return result[-6:]



def lab_calc_vsa(df):
    """
    VSA (Volume Spread Analysis) паттерны для Лаборатории.
    Три паттерна из исследования:
    - Selling Climax: высокий объём + широкий спред + закрытие в верхней половине
    - Upthrust:       высокий объём + обновление хая + закрытие у минимума (пин-бар)
    - No Demand:      восходящая свеча + узкий спред + объём ниже 2 предыдущих
    Возвращает список: {type, ts, price, label}
    """
    if df is None or len(df) < 10:
        return []

    hi  = df['high'].values
    lo  = df['low'].values
    cl  = df['close'].values
    op  = df['open'].values
    vol = df['volume'].values
    ts  = df['timestamp'].values
    n   = len(df)

    spread    = hi - lo
    vol_ma    = pd.Series(vol).rolling(20).mean().values
    spread_ma = pd.Series(spread).rolling(20).mean().values
    signals   = []

    for i in range(20, n):
        if spread_ma[i] == 0 or vol_ma[i] == 0:
            continue

        mid    = (hi[i] + lo[i]) / 2
        is_bear = cl[i] < op[i]
        is_bull = cl[i] > op[i]

        # Selling Climax: медвежья свеча, высокий объём, широкий спред,
        # но закрытие В ВЕРХНЕЙ половине — покупатели поглотили панику
        if (is_bear and
            vol[i] > vol_ma[i] * 1.8 and
            spread[i] > spread_ma[i] * 1.5 and
            cl[i] > mid):
            signals.append({
                'type': 'SC', 'ts': ts[i], 'price': lo[i],
                'label': '🟡 SC', 'color': '#FFD700'
            })

        # Upthrust: новый хай, высокий объём, закрытие у нижней трети диапазона
        if (hi[i] > hi[i-1] and
            vol[i] > vol_ma[i] * 1.5 and
            spread[i] > spread_ma[i] * 1.2 and
            cl[i] < lo[i] + spread[i] * 0.35):
            signals.append({
                'type': 'UT', 'ts': ts[i], 'price': hi[i],
                'label': '🔴 UT', 'color': '#FF4444'
            })

        # No Demand: бычья свеча, узкий спред, объём ниже 2 предыдущих
        if (is_bull and
            spread[i] < spread_ma[i] * 0.7 and
            vol[i] < vol[i-1] and vol[i] < vol[i-2] and
            cl[i] < mid + spread[i] * 0.2):
            signals.append({
                'type': 'ND', 'ts': ts[i], 'price': hi[i],
                'label': '⚪ ND', 'color': '#AAAAAA'
            })

    return signals[-30:]  # последние 30 сигналов


def lab_calc_rsi_divergence(df, period=14, sw=5):
    """
    RSI + поиск дивергенций цена/RSI.
    Возвращает (rsi_series, bull_divs, bear_divs).
    bull_divs / bear_divs = список (ts1, ts2, p1, p2, r1, r2)
    """
    if df is None or len(df) < period + sw * 2:
        return None, [], []

    delta = df['close'].diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = (100 - 100 / (1 + rs)).values
    ts    = df['timestamp'].values
    pr    = df['close'].values
    n     = len(df)

    bull_divs, bear_divs = [], []

    # Ищем пивоты цены и сравниваем с RSI
    for i in range(sw + period, n - sw):
        # Пивот Low цены
        if pr[i] == min(pr[i-sw:i+sw+1]) and not np.isnan(rsi[i]):
            # Ищем предыдущий пивот Low
            for j in range(i - sw - 1, sw + period - 1, -1):
                if pr[j] == min(pr[j-sw:j+sw+1]) and not np.isnan(rsi[j]):
                    if i - j < 50:  # не слишком далеко
                        # Бычья дивергенция: цена LL, RSI HL
                        if pr[i] < pr[j] and rsi[i] > rsi[j]:
                            bull_divs.append((ts[j], ts[i], pr[j], pr[i], rsi[j], rsi[i]))
                    break

        # Пивот High цены
        if pr[i] == max(pr[i-sw:i+sw+1]) and not np.isnan(rsi[i]):
            for j in range(i - sw - 1, sw + period - 1, -1):
                if pr[j] == max(pr[j-sw:j+sw+1]) and not np.isnan(rsi[j]):
                    if i - j < 50:
                        # Медвежья дивергенция: цена HH, RSI LH
                        if pr[i] > pr[j] and rsi[i] < rsi[j]:
                            bear_divs.append((ts[j], ts[i], pr[j], pr[i], rsi[j], rsi[i]))
                    break

    return rsi, bull_divs[-5:], bear_divs[-5:]


def lab_calc_bb_squeeze(df, period=20, mult=2.0):
    """
    Bollinger Bands + Squeeze детектор.
    Squeeze = полосы сужаются (волатильность падает) перед импульсом.
    Возвращает (upper, mid, lower, squeeze_bool_series).
    """
    if df is None or len(df) < period:
        return None, None, None, None
    cl    = df['close']
    mid   = cl.rolling(period).mean()
    std   = cl.rolling(period).std()
    upper = mid + mult * std
    lower = mid - mult * std
    width = (upper - lower) / mid
    # Squeeze: текущая ширина ниже минимума за последние 50 периодов
    min_w = width.rolling(50).min()
    squeeze = (width <= min_w * 1.05)
    return upper.values, mid.values, lower.values, squeeze.values


def lab_calc_asian_session(df):
    """
    ICT Power of 3 — определяем диапазон Азиатской сессии (00:00-08:00 UTC).
    Возвращает список: {date, high, low, ts_start, ts_end}
    """
    if df is None or len(df) < 2:
        return []

    try:
        times = pd.to_datetime(df['timestamp'], utc=True)
    except Exception:
        return []

    hi  = df['high'].values
    lo  = df['low'].values

    sessions = []
    by_date  = {}

    for i in range(len(df)):
        t  = times.iloc[i]
        if 0 <= t.hour < 8:
            d = t.date()
            if d not in by_date:
                by_date[d] = {'highs': [], 'lows': [], 'start': t, 'end': t}
            by_date[d]['highs'].append(float(hi[i]))
            by_date[d]['lows'].append(float(lo[i]))
            by_date[d]['end'] = t

    for d, v in list(by_date.items())[-10:]:  # последние 10 дней
        if v['highs']:
            sessions.append({
                'date':     d,
                'high':     max(v['highs']),
                'low':      min(v['lows']),
                'ts_start': v['start'],
                'ts_end':   pd.Timestamp(d, tz='UTC') + pd.Timedelta(hours=23, minutes=59),
            })

    return sessions


def lab_load_spot_bybit(symbol, limit=300):
    """Загружает 1D спот данные с Bybit."""
    import requests as _req
    try:
        r = _req.get("https://api.bybit.com/v5/market/kline",
            params={"category":"spot","symbol":symbol,"interval":"D","limit":limit},
            timeout=10)
        raw = r.json().get('result',{}).get('list',[])
        if not raw:
            return None
        df = pd.DataFrame(raw, columns=['ts','open','high','low','close','volume','turnover'])
        for c in ['open','high','low','close','volume']:
            df[c] = df[c].astype(float)
        df['timestamp'] = pd.to_datetime(df['ts'].astype(float), unit='ms')
        return df.sort_values('timestamp').reset_index(drop=True)
    except Exception:
        return None


def lab_load_perp_bybit(symbol, limit=300):
    """Загружает 1D фьючерс (linear perp) данные с Bybit."""
    import requests as _req
    try:
        r = _req.get("https://api.bybit.com/v5/market/kline",
            params={"category":"linear","symbol":symbol,"interval":"D","limit":limit},
            timeout=10)
        raw = r.json().get('result',{}).get('list',[])
        if not raw:
            return None
        df = pd.DataFrame(raw, columns=['ts','open','high','low','close','volume','turnover'])
        for c in ['open','high','low','close','volume']:
            df[c] = df[c].astype(float)
        df['timestamp'] = pd.to_datetime(df['ts'].astype(float), unit='ms')
        return df.sort_values('timestamp').reset_index(drop=True)
    except Exception:
        return None


def lab_calc_zscore_spread(df1, df2, window=30):
    """
    Z-Score спреда между двумя активами (коинтеграционный арбитраж).
    Используется для пар BTC/ETH, BTC/XRP и т.д.
    Z-Score > 2.0: первый актив переоценён → продавать первый, покупать второй.
    Z-Score < -2.0: второй актив переоценён → наоборот.
    """
    if df1 is None or df2 is None:
        return None, None
    # Выравниваем по времени
    merged = pd.merge(
        df1[['timestamp','close']].rename(columns={'close':'c1'}),
        df2[['timestamp','close']].rename(columns={'close':'c2'}),
        on='timestamp', how='inner')
    if len(merged) < window + 5:
        return None, None
    # Нормализованный спред
    spread = np.log(merged['c1'].values) - np.log(merged['c2'].values)
    spread_s = pd.Series(spread)
    mean_s   = spread_s.rolling(window).mean()
    std_s    = spread_s.rolling(window).std()
    zscore   = ((spread_s - mean_s) / std_s.replace(0, np.nan)).values
    return merged['timestamp'].values, zscore


def lab_calc_spot_perp_divergence(spot_df, perp_df):
    """
    Спот vs Фьючерс CVD дивергенция.
    Если фьюч CVD падает, а спот CVD держится → Long Squeeze → сигнал на лонг.
    Если фьюч CVD растёт, а спот CVD падает → Short Squeeze → сигнал на шорт.
    Возвращает: 'LONG_SQUEEZE', 'SHORT_SQUEEZE' или None.
    """
    if spot_df is None or perp_df is None:
        return None
    if 'cvd' not in spot_df.columns or 'cvd' not in perp_df.columns:
        return None
    # Последние 5 баров
    n = 5
    spot_cvd_chg = float(spot_df['cvd'].iloc[-1]) - float(spot_df['cvd'].iloc[-n])
    perp_cvd_chg = float(perp_df['cvd'].iloc[-1]) - float(perp_df['cvd'].iloc[-n])
    # Long Squeeze: фьюч CVD падает сильно, спот держится или растёт
    if perp_cvd_chg < 0 and spot_cvd_chg >= perp_cvd_chg * 0.3:
        return 'LONG_SQUEEZE'
    # Short Squeeze: спот CVD падает, фьюч растёт
    if spot_cvd_chg < 0 and perp_cvd_chg >= spot_cvd_chg * 0.3:
        return 'SHORT_SQUEEZE'
    return None


def lab_calc_adx(df, period=14):
    """
    Average Directional Index (ADX) — фильтр силы тренда.
    ADX < 20: флэт, нет тренда — сигналы игнорировать
    ADX 20-40: умеренный тренд
    ADX > 40: сильный тренд
    Возвращает (adx, plus_di, minus_di) как numpy arrays.
    """
    if df is None or len(df) < period * 2:
        return None, None, None
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    n  = len(df)

    tr    = np.zeros(n)
    pdm   = np.zeros(n)  # +DM
    ndm   = np.zeros(n)  # -DM

    for i in range(1, n):
        hl  = hi[i] - lo[i]
        hpc = abs(hi[i] - cl[i-1])
        lpc = abs(lo[i] - cl[i-1])
        tr[i] = max(hl, hpc, lpc)
        up   = hi[i] - hi[i-1]
        down = lo[i-1] - lo[i]
        pdm[i] = up   if (up > down and up > 0)   else 0
        ndm[i] = down if (down > up and down > 0) else 0

    # Smoothed (Wilder)
    atr_w  = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
    pdi_w  = pd.Series(pdm).ewm(alpha=1/period, adjust=False).mean().values
    ndi_w  = pd.Series(ndm).ewm(alpha=1/period, adjust=False).mean().values

    with np.errstate(divide='ignore', invalid='ignore'):
        pdi = np.where(atr_w > 0, 100 * pdi_w / atr_w, 0)
        ndi = np.where(atr_w > 0, 100 * ndi_w / atr_w, 0)
        dx  = np.where((pdi + ndi) > 0, 100 * np.abs(pdi - ndi) / (pdi + ndi), 0)

    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
    return adx, pdi, ndi


def lab_calc_squeeze_momentum(df, bb_period=20, bb_mult=2.0, kc_period=20, kc_mult=1.5):
    """
    Squeeze Momentum (LazyBear) — детектор сжатия волатильности.
    Squeeze ON:  BB внутри KC → серый крест → сжатие, движение готовится
    Squeeze OFF: BB вышла из KC → выход из сжатия → ВХОДИТЬ
    Momentum гистограмма = направление импульса после выхода из сжатия.

    Возвращает:
        squeeze_on  — bool array: True = сжатие активно
        squeeze_off — bool array: True = только что вышли из сжатия (сигнал!)
        momentum    — float array: > 0 бычий, < 0 медвежий
    """
    if df is None or len(df) < bb_period * 2:
        return None, None, None

    cl  = df['close'].values
    hi  = df['high'].values
    lo  = df['low'].values
    n   = len(df)

    # Bollinger Bands
    cl_s  = pd.Series(cl)
    bb_mid = cl_s.rolling(bb_period).mean().values
    bb_std = cl_s.rolling(bb_period).std().values
    bb_up  = bb_mid + bb_mult * bb_std
    bb_dn  = bb_mid - bb_mult * bb_std

    # Keltner Channels
    tr    = np.maximum(hi - lo,
            np.maximum(np.abs(hi - np.roll(cl, 1)),
                       np.abs(lo - np.roll(cl, 1))))
    tr[0] = hi[0] - lo[0]
    atr_k = pd.Series(tr).rolling(kc_period).mean().values
    kc_up = bb_mid + kc_mult * atr_k
    kc_dn = bb_mid - kc_mult * atr_k

    # Squeeze: BB внутри KC
    squeeze_on = (bb_up < kc_up) & (bb_dn > kc_dn)

    # Squeeze OFF: предыдущий бар был ON, текущий OFF → выход из сжатия
    squeeze_off = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if squeeze_on[i-1] and not squeeze_on[i]:
            squeeze_off[i] = True

    # Momentum = линейная регрессия (close - midpoint(high/low, bb_mid))
    delta = cl - (pd.Series((hi + lo) / 2).rolling(bb_period).mean().values +
                  bb_mid) / 2
    momentum = pd.Series(delta).rolling(bb_period).mean().values

    return squeeze_on, squeeze_off, momentum


def lab_get_silver_bullet_windows(df):
    """
    ICT Silver Bullet — временные окна алгоритмической ребалансировки.
    Окна (EST = UTC-5, летом UTC-4):
      London Open:  03:00-04:00 EST = 08:00-09:00 UTC
      NY AM:        10:00-11:00 EST = 15:00-16:00 UTC
      NY PM:        14:00-15:00 EST = 19:00-20:00 UTC
    Возвращает список {x0, x1, label, color} для отрисовки vrect.
    """
    if df is None or len(df) < 2:
        return []

    try:
        times = pd.to_datetime(df['timestamp'], utc=True)
    except Exception:
        return []

    windows = []
    seen_dates = set()

    for t in times:
        d = t.date()
        if d in seen_dates:
            continue
        seen_dates.add(d)
        base = pd.Timestamp(d, tz='UTC')

        windows.extend([
            {'x0': base + pd.Timedelta(hours=8),
             'x1': base + pd.Timedelta(hours=9),
             'label': '🗡 SB London', 'color': 'rgba(255,215,0,0.10)'},
            {'x0': base + pd.Timedelta(hours=15),
             'x1': base + pd.Timedelta(hours=16),
             'label': '🗡 SB NY AM',  'color': 'rgba(100,200,255,0.10)'},
            {'x0': base + pd.Timedelta(hours=19),
             'x1': base + pd.Timedelta(hours=20),
             'label': '🗡 SB NY PM',  'color': 'rgba(180,100,255,0.10)'},
        ])

    return windows[-60:]  # последние 20 дней × 3 окна

def lab_get_sessions(df):
    """
    Определяет временные зоны торговых сессий для каждой свечи.
    Возвращает список dict {ts_start, ts_end, session} для подсветки.
    Сессии в UTC:
      Азия:    00:00 - 08:00
      Лондон:  08:00 - 16:00 (Европа открывается в 07:00, активная фаза 08:00)
      Нью-Йорк: 13:00 - 21:00
    """
    if df is None or len(df) < 2:
        return []

    try:
        times = pd.to_datetime(df['timestamp'], utc=True)
    except Exception:
        return []

    sessions = []
    seen_dates = set()

    for i in range(len(df)):
        t  = times.iloc[i]
        dt = t.date()
        if dt in seen_dates:
            continue
        seen_dates.add(dt)

        base = pd.Timestamp(dt, tz='UTC')
        sessions.append({
            'label': 'Азия',
            'color': 'rgba(100,150,255,0.06)',
            'x0': base,
            'x1': base + pd.Timedelta(hours=8),
        })
        sessions.append({
            'label': 'Лондон',
            'color': 'rgba(255,200,50,0.07)',
            'x0': base + pd.Timedelta(hours=8),
            'x1': base + pd.Timedelta(hours=16),
        })
        sessions.append({
            'label': 'Нью-Йорк',
            'color': 'rgba(50,220,120,0.07)',
            'x0': base + pd.Timedelta(hours=13),
            'x1': base + pd.Timedelta(hours=21),
        })

    return sessions[-60:]  # последние 20 дней


def lab_add_fvg_to_fig(fig, fvgs, row=1):
    """Рисует FVG зоны на субплоте row.
    Каждая зона — прямоугольник от момента появления до правого края.
    Митигированные зоны не рисуются.
    """
    for z in fvgs:
        if z['mitigated']:
            continue
        color  = 'rgba(0,200,80,0.20)'  if z['dir'] == 1 else 'rgba(220,50,50,0.20)'
        border = 'rgba(0,220,80,0.7)'   if z['dir'] == 1 else 'rgba(220,50,50,0.7)'
        label  = '⬆ FVG'               if z['dir'] == 1 else '⬇ FVG'
        # Прямоугольник ограничен по X — от появления до последней свечи
        fig.add_shape(
            type='rect',
            x0=z['ts_start'], x1=z['ts_end'],
            y0=z['y0'],       y1=z['y1'],
            fillcolor=color,
            line=dict(color=border, width=1),
            row=row, col=1)
        fig.add_annotation(
            x=z['ts_start'],
            y=(z['y0'] + z['y1']) / 2,
            text=label, showarrow=False,
            font=dict(color=border, size=9),
            xanchor='left', row=row, col=1)


def lab_add_ob_to_fig(fig, obs, row=1):
    """Рисует Order Block зоны на субплоте row.
    Прямоугольник от момента появления OB до правого края.
    Митигированные OB не рисуются.
    """
    for ob in obs:
        if ob.get('mitigated', False):
            continue
        color  = 'rgba(0,180,255,0.15)' if ob['dir'] == 1 else 'rgba(255,140,0,0.15)'
        border = 'rgba(0,180,255,0.7)'  if ob['dir'] == 1 else 'rgba(255,140,0,0.7)'
        label  = '🔷 OB' if ob['dir'] == 1 else '🔶 OB'
        fig.add_shape(
            type='rect',
            x0=ob['ts_start'], x1=ob['ts_end'],
            y0=ob['y0'],       y1=ob['y1'],
            fillcolor=color,
            line=dict(color=border, width=1),
            row=row, col=1)
        fig.add_annotation(
            x=ob['ts_start'],
            y=(ob['y0'] + ob['y1']) / 2,
            text=label, showarrow=False,
            font=dict(color=border, size=9),
            xanchor='left', row=row, col=1)


def lab_add_sessions_to_fig(fig, sessions, row=1):
    """Рисует цветные полосы торговых сессий."""
    for s in sessions:
        fig.add_vrect(
            x0=s['x0'], x1=s['x1'],
            fillcolor=s['color'],
            line_width=0,
            row=row, col=1)


def lab_add_atr_stops_to_fig(fig, df, row=1):
    """Рисует динамические ATR стопы на графике."""
    atr_val, stop_long, stop_short = lab_calc_atr_stops(df)
    if atr_val is None:
        return
    last_ts = df['timestamp'].iloc[-1]
    prev_ts = df['timestamp'].iloc[-5]

    fig.add_shape(type='line',
        x0=prev_ts, x1=last_ts, y0=stop_long, y1=stop_long,
        line=dict(color='rgba(0,230,118,0.8)', width=1.5, dash='dash'),
        row=row, col=1)
    fig.add_shape(type='line',
        x0=prev_ts, x1=last_ts, y0=stop_short, y1=stop_short,
        line=dict(color='rgba(255,82,82,0.8)', width=1.5, dash='dash'),
        row=row, col=1)
    close = float(df['close'].iloc[-1])
    fig.add_annotation(x=last_ts, y=stop_long,
        text=f'SL Long {stop_long:.4f}', showarrow=False,
        font=dict(color='#00E676', size=9),
        xanchor='right', row=row, col=1)
    fig.add_annotation(x=last_ts, y=stop_short,
        text=f'SL Short {stop_short:.4f}', showarrow=False,
        font=dict(color='#FF5252', size=9),
        xanchor='right', row=row, col=1)


def _dpoc_colored_traces(dpoc_x, dpoc_y) -> list:
    """
    Разбивает линию dPOC на сегменты по направлению миграции и возвращает
    список Plotly-трасс с цветовой кодировкой.

    Цвета:
      ↑ Рост    → #00E676 (зелёный)  — институциональный спрос, бычье давление
      ↓ Падение → #FF1744 (красный)  — распределение, медвежье давление
      → Стоит   → #FFD600 (жёлтый)  — консолидация вокруг справедливой цены
    """
    if len(dpoc_y) < 2:
        return []

    DIR_COLORS = {'up': '#00E676', 'down': '#FF1744', 'flat': '#FFD600'}

    # Определяем направление каждого шага
    dirs = []
    for i in range(1, len(dpoc_y)):
        if dpoc_y[i] > dpoc_y[i - 1] + 1e-10:
            dirs.append('up')
        elif dpoc_y[i] < dpoc_y[i - 1] - 1e-10:
            dirs.append('down')
        else:
            dirs.append('flat')

    # Группируем смежные шаги одного направления в сегменты
    segments = []
    seg_start = 0
    cur_dir   = dirs[0]
    for i in range(1, len(dirs)):
        if dirs[i] != cur_dir:
            segments.append((seg_start, i, cur_dir))
            seg_start = i
            cur_dir   = dirs[i]
    segments.append((seg_start, len(dirs), cur_dir))

    traces = []
    for start, end, direction in segments:
        # +1 чтобы сегмент стыковался со следующим (нет разрывов на ступенях)
        seg_x = dpoc_x[start: end + 1]
        seg_y = dpoc_y[start: end + 1]
        traces.append(go.Scatter(
            x=seg_x, y=seg_y,
            mode='lines',
            line=dict(color=DIR_COLORS[direction], width=2, shape='hv'),
            showlegend=False,
            hoverinfo='skip',
        ))
    return traces


def calc_candle_divergences(df, lookback=5, min_dist=8):
    """Дивергенции CVD vs Price на свечных графиках."""
    try:
        df2 = df.copy()
        if 'cvd' not in df2.columns:
            if 'delta' in df2.columns:
                df2['cvd'] = df2['delta'].cumsum()
            else:
                return []
        return find_divergences(df2, lookback=lookback, min_dist=min_dist)
    except:
        return []


def calc_ftr_zones_renko(rb, asset, src="crypto"):
    """
    FTR зоны для Renko — специальная логика по кирпичам.

    Ключевые отличия от свечной логики:
    - Импульс = 2+ кирпича подряд одного направления
    - База = 1-3 кирпича перед импульсом
    - Объём подтверждение мягче (vol_mult ниже)
    - Параметры адаптированы под каждый инструмент
    """
    # Параметры по инструменту
    ALTS_MAIN = ["XRPUSDT","XLMUSDT","SOLUSDT","TRXUSDT","BCHUSDT","LTCUSDT",
                 "AAVEUSDT","BNBUSDT","AVAXUSDT","ETCUSDT","LINKUSDT","DOGEUSDT",
                 "NEARUSDT","ATOMUSDT","UNIUSDT","POLUSDT","ADAUSDT","DOTUSDT",
                 "ETHUSDT","OPUSDT","FILUSDT","ARBUSDT"]
    ALTS_2    = ["CAKEUSDT","WOOUSDT","VETUSDT","ZILUSDT","STXUSDT","WAVESUSDT",
                 "ZRXUSDT","XTZUSDT","JASMYUSDT","ORDIUSDT","TONUSDT","INJUSDT",
                 "SEIUSDT","PEOPLEUSDT","FLOKIUSDT","YGGUSDT","GALAUSDT","RUNEUSDT",
                 "CRVUSDT","ICPUSDT","MASKUSDT","APEUSDT","SUIUSDT","APTUSDT",
                 "MANAUSDT","THETAUSDT","GMTUSDT","HBARUSDT","SANDUSDT","KSMUSDT"]

    if "BTC" in asset:
        vol_mult, base_lb, min_dist, max_z = 1.5, 3, 20, 8
    elif "ETH" in asset or asset in ALTS_MAIN or asset in ALTS_2:
        vol_mult, base_lb, min_dist, max_z = 1.3, 3, 15, 10
    elif asset in ("GC=F",):         # Золото
        vol_mult, base_lb, min_dist, max_z = 1.6, 4, 25, 6
    elif asset in ("EURUSD=X",):     # EUR/USD
        vol_mult, base_lb, min_dist, max_z = 1.2, 3, 12, 12
    elif asset in ("GBPUSD=X",):     # GBP/USD
        vol_mult, base_lb, min_dist, max_z = 1.3, 3, 15, 10
    else:
        vol_mult, base_lb, min_dist, max_z = 1.3, 3, 15, 10

    n      = len(rb)
    vm     = rb['vol'].rolling(30).mean()
    vol_ok = rb['vol'] >= vm * vol_mult

    zones    = []
    last_bar = -min_dist

    for i in range(base_lb + 2, n):
        # ── Инвалидация активных зон
        for z in zones:
            if not z['active']: continue
            cur_close = float(rb['close'].iloc[i])
            if z['dir'] == 1  and cur_close < z['zl']: z['active'] = False
            if z['dir'] == -1 and cur_close > z['zh']: z['active'] = False

        # ── Касания (touch_count)
        for z in zones:
            if not z['active']: continue
            in_now  = z['zl'] <= float(rb['close'].iloc[i])   <= z['zh']
            in_prev = z['zl'] <= float(rb['close'].iloc[i-1]) <= z['zh']
            if in_now and not in_prev:
                z['touch_count'] = z.get('touch_count', 0) + 1
                if z['touch_count'] >= 3:
                    z['active'] = False

        # ── Поиск импульса: 2+ кирпича подряд одного направления
        impulse_len_bull = 0
        impulse_len_bear = 0
        for k in range(i, max(i-5, 0), -1):
            if rb['bull'].iloc[k]:
                impulse_len_bull += 1
            else:
                break
        for k in range(i, max(i-5, 0), -1):
            if not rb['bull'].iloc[k]:
                impulse_len_bear += 1
            else:
                break

        is_bull_imp = (impulse_len_bull >= 2) and bool(vol_ok.iloc[i])
        is_bear_imp = (impulse_len_bear >= 2) and bool(vol_ok.iloc[i])

        if not (is_bull_imp or is_bear_imp):
            continue
        if i - last_bar < min_dist:
            continue

        # ── База: base_lb кирпичей ДО импульса
        base_start = max(0, i - impulse_len_bull - base_lb) if is_bull_imp                      else max(0, i - impulse_len_bear - base_lb)
        base_end   = max(0, i - (impulse_len_bull if is_bull_imp else impulse_len_bear))

        if base_end <= base_start:
            continue

        base_slice = rb.iloc[base_start:base_end]
        zl = float(base_slice['low'].min())
        zh = float(base_slice['high'].max())
        if zh <= zl:
            continue

        direction = 1 if is_bull_imp else -1
        zones.append({
            'dir':         direction,
            'zl':          zl,
            'zh':          zh,
            'i':           i,
            'active':      True,
            'touch_count': 0,
            'impulse_len': impulse_len_bull if is_bull_imp else impulse_len_bear,
        })
        last_bar = i
        if len(zones) > max_z:
            zones.pop(0)

    return zones

# ─────────────────────────────────────────────────────────────────────────────
# 8. ИНТЕРФЕЙС
# ─────────────────────────────────────────────────────────────────────────────
# ── Запуск мониторинга безубытка (после определения всех функций)
if 'be_monitor_started' not in st.session_state:
    _start_breakeven_monitor_if_needed()
    st.session_state['be_monitor_started'] = True

# ── Автообновление
import streamlit.components.v1 as components

st.sidebar.header("Навигация")

# Кнопка ручного обновления
col_r1, col_r2 = st.sidebar.columns(2)
if col_r1.button("🔄 Обновить", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# Автообновление
auto_refresh = col_r2.toggle("⏱ Авто", value=False)
if auto_refresh:
    refresh_sec = st.sidebar.select_slider(
        "Интервал обновления:",
        options=[15, 30, 60, 120, 300],
        value=30,
        format_func=lambda x: f"{x} сек" if x < 60 else f"{x//60} мин"
    )
    # Инжектируем JS таймер который перезагружает страницу
    components.html(
        f"""
        <script>
            setTimeout(function() {{
                window.parent.location.reload();
            }}, {refresh_sec * 1000});
        </script>
        """,
        height=0,
    )
    st.sidebar.caption(f"⏱ Следующее обновление через {refresh_sec} сек")
tab = st.sidebar.radio("Вкладки:", [
    "📈 Crypto Candles",
    "💹 Forex Candles",
    "🧱 Renko",
    "🧠 ML Zone Lab",
    "📡 Strategy Live",
])
tf        = st.sidebar.radio("Таймфрейм:", ["15m","1h","1D"], horizontal=True)
show_divs = st.sidebar.checkbox("Дивергенции CVD", value=True)
show_mde  = st.sidebar.checkbox("Market Decision Engine", value=True)

# ── Telegram уведомления
st.sidebar.markdown("---")
with st.sidebar.expander("📲 Telegram & Сканер", expanded=not bool(TG_TOKEN)):
    if TG_TOKEN and TG_CHAT_ID:
        st.success("✅ Telegram подключён")
        st.markdown("---")

        # ── Выбор рынка
        scan_market = st.radio(
            "Рынок:", ["🔵 Крипто", "🟡 Форекс"],
            horizontal=True, key="scan_market"
        )
        # ── Выбор ТФ
        scan_tf = st.selectbox(
            "Таймфрейм:", ["15m", "1h", "1D"],
            index=0, key="scan_tf_select"
        )
        st.markdown("---")

        if scan_market == "🔵 Крипто":
            # ── Один сканер для всех крипто пар
            st.markdown("**⚡️ Сканер Крипто Сетапов**")
            st.caption("Breakout Retest · FTR · Liq.Grab · Failed Auction · Absorption")
            st.caption("Все 84 пары · RF фильтр · Выбранный ТФ")
            e_last = _scan_status.get('entry_last')
            if e_last:
                e_ago = int(time.time() - e_last)
                st.caption(f"⏱ {e_ago}с назад · Найдено: {_scan_status['entry_found']} · Всего: {_scan_status['entry_total']}")
            else:
                st.caption("Ещё не запускалось")
            if st.button(f"⚡️ Сканировать {scan_tf}", key="scan_entry_btn", use_container_width=True):
                with st.spinner(f"Сканирую все пары на {scan_tf}..."):
                    _run_entry_scan_once(tf=scan_tf)
                n = _scan_status['entry_found']
                if n > 0:
                    st.success(f"✅ Найдено сетапов: {n} — отправлено в TG")
                else:
                    st.info("Сетапов не найдено")

        else:
            # ── Форекс: все FTMO пары
            _FTMO_PAIRS = [
                "EUR/USD","GBP/USD","USD/JPY","USD/CHF","USD/CAD","AUD/USD","NZD/USD",
                "EUR/GBP","EUR/JPY","EUR/CHF","EUR/CAD","EUR/AUD","EUR/NZD",
                "GBP/JPY","GBP/CHF","GBP/CAD","GBP/AUD","GBP/NZD",
                "AUD/JPY","AUD/CHF","AUD/CAD","AUD/NZD",
                "NZD/JPY","NZD/CHF","NZD/CAD","CAD/JPY","CAD/CHF","CHF/JPY",
                "XAU/USD","XAG/USD",
                "US30","NAS100","SP500","DAX40","UK100","JPN225",
                "USOIL","UKOIL",
            ]
            st.markdown("**🟡 Сканер Форекс (FTMO)**")
            st.caption("Все инструменты FTMO — Majors, Crosses, Metals, Indices, Energy")
            if st.button(f"🟡 Сканировать Форекс {scan_tf}", key="scan_forex_btn", use_container_width=True):
                found_fx = 0
                errors_fx = 0
                with st.spinner(f"Сканирую {len(_FTMO_PAIRS)} форекс инструментов на {scan_tf}..."):
                    for fx_sym in _FTMO_PAIRS:
                        try:
                            result = scan_valid_setups(fx_sym, scan_tf)
                            if result:
                                tg_route_and_send(result)
                                found_fx += 1
                                print(f"[FOREX] {fx_sym} {scan_tf}: {result.get('setup')} {result.get('dir')}")
                        except Exception as e:
                            errors_fx += 1
                            print(f"[FOREX] Ошибка {fx_sym}: {e}")
                if found_fx > 0:
                    st.success(f"✅ Форекс сигналов: {found_fx} — отправлено в TG")
                else:
                    st.info(f"Форекс сигналов не найдено (ошибок: {errors_fx})")

        st.markdown("---")
        if st.button("🧪 Тест уведомления", key="tg_test"):
            tg_send("✅ <b>Pro Screener</b>: тест уведомления работает!")
            st.success("Отправлено!")
    else:
        st.warning("⚠️ Telegram не настроен")
        st.info("Установи переменные окружения на VPS:")
        st.code("export TG_BOT_TOKEN=токен\nexport TG_CHAT_ID=chat_id", language="bash")

# ── Настройки FTR зон
with st.sidebar.expander("⚙️ Настройки FTR зон (не BTC)", expanded=False):
    st.caption("⚠️ Для BTC параметры фиксированы по оригиналу AlgoPoint")
    ftr_atr_mult  = st.slider("ATR множитель (размер свечи)",  0.5, 3.0, 1.5, 0.1, key="ftr_atr")
    ftr_vol_mult  = st.slider("Vol множитель (SMA × ?)",       1.0, 4.0, 1.8, 0.1, key="ftr_vol")
    ftr_min_dist  = st.slider("Мин. баров между зонами",        5,  100,  30,   5,  key="ftr_dist")
    ftr_max_zones = st.slider("Макс. зон на графике",           3,   30,  10,   1,  key="ftr_maxz")
    st.caption("ATR↓ и Vol↓ = больше зон. Dist↓ = зоны ближе друг к другу.")

# Лабораторные вкладки имеют собственную маршрутизацию — пропускаем основную
_LAB_TABS = ("18. 🔵 Крипто Фьючерсы", "19. 🟡 Форекс & Металлы",
             "20. 🟢 Крипто Спот", "21. 📊 RS & Heatmap")


# ══════════════════════════════════════════════════════════════════════════════
#  SCREENER v2 (M1 25.05.2026) — Universal Chart View + Toolbox (9 toggles)
#  Используется в Tab Crypto / Forex / Renko. Каждый инструмент — independent
#  toggle для visual validation реализации.
# ══════════════════════════════════════════════════════════════════════════════

# ── Сигнатура каждого toggle (используется в 3 вкладках) ─────────────────────
_TOOLBOX_DEFAULT = {
    'hh_hl':    False,  # HH/HL/LL/LH swing structure
    'hvn':      False,  # High Volume Nodes
    'lvn':      False,  # Low Volume Nodes
    'vah':      False,  # Value Area High
    'val':      False,  # Value Area Low
    'poc':      False,  # Point of Control
    'cvd':      False,  # Cumulative Delta (subplot)
    'cvd_div':  False,  # CVD Divergences (Classical/Hidden/Absorption per research)
    'bos':      False,  # Break of Structure
    'fvg':      False,  # Fair Value Gap
    'balance':  False,  # Balance Zone (ML detector) — обучается через ML Zone Lab
    'vp_range': False,  # Fixed Range Volume Profile (box-select inside chart)
    'phase':    False,  # Two Market Phases — Balance/Imbalance per Trend Scanning (López de Prado)
    'rule80':   False,  # 80% Rule events (Auction Market Theory)
    'struct_bal': False,  # Structural Balance Zones (HH/HL/LL/LH cluster) — Wyckoff/SMC
}

_TOOLBOX_LABELS = {
    'hh_hl':    '📐 HH/HL/LL/LH (swing)',
    'hvn':      '🔵 HVN (high vol node)',
    'lvn':      '⚪ LVN (low vol node)',
    'vah':      '⬆️ VAH',
    'val':      '⬇️ VAL',
    'poc':      '🟡 POC',
    'cvd':      '📊 CVD (subplot)',
    'cvd_div':  '⚡ CVD Divergence',
    'bos':      '🔀 BOS',
    'fvg':      '🟦 FVG',
    'balance':  '🟪 Balance Zone (ML)',
    'vp_range': '📦 VP Range (Fixed)',
    'phase':    '🌊 Market Phase',
    'rule80':   '📐 80% Rule',
    'struct_bal': '🧱 Struct Balance',
}


def _render_toolbox(prefix: str) -> dict:
    """Toolbox панель: 11 checkbox'ов + All on/All off + параметры HH/HL и VP.

    Возвращает dict с toggles + опциональными ключами:
      'hh_hl_swing_len': int  — L (1=3-bar ICT, 2=5-bar Williams, 3=7-bar)
      'hh_hl_atr_mult':  float — фильтр незначимых свингов (0 = off)
    """
    with st.expander("🧰 **Toolbox** — наложение инструментов на график", expanded=True):
        bcols = st.columns([1, 1, 6])
        if bcols[0].button("✓ All on", key=f"{prefix}_tb_on", use_container_width=True):
            for k in _TOOLBOX_DEFAULT.keys():
                st.session_state[f"{prefix}_tg_{k}"] = True
            st.rerun()
        if bcols[1].button("✗ All off", key=f"{prefix}_tb_off", use_container_width=True):
            for k in _TOOLBOX_DEFAULT.keys():
                st.session_state[f"{prefix}_tg_{k}"] = False
            st.rerun()

        # 11 checkbox'ов в ряды по 3
        toggles = {}
        keys = list(_TOOLBOX_DEFAULT.keys())
        for row_start in range(0, len(keys), 3):
            cols = st.columns(3)
            for i, k in enumerate(keys[row_start:row_start + 3]):
                state_k = f"{prefix}_tg_{k}"
                _def = st.session_state.get(state_k, _TOOLBOX_DEFAULT[k])
                toggles[k] = cols[i].checkbox(
                    _TOOLBOX_LABELS[k], value=_def, key=state_k,
                )

        # ── Параметры HH/HL (активны только если toggle включён) ──────────────
        if toggles.get('hh_hl', False):
            st.markdown("---")
            st.caption(
                "📐 **HH/HL параметры** · Causal detection per research "
                "(N-bar fractal + State Machine + BOS/CHoCH by close)"
            )
            p_cols = st.columns([2, 2, 4])
            swing_choice = p_cols[0].selectbox(
                "Swing L",
                options=[1, 2, 3],
                index=[1, 2, 3].index(st.session_state.get(f"{prefix}_hh_L", 2)),
                format_func=lambda x: {
                    1: "L=1 (3-bar ICT)",
                    2: "L=2 (5-bar Williams)",
                    3: "L=3 (7-bar)",
                }[x],
                key=f"{prefix}_hh_L",
                help="Полу-ширина окна для fractal-детекции. Больше = чище, медленнее.",
            )
            atr_choice = p_cols[1].selectbox(
                "ATR фильтр",
                options=[0.0, 0.25, 0.5, 1.0],
                index=[0.0, 0.25, 0.5, 1.0].index(
                    st.session_state.get(f"{prefix}_hh_atr", 0.5)
                ),
                format_func=lambda x: "Off" if x == 0 else f"× {x} ATR200",
                key=f"{prefix}_hh_atr",
                help="Свинг отбрасывается, если движение < mult × ATR200.",
            )
            toggles['hh_hl_swing_len'] = int(swing_choice)
            toggles['hh_hl_atr_mult']  = float(atr_choice)
        else:
            toggles['hh_hl_swing_len'] = 2
            toggles['hh_hl_atr_mult']  = 0.5

        # ── Параметры VP (HVN/LVN/VAH/VAL/POC + VP Range) — research methodology ─
        vp_active = any(toggles.get(k, False)
                        for k in ('hvn', 'lvn', 'vah', 'val', 'poc', 'vp_range'))
        if vp_active:
            st.markdown("---")
            st.caption(
                "🔵⚪🟡 **Volume Profile параметры** · KDE + Silverman bandwidth + "
                "scipy.find_peaks (HVN height + prominence)"
            )
            v_cols = st.columns([2, 2, 2, 2])
            va_choice = v_cols[0].selectbox(
                "VA %",
                options=[0.6827, 0.70, 0.80],
                index=[0.6827, 0.70, 0.80].index(
                    st.session_state.get(f"{prefix}_vp_va", 0.70)
                ),
                format_func=lambda x: {0.6827: "68.27% (1σ)", 0.70: "70% (std)", 0.80: "80%"}[x],
                key=f"{prefix}_vp_va",
                help="Доля общего объёма для Value Area вокруг POC.",
            )
            hvn_h_choice = v_cols[1].selectbox(
                "HVN height γ",
                options=[0.4, 0.6, 0.8],
                index=[0.4, 0.6, 0.8].index(
                    st.session_state.get(f"{prefix}_vp_hvn_h", 0.6)
                ),
                format_func=lambda x: f"× {x} max(KDE)",
                key=f"{prefix}_vp_hvn_h",
                help="HVN должен иметь плотность ≥ γ × max. Research 0.8 для dominant zones.",
            )
            prom_choice = v_cols[2].selectbox(
                "Prominence γ",
                options=[0.2, 0.3, 0.5],
                index=[0.2, 0.3, 0.5].index(
                    st.session_state.get(f"{prefix}_vp_prom", 0.3)
                ),
                format_func=lambda x: f"× {x} range",
                key=f"{prefix}_vp_prom",
                help="Выразительность пика относительно размаха плотности.",
            )
            decay_choice = v_cols[3].selectbox(
                "Decay λ",
                options=[1.0, 0.99, 0.95],
                index=[1.0, 0.99, 0.95].index(
                    st.session_state.get(f"{prefix}_vp_decay", 1.0)
                ),
                format_func=lambda x: "Off" if x == 1.0 else f"λ = {x}",
                key=f"{prefix}_vp_decay",
                help="Экспоненциальное затухание весов старых баров. 1.0 = равные веса.",
            )
            toggles['vp_va_pct']     = float(va_choice)
            toggles['vp_hvn_height'] = float(hvn_h_choice)
            toggles['vp_prominence'] = float(prom_choice)
            toggles['vp_decay']      = float(decay_choice)
        else:
            toggles['vp_va_pct']     = 0.70
            toggles['vp_hvn_height'] = 0.6
            toggles['vp_prominence'] = 0.3
            toggles['vp_decay']      = 1.0

        # ── Параметры CVD Divergence — research methodology ──────────────────
        if toggles.get('cvd_div', False):
            st.markdown("---")
            st.caption(
                "⚡ **CVD Divergence параметры** · Causal N-bar fractal на price И на CVD, "
                "3 типа: Classical (exhaustion) / Hidden (continuation) / Absorption"
            )
            c_cols = st.columns([2, 2, 2, 2])
            cvd_swl = c_cols[0].selectbox(
                "Swing L",
                options=[1, 2, 3],
                index=[1, 2, 3].index(st.session_state.get(f"{prefix}_cvd_L", 2)),
                format_func=lambda x: {1: "L=1 (3-bar)", 2: "L=2 (5-bar)", 3: "L=3 (7-bar)"}[x],
                key=f"{prefix}_cvd_L",
                help="Полу-ширина окна fractal-детекции swing'ов (price + cvd).",
            )
            cvd_atr = c_cols[1].selectbox(
                "ATR фильтр",
                options=[0.0, 0.25, 0.5, 1.0],
                index=[0.0, 0.25, 0.5, 1.0].index(
                    st.session_state.get(f"{prefix}_cvd_atr", 0.5)
                ),
                format_func=lambda x: "Off" if x == 0 else f"× {x} ATR200",
                key=f"{prefix}_cvd_atr",
                help="Свинг цены отбрасывается, если |move| < mult × ATR200.",
            )
            cvd_zthr = c_cols[2].selectbox(
                "CVD z-min",
                options=[0.10, 0.30, 0.50, 1.0],
                index=[0.10, 0.30, 0.50, 1.0].index(
                    st.session_state.get(f"{prefix}_cvd_z", 0.30)
                ),
                format_func=lambda x: f"z ≥ {x}",
                key=f"{prefix}_cvd_z",
                help="Мин. z-score между CVD-свингами для classical/hidden. Шум-фильтр.",
            )
            cvd_abs_z = c_cols[3].selectbox(
                "Absorption z",
                options=[1.5, 2.0, 2.5, 3.0],
                index=[1.5, 2.0, 2.5, 3.0].index(
                    st.session_state.get(f"{prefix}_cvd_abs_z", 2.0)
                ),
                format_func=lambda x: f"|ΔCVD| z ≥ {x}",
                key=f"{prefix}_cvd_abs_z",
                help="Порог |Δcvd|/(σ·√W) для absorption (резкость CVD при flat-price).",
            )
            toggles['cvd_div_swing_len']     = int(cvd_swl)
            toggles['cvd_div_atr_mult']      = float(cvd_atr)
            toggles['cvd_div_z']             = float(cvd_zthr)
            toggles['cvd_div_absorption_z']  = float(cvd_abs_z)
        else:
            toggles['cvd_div_swing_len']     = 2
            toggles['cvd_div_atr_mult']      = 0.5
            toggles['cvd_div_z']             = 0.30
            toggles['cvd_div_absorption_z']  = 2.0

        # ── Параметры Market Phase — Trend Scanning (López de Prado) ─────────
        if toggles.get('phase', False):
            st.markdown("---")
            st.caption(
                "🌊 **Market Phase параметры** · Trend Scanning per López de Prado: "
                "argmax|t-stat| slope линейной регрессии log(price) на скользящих окнах"
            )
            ph_cols = st.columns([2, 2, 2, 2])
            ph_Lmin = ph_cols[0].selectbox(
                "L min",
                options=[10, 20, 30],
                index=[10, 20, 30].index(st.session_state.get(f"{prefix}_ph_Lmin", 20)),
                format_func=lambda x: f"L_min={x}",
                key=f"{prefix}_ph_Lmin",
                help="Мин. длина окна regression (бары).",
            )
            ph_Lmax = ph_cols[1].selectbox(
                "L max",
                options=[40, 60, 80, 100, 150],
                index=[40, 60, 80, 100, 150].index(st.session_state.get(f"{prefix}_ph_Lmax", 80)),
                format_func=lambda x: f"L_max={x}",
                key=f"{prefix}_ph_Lmax",
                help="Макс. длина окна regression. Больше = чувствительнее к долгим трендам.",
            )
            ph_thr = ph_cols[2].selectbox(
                "|t-stat| порог",
                options=[2.0, 2.5, 3.0, 4.0, 5.0],
                index=[2.0, 2.5, 3.0, 4.0, 5.0].index(
                    st.session_state.get(f"{prefix}_ph_thr", 3.0)
                ),
                format_func=lambda x: f"t ≥ {x}",
                key=f"{prefix}_ph_thr",
                help="|t-stat| > порог → IMBALANCE. 3.0 ≈ 99.7% CI.",
            )
            ph_minseg = ph_cols[3].selectbox(
                "min сегмент",
                options=[1, 3, 5, 10],
                index=[1, 3, 5, 10].index(st.session_state.get(f"{prefix}_ph_minseg", 5)),
                format_func=lambda x: f"{x} баров",
                key=f"{prefix}_ph_minseg",
                help="Короче этого — сольётся с предыдущим сегментом.",
            )
            toggles['phase_L_min']        = int(ph_Lmin)
            toggles['phase_L_max']        = int(ph_Lmax)
            toggles['phase_t_threshold']  = float(ph_thr)
            toggles['phase_min_segment']  = int(ph_minseg)
        else:
            toggles['phase_L_min']        = 20
            toggles['phase_L_max']        = 80
            toggles['phase_t_threshold']  = 3.0
            toggles['phase_min_segment']  = 5

        # ── Параметры 80% Rule (Auction Market Theory) ───────────────────────
        if toggles.get('rule80', False):
            st.markdown("---")
            st.caption(
                "📐 **80% Rule параметры** · цена выходит за VA → возвращается → удерживается "
                "→ 80% вероятность пересечения VA к противоположной границе"
            )
            r_cols = st.columns([2, 2, 2, 6])
            r_vaw = r_cols[0].selectbox(
                "VA окно",
                options=[30, 50, 80, 100],
                index=[30, 50, 80, 100].index(st.session_state.get(f"{prefix}_r80_vaw", 50)),
                format_func=lambda x: f"{x} баров",
                key=f"{prefix}_r80_vaw",
                help="Rolling-окно для расчёта Value Area.",
            )
            r_vapct = r_cols[1].selectbox(
                "VA %",
                options=[0.70, 0.75],
                index=[0.70, 0.75].index(st.session_state.get(f"{prefix}_r80_pct", 0.70)),
                format_func=lambda x: f"VA {int(x*100)}%",
                key=f"{prefix}_r80_pct",
                help="Доля общего объёма в Value Area.",
            )
            r_hold = r_cols[2].selectbox(
                "Hold баров",
                options=[1, 2, 3],
                index=[1, 2, 3].index(st.session_state.get(f"{prefix}_r80_hold", 2)),
                format_func=lambda x: f"{x} бар",
                key=f"{prefix}_r80_hold",
                help="Сколько баров подряд внутри VA после возврата.",
            )
            toggles['rule80_va_window'] = int(r_vaw)
            toggles['rule80_va_pct']    = float(r_vapct)
            toggles['rule80_hold']      = int(r_hold)
        else:
            toggles['rule80_va_window'] = 50
            toggles['rule80_va_pct']    = 0.70
            toggles['rule80_hold']      = 2

        # ── Параметры Structural Balance — на базе HH/HL swing кластеров ────
        if toggles.get('struct_bal', False):
            st.markdown("---")
            st.caption(
                "🧱 **Struct Balance параметры** · sliding-window K swings: highs/lows "
                "кластеризуются → горизонтальный ренж (Wyckoff/SMC)"
            )
            sb_cols = st.columns([2, 2, 2, 2])
            sb_minsw = sb_cols[0].selectbox(
                "Min swings",
                options=[3, 4, 5, 6],
                index=[3, 4, 5, 6].index(st.session_state.get(f"{prefix}_sb_minsw", 4)),
                format_func=lambda x: f"≥ {x}",
                key=f"{prefix}_sb_minsw",
                help="Минимум swings в окне для признания balance'ом.",
            )
            sb_maxatr = sb_cols[1].selectbox(
                "Range ≤ N×ATR",
                options=[2.0, 3.0, 4.0, 5.0],
                index=[2.0, 3.0, 4.0, 5.0].index(
                    st.session_state.get(f"{prefix}_sb_maxatr", 3.0)
                ),
                format_func=lambda x: f"≤ {x} ATR",
                key=f"{prefix}_sb_maxatr",
                help="Макс. ширина зоны в ATR.",
            )
            sb_cluster = sb_cols[2].selectbox(
                "Cluster spread",
                options=[0.30, 0.40, 0.50, 0.60],
                index=[0.30, 0.40, 0.50, 0.60].index(
                    st.session_state.get(f"{prefix}_sb_cluster", 0.40)
                ),
                format_func=lambda x: f"≤ {int(x*100)}%",
                key=f"{prefix}_sb_cluster",
                help="max(highs)-min(highs) ≤ frac × range (то же для lows).",
            )
            sb_swl = sb_cols[3].selectbox(
                "Swing L",
                options=[1, 2, 3],
                index=[1, 2, 3].index(st.session_state.get(f"{prefix}_sb_swl", 2)),
                format_func=lambda x: f"L={x}",
                key=f"{prefix}_sb_swl",
                help="Параметр причинной N-bar fractal детекции swings.",
            )
            toggles['struct_bal_min_swings']    = int(sb_minsw)
            toggles['struct_bal_max_range_atr'] = float(sb_maxatr)
            toggles['struct_bal_cluster_frac']  = float(sb_cluster)
            toggles['struct_bal_swing_len']     = int(sb_swl)
        else:
            toggles['struct_bal_min_swings']    = 4
            toggles['struct_bal_max_range_atr'] = 3.0
            toggles['struct_bal_cluster_frac']  = 0.40
            toggles['struct_bal_swing_len']     = 2

    return toggles


# ── Cached compute-functions per (sym, tf, n_bars) ────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def _sv2_load_ohlcv_crypto(sym: str, tf: str, n_bars: int) -> pd.DataFrame:
    """Online-loader: parquet (история) + Bybit V5 API (свежие бары) → splice.

    1. Грузим parquet через load_ohlcv_bars (история до ~даты последнего скачивания).
    2. Тянем 1000 свежих баров с Bybit /v5/market/kline.
    3. Склейка: parquet до min(API timestamp), API после.
    TTL=60 сек — балансируем актуальность vs нагрузку.
    """
    from balance_zone_markup import load_ohlcv_bars
    # 1. Parquet история
    df_hist = load_ohlcv_bars(sym, tf, n_bars=0)  # 0 = всё доступно
    if not df_hist.empty:
        df_hist = df_hist.reset_index(drop=True)
        df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'], utc=True)

    # 2. Свежие 1000 баров через Bybit
    bybit_tf_map = {'1m': '1', '3m': '3', '15m': '15', '1h': '60', '1D': 'D'}
    interval = bybit_tf_map.get(tf, '60')
    df_live = pd.DataFrame()
    try:
        res = st.session_state.session.get_kline(
            category="linear", symbol=sym, interval=interval, limit=1000,
        ).get('result', {}).get('list', [])
        if res:
            df_live = pd.DataFrame(res, columns=['ts', 'o', 'h', 'l', 'c', 'v', 't'])
            df_live['timestamp'] = pd.to_datetime(pd.to_numeric(df_live['ts']),
                                                   unit='ms', utc=True)
            df_live = df_live.sort_values('timestamp').reset_index(drop=True)
            for c, src in zip(['open', 'high', 'low', 'close', 'volume', 'turnover'],
                              ['o', 'h', 'l', 'c', 'v', 't']):
                df_live[c] = pd.to_numeric(df_live[src], errors='coerce')
            df_live = df_live[['timestamp', 'open', 'high', 'low', 'close',
                                'volume', 'turnover']]
    except Exception:
        df_live = pd.DataFrame()

    # 3. Splice
    if not df_hist.empty and not df_live.empty:
        api_start = df_live['timestamp'].min()
        df_old = df_hist[df_hist['timestamp'] < api_start]
        d = pd.concat([df_old, df_live], ignore_index=True)
    elif not df_live.empty:
        d = df_live
    elif not df_hist.empty:
        d = df_hist
    else:
        return pd.DataFrame()
    d = d.sort_values('timestamp').drop_duplicates(subset=['timestamp'],
                                                    keep='last').reset_index(drop=True)
    if n_bars > 0 and len(d) > n_bars:
        d = d.tail(n_bars).reset_index(drop=True)
    return d


@st.cache_data(ttl=60, show_spinner=False)
def _sv2_load_ohlcv_forex_online(sym: str, tf: str, n_bars: int) -> pd.DataFrame:
    """Online-loader форекс: parquet + yfinance splice.

    yfinance не умеет 3m → ресемплим из 1m. 1m только за 5 дней. 15m/1h/1D — напрямую.
    """
    import yfinance as yf
    from pathlib import Path as _P

    # 1. Parquet история
    forex_dir = _P(__file__).parent / 'backtest_data' / 'cache_forex'
    fname = sym[:3] + '_' + sym[3:]
    p = forex_dir / f'{fname}_{tf}.parquet'
    df_hist = pd.DataFrame()
    if p.exists():
        df_hist = pd.read_parquet(p)
        df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'], utc=True)
        df_hist = df_hist.sort_values('timestamp').reset_index(drop=True)

    # 2. yfinance свежие
    # Ticker для yfinance: EURUSD=X, USDJPY=X, XAUUSD=X
    yf_ticker = f"{sym}=X"
    df_live = pd.DataFrame()
    try:
        if tf == '3m':
            # ресемплим из 1m (yfinance не имеет 3m)
            raw = yf.download(yf_ticker, interval='1m', period='5d',
                              progress=False, auto_adjust=False)
            if not raw.empty:
                raw = raw.reset_index()
                raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
                raw = raw.rename(columns={'Datetime': 'timestamp', 'Date': 'timestamp',
                                            'Open': 'open', 'High': 'high', 'Low': 'low',
                                            'Close': 'close', 'Volume': 'volume'})
                raw['timestamp'] = pd.to_datetime(raw['timestamp'], utc=True)
                raw = (raw.set_index('timestamp').resample('3min')
                        .agg(open=('open', 'first'), high=('high', 'max'),
                             low=('low', 'min'), close=('close', 'last'),
                             volume=('volume', 'sum'))
                        .dropna(subset=['open']).reset_index())
                df_live = raw
        else:
            tf_y = {'15m': '15m', '1h': '1h', '1D': '1d'}
            period_map = {'15m': '60d', '1h': '730d', '1D': 'max'}
            raw = yf.download(yf_ticker, interval=tf_y.get(tf, '1h'),
                              period=period_map.get(tf, '60d'),
                              progress=False, auto_adjust=False)
            if not raw.empty:
                raw = raw.reset_index()
                raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
                raw = raw.rename(columns={'Datetime': 'timestamp', 'Date': 'timestamp',
                                            'Open': 'open', 'High': 'high', 'Low': 'low',
                                            'Close': 'close', 'Volume': 'volume'})
                raw['timestamp'] = pd.to_datetime(raw['timestamp'], utc=True)
                df_live = raw[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    except Exception:
        df_live = pd.DataFrame()

    # 3. Splice
    if not df_hist.empty and not df_live.empty:
        api_start = df_live['timestamp'].min()
        df_old = df_hist[df_hist['timestamp'] < api_start]
        d = pd.concat([df_old, df_live], ignore_index=True, sort=False)
    elif not df_live.empty:
        d = df_live
    elif not df_hist.empty:
        d = df_hist
    else:
        return pd.DataFrame()
    d = d.sort_values('timestamp').drop_duplicates(subset=['timestamp'],
                                                    keep='last').reset_index(drop=True)
    if n_bars > 0 and len(d) > n_bars:
        d = d.tail(n_bars).reset_index(drop=True)
    return d


@st.cache_data(ttl=600, show_spinner=False)
def _sv2_zigzag(_df_hash: int, df_pickle: bytes):
    """Hash через len+первый+последний timestamp — кеш по содержимому df."""
    import pickle
    df = pickle.loads(df_pickle)
    try:
        from smc_core import calc_zigzag
        return calc_zigzag(df, atr_mult=0.5, min_pct=0.003, max_pivots=200)
    except Exception:
        return []


@st.cache_data(ttl=600, show_spinner=False)
def _sv2_market_structure(_df_hash: int, df_pickle: bytes,
                          swing_len: int = 2, atr_mult: float = 0.5):
    """Causal HH/HL/LL/LH + BOS/CHoCH per research methodology.

    Returns: {'swings': [...], 'bos': [...], 'choch': [...], 'final_state': str}
    """
    import pickle
    df = pickle.loads(df_pickle)
    try:
        from smc_core import detect_market_structure_causal
        return detect_market_structure_causal(
            df,
            swing_len=int(swing_len),
            atr_period=200,
            atr_filter_mult=float(atr_mult),
            use_close_break=True,
        )
    except Exception:
        return {'swings': [], 'bos': [], 'choch': [], 'final_state': 'undef'}


@st.cache_data(ttl=600, show_spinner=False)
def _sv2_vp(_df_hash: int, df_pickle: bytes):
    """[LEGACY] Histogram VP — оставлен для обратной совместимости."""
    import pickle
    df = pickle.loads(df_pickle)
    try:
        from balance_zone_markup import compute_volume_profile
        return compute_volume_profile(df, rows=80, va_pct=0.70)
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def _sv2_vp_research(_df_hash: int, df_pickle: bytes,
                     va_pct: float = 0.70,
                     hvn_height: float = 0.6,
                     prominence: float = 0.3,
                     decay: float = 1.0):
    """KDE-based VP per research (HVN LVN.txt + Volume Profile txt).

    Returns dict with: poc, vah, val, va_width, hvn_levels, lvn_levels,
                       hvn_widths, lvn_widths, kde_x, kde_density, bandwidth.
    """
    import pickle
    df = pickle.loads(df_pickle)
    try:
        from smc_core import compute_volume_profile_research
        return compute_volume_profile_research(
            df,
            va_pct=float(va_pct),
            hvn_height=float(hvn_height),
            hvn_prominence=float(prominence),
            lvn_prominence=float(prominence),
            decay=float(decay),
            n_points=256,
            atr_width_mult=0.0,
        )
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def _sv2_bos_ob_fvg(_df_hash: int, df_pickle: bytes):
    import pickle
    df = pickle.loads(df_pickle)
    try:
        from smc_core import find_bos_ob_fvg
        return find_bos_ob_fvg(df, sw=5, fvg_mult=0.5)
    except Exception:
        return [], [], []


@st.cache_data(ttl=600, show_spinner=False)
def _sv2_cvd(_df_hash: int, df_pickle: bytes):
    import pickle
    df = pickle.loads(df_pickle)
    try:
        from smc_core import apply_order_flow
        d = apply_order_flow(df)
        return d['cvd'].to_numpy() if 'cvd' in d.columns else None
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def _sv2_cvd_full(_df_hash: int, df_pickle: bytes):
    """Полный CVD-пакет: cvd, delta, imbalance_ratio (NOFI proxy) — для subplot."""
    import pickle
    df = pickle.loads(df_pickle)
    try:
        from smc_core import apply_order_flow
        d = apply_order_flow(df)
        cols = ['cvd', 'delta']
        out = {c: d[c].to_numpy() for c in cols if c in d.columns}
        if 'imbalance_ratio' in d.columns:
            sign = np.sign(d['delta'].to_numpy())
            out['nofi'] = (d['imbalance_ratio'].to_numpy() * sign)
        return out
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def _sv2_cvd_divergences(_df_hash: int, df_pickle: bytes,
                          swing_len: int = 2,
                          min_dist: int = 8,
                          atr_filter_mult: float = 0.5,
                          cvd_z_threshold: float = 0.30,
                          absorption_window: int = 20,
                          absorption_cvd_z: float = 2.0,
                          absorption_price_z: float = 0.5):
    """CVD-дивергенции (Classical/Hidden/Absorption) per CVD.txt research."""
    import pickle
    df = pickle.loads(df_pickle)
    try:
        from smc_core import detect_cvd_divergences_research
        return detect_cvd_divergences_research(
            df,
            swing_len=int(swing_len),
            min_dist=int(min_dist),
            atr_filter_mult=float(atr_filter_mult),
            cvd_z_threshold=float(cvd_z_threshold),
            absorption_window=int(absorption_window),
            absorption_cvd_z=float(absorption_cvd_z),
            absorption_price_z=float(absorption_price_z),
        )
    except Exception:
        return {'events': [], 'price_swings': [], 'cvd_swings': []}


@st.cache_data(ttl=600, show_spinner=False)
def _sv2_market_phases(_df_hash: int, df_pickle: bytes,
                       L_min: int = 20, L_max: int = 80, L_step: int = 10,
                       t_stat_threshold: float = 3.0,
                       min_segment_bars: int = 5):
    """Trend Scanning per López de Prado — Balance/Imbalance детекция."""
    import pickle
    df = pickle.loads(df_pickle)
    try:
        from smc_core import detect_market_phases_research
        return detect_market_phases_research(
            df,
            L_min=int(L_min), L_max=int(L_max), L_step=int(L_step),
            t_stat_threshold=float(t_stat_threshold),
            min_segment_bars=int(min_segment_bars),
        )
    except Exception:
        return {'phase': None, 'best_t_stat': None, 'best_L': None, 'segments': []}


@st.cache_data(ttl=600, show_spinner=False)
def _sv2_80pct_rule(_df_hash: int, df_pickle: bytes,
                     va_window: int = 50, va_pct: float = 0.70, hold_bars: int = 2):
    """80% Rule — Auction Market Theory."""
    import pickle
    df = pickle.loads(df_pickle)
    try:
        from smc_core import detect_80pct_rule_events
        return detect_80pct_rule_events(
            df,
            va_window=int(va_window), va_pct=float(va_pct), hold_bars=int(hold_bars),
        )
    except Exception:
        return []


@st.cache_data(ttl=600, show_spinner=False)
def _sv2_struct_balance(_df_hash: int, df_pickle: bytes,
                         swing_len: int = 2, min_swings: int = 4,
                         max_range_atr: float = 3.0,
                         cluster_spread_frac: float = 0.40):
    """Structural Balance Zones через HH/HL swing-кластеры."""
    import pickle
    df = pickle.loads(df_pickle)
    try:
        from smc_core import detect_structural_balance_zones
        return detect_structural_balance_zones(
            df, swing_len=int(swing_len), atr_filter_mult=0.5,
            min_swings=int(min_swings), max_swings=10,
            max_range_atr=float(max_range_atr),
            cluster_spread_frac=float(cluster_spread_frac),
        )
    except Exception:
        return []


@st.cache_data(ttl=600, show_spinner="Сканирую Balance Zones (ML)…")
def _sv2_balance_zones(_df_hash: int, df_pickle: bytes, tf: str, symbol: str):
    """ML zone detector — обучается через ML Zone Lab.
    Возвращает DataFrame с балансовыми зонами или None если model нет / tf не поддерживается.
    """
    import pickle
    df = pickle.loads(df_pickle)
    try:
        from detect_balance_zones import detect_balance_zones_ml, ML_MODEL_PATH
        if not ML_MODEL_PATH.exists():
            return None
        # Модель тренирована на 15m/1h/1D — другие TF (Renko) skip
        if tf not in ('15m', '1h', '1D'):
            return None
        return detect_balance_zones_ml(df, tf=tf, symbol=symbol)
    except Exception:
        return None


def _sv2_df_hash(df: pd.DataFrame) -> tuple:
    """Хэш df для cache key: длина + первый + последний timestamp."""
    if df.empty:
        return (0, 0, 0)
    return (
        len(df),
        int(pd.to_datetime(df['timestamp'].iloc[0]).value),
        int(pd.to_datetime(df['timestamp'].iloc[-1]).value),
    )


def _render_chart_with_tools(df: pd.DataFrame, symbol: str, tf: str,
                              toggles: dict, chart_key: str,
                              is_renko: bool = False,
                              strategy_signals: list = None) -> None:
    """Главная функция: рендерит свечной chart + наложения по toggles.

    Каждый инструмент включается ТОЛЬКО при toggles[key]=True (для validation).
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pickle

    use_cvd_subplot = toggles.get('cvd', False)

    if use_cvd_subplot:
        # CVD subplot имеет secondary_y: основная ось — delta bars (signed volume),
        # вторичная — накопительная CVD-линия и NOFI ribbon.
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.03, row_heights=[0.78, 0.22],
            subplot_titles=('', 'Δ · CVD · NOFI'),
            specs=[[{}], [{"secondary_y": True}]],
        )
        candle_row = 1
        cvd_row = 2
    else:
        fig = make_subplots(rows=1, cols=1)
        candle_row = 1

    # ── Свечи ────────────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df['timestamp'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        increasing_line_color='#00E676', decreasing_line_color='#FF1744',
        name='OHLC', showlegend=False,
    ), row=candle_row, col=1)

    # ── Cached hash для compute-functions ────────────────────────────────────
    _h = _sv2_df_hash(df)
    _pkl = pickle.dumps(df)

    # ── VP Range (Fixed Range VP per TradingView) — box-select inside chart ─
    # Если 📦 VP Range активен и пользователь выделил прямоугольник на оси X —
    # VP считается ТОЛЬКО на барах внутри выделения. Иначе — всё окно (default).
    vp_range_active = toggles.get('vp_range', False)
    vp_box_key      = f"vp_box_{chart_key}"
    _selection_ts   = st.session_state.get(vp_box_key)   # tuple (ts0, ts1) | None

    _vp_in_range = False
    if vp_range_active and _selection_ts is not None:
        try:
            _ts0, _ts1 = _selection_ts
            _mask = (df['timestamp'] >= _ts0) & (df['timestamp'] <= _ts1)
            _df_for_vp = df[_mask].reset_index(drop=True)
            if len(_df_for_vp) >= 5:
                vp_x0 = _df_for_vp['timestamp'].iloc[0]
                vp_x1 = _df_for_vp['timestamp'].iloc[-1]
                _vp_in_range = True
            else:
                _df_for_vp = df
                vp_x0 = df['timestamp'].iloc[0]
                vp_x1 = df['timestamp'].iloc[-1]
        except Exception:
            _df_for_vp = df
            vp_x0 = df['timestamp'].iloc[0]
            vp_x1 = df['timestamp'].iloc[-1]
    else:
        _df_for_vp = df
        vp_x0 = df['timestamp'].iloc[0]
        vp_x1 = df['timestamp'].iloc[-1]

    _h_vp   = _sv2_df_hash(_df_for_vp)
    _pkl_vp = pickle.dumps(_df_for_vp) if _vp_in_range else _pkl

    # ── 1. HH/HL/LL/LH + BOS/CHoCH (causal, per research) ───────────────────
    if toggles.get('hh_hl', False):
        ms = _sv2_market_structure(
            hash(_h), _pkl,
            swing_len=toggles.get('hh_hl_swing_len', 2),
            atr_mult=toggles.get('hh_hl_atr_mult', 0.5),
        )
        swings = ms.get('swings', [])
        bos_events = ms.get('bos', [])
        choch_events = ms.get('choch', [])

        # ── Свинги: HH (золотой) / LH (серый) / HL (зелёный) / LL (красный) ─
        for label, color, sym, pos in [
            ('HH', '#FFD700', 'triangle-down', 'top center'),
            ('LH', '#888888', 'triangle-down', 'top center'),
            ('HL', '#00E676', 'triangle-up',   'bottom center'),
            ('LL', '#FF1744', 'triangle-up',   'bottom center'),
        ]:
            pts = [s for s in swings if s.get('label') == label]
            if not pts:
                continue
            fig.add_trace(go.Scatter(
                x=[p['timestamp'] for p in pts],
                y=[p['price'] for p in pts],
                mode='markers+text',
                marker=dict(symbol=sym, size=11, color=color),
                text=[label] * len(pts), textposition=pos,
                textfont=dict(size=9, color=color),
                name=label, hoverinfo='text+x+y', showlegend=False,
            ), row=candle_row, col=1)

        # ── BOS: пунктирная линия от свинга до бара пробоя + подпись ────────
        for evt in bos_events:
            color = '#00C853' if evt['direction'] == 'up' else '#D50000'
            fig.add_trace(go.Scatter(
                x=[evt['broken_swing_ts'], evt['timestamp']],
                y=[evt['price'], evt['price']],
                mode='lines',
                line=dict(color=color, width=1.4, dash='dash'),
                name=f"BOS {evt['direction']}",
                hovertemplate=f"BOS {evt['direction']} @ {evt['price']:.4f}<extra></extra>",
                showlegend=False,
            ), row=candle_row, col=1)
            fig.add_annotation(
                x=evt['timestamp'], y=evt['price'],
                text=f"BOS",
                showarrow=False, xanchor='left', yanchor='middle',
                xshift=4, font=dict(size=9, color=color),
                row=candle_row, col=1,
            )

        # ── CHoCH: жирная линия + явная подпись (слом тренда) ───────────────
        for evt in choch_events:
            color = '#FF6D00' if evt['direction'] == 'up' else '#AA00FF'
            fig.add_trace(go.Scatter(
                x=[evt['broken_swing_ts'], evt['timestamp']],
                y=[evt['price'], evt['price']],
                mode='lines',
                line=dict(color=color, width=2.2, dash='solid'),
                name=f"CHoCH {evt['direction']}",
                hovertemplate=f"CHoCH {evt['direction']} @ {evt['price']:.4f}<extra></extra>",
                showlegend=False,
            ), row=candle_row, col=1)
            fig.add_annotation(
                x=evt['timestamp'], y=evt['price'],
                text=f"<b>CHoCH</b>",
                showarrow=False, xanchor='left', yanchor='middle',
                xshift=4, font=dict(size=10, color=color),
                row=candle_row, col=1,
            )

    # ── 2/3/4/5/6. KDE-based Volume Profile (research methodology) ──────────
    # HVN/LVN — полосы шириной FWHM; VAH/VAL/POC — линии.
    # Когда 📦 VP Range активен И есть выделение — рисуем ВСЕ 5 компонентов
    # (POC + VAH + VAL + HVN + LVN) внутри [vp_x0, vp_x1], как TradingView
    # Fixed Range VP. Индивидуальные toggle'ы при этом игнорируются.
    _force_show_vp = vp_range_active and _vp_in_range
    need_vp = _force_show_vp or any(
        toggles.get(k, False) for k in ('hvn', 'lvn', 'vah', 'val', 'poc')
    )
    if need_vp:
        vp = _sv2_vp_research(
            hash(_h_vp), _pkl_vp,
            va_pct=toggles.get('vp_va_pct', 0.70),
            hvn_height=toggles.get('vp_hvn_height', 0.6),
            prominence=toggles.get('vp_prominence', 0.3),
            decay=toggles.get('vp_decay', 1.0),
        )
        if vp:
            # ── VP Range: force-detect HVN/LVN без порогов ────────────────────
            # В режиме 📦 VP Range пользователь хочет видеть HVN/LVN всегда,
            # независимо от prominence/height. Берём ВСЕ локальные max/min
            # плотности внутри [pl, ph]; fallback на глобальный argmax/argmin
            # если плотность монотонна. FWHM-ширина считается напрямую.
            if _force_show_vp:
                _xg_all  = np.asarray(vp.get('kde_x',       []), dtype=float)
                _den_all = np.asarray(vp.get('kde_density', []), dtype=float)
                _pl_r = float(vp.get('range_low',  _xg_all.min() if _xg_all.size else 0.0))
                _ph_r = float(vp.get('range_high', _xg_all.max() if _xg_all.size else 1.0))
                if _xg_all.size >= 3 and _den_all.size == _xg_all.size:
                    _mask_r = (_xg_all >= _pl_r) & (_xg_all <= _ph_r)
                    if _mask_r.sum() >= 3:
                        _xg_r  = _xg_all[_mask_r]
                        _den_r = _den_all[_mask_r]
                        _n_r   = len(_xg_r)

                        # HVN: все локальные max плотности (без порогов)
                        _dl = _den_r[1:-1] > _den_r[:-2]
                        _dr = _den_r[1:-1] > _den_r[2:]
                        _hvn_idx = (np.where(_dl & _dr)[0] + 1).tolist()
                        if not _hvn_idx:
                            _hvn_idx = [int(np.argmax(_den_r))]

                        # LVN: все локальные min плотности
                        _il = _den_r[1:-1] < _den_r[:-2]
                        _ir = _den_r[1:-1] < _den_r[2:]
                        _lvn_idx = (np.where(_il & _ir)[0] + 1).tolist()
                        if not _lvn_idx:
                            _lvn_idx = [int(np.argmin(_den_r))]

                        # Ограниченная ширина: half-distance до ближайшего соседнего
                        # экстремума (пик ИЛИ впадина) с каждой стороны. Гарантирует
                        # что HVN/LVN полосы не перекрываются между собой.
                        _all_ext = sorted(set(_hvn_idx + _lvn_idx + [0, _n_r - 1]))

                        def _bounded_widths(_idxs):
                            _ws = []
                            for _idx in _idxs:
                                _left  = 0
                                _right = _n_r - 1
                                for _e in _all_ext:
                                    if _e < _idx:
                                        _left = _e
                                    elif _e > _idx:
                                        _right = _e
                                        break
                                _hl = (float(_xg_r[_idx]) - float(_xg_r[_left]))  / 2.0
                                _hr = (float(_xg_r[_right]) - float(_xg_r[_idx])) / 2.0
                                _half = min(_hl, _hr)
                                if _half <= 0:
                                    _half = (float(_xg_r[_right]) - float(_xg_r[_left])) / 4.0
                                _ws.append(float(_half * 2.0))
                            return _ws

                        vp['hvn_levels'] = [float(_xg_r[i]) for i in _hvn_idx]
                        vp['hvn_widths'] = _bounded_widths(_hvn_idx)
                        vp['lvn_levels'] = [float(_xg_r[i]) for i in _lvn_idx]
                        vp['lvn_widths'] = _bounded_widths(_lvn_idx)

            # ── VP Range: полупрозрачная заливка выделенной зоны ─────────────
            if _vp_in_range:
                fig.add_vrect(
                    x0=vp_x0, x1=vp_x1,
                    fillcolor='rgba(180, 100, 255, 0.05)',
                    line=dict(color='rgba(180, 100, 255, 0.50)', width=1, dash='dash'),
                    layer='below',
                    row=candle_row, col=1,
                )

            # ── KDE-силуэт распределения объёмов (Fixed Range VP, TradingView-style) ─
            # Рисуем ТОЛЬКО в режиме 📦 VP Range (когда выделен прямоугольник).
            # Силуэт занимает ~30% ширины бокса от vp_x0. VA — зелёная заливка,
            # POC-bin — жёлтая, остальное — синяя.
            kde_x   = vp.get('kde_x')
            kde_den = vp.get('kde_density')
            if _force_show_vp and kde_x is not None and kde_den is not None and len(kde_x) > 2:
                try:
                    # Ширина силуэта = 30% длины бокса по времени
                    _x0_ts = pd.Timestamp(vp_x0)
                    _x1_ts = pd.Timestamp(vp_x1)
                    _span_s = max((_x1_ts - _x0_ts).total_seconds(), 1.0)
                    _max_w_s = _span_s * 0.30

                    _den = np.asarray(kde_den, dtype=float)
                    _xg  = np.asarray(kde_x,   dtype=float)
                    _den_max = float(_den.max()) if _den.size else 0.0
                    if _den_max > 0:
                        _den_norm = _den / _den_max
                        # Силуэт: точки (x0+w(y), y), замкнутый назад к x0 → fill=toself
                        _xs_right = [_x0_ts + pd.Timedelta(seconds=float(d) * _max_w_s)
                                     for d in _den_norm]
                        _xs = _xs_right + [_x0_ts, _x0_ts]
                        _ys = list(_xg) + [float(_xg[-1]), float(_xg[0])]
                        fig.add_trace(go.Scatter(
                            x=_xs, y=_ys,
                            mode='lines',
                            fill='toself',
                            fillcolor='rgba(100,160,255,0.22)',
                            line=dict(color='rgba(100,160,255,0.65)', width=1),
                            hoverinfo='skip', showlegend=False,
                            name='VP',
                        ), row=candle_row, col=1)

                        # Подсветка Value Area: зелёная заливка между VAL и VAH
                        _val_y = float(vp.get('val', _xg[0]))
                        _vah_y = float(vp.get('vah', _xg[-1]))
                        _va_mask = (_xg >= _val_y) & (_xg <= _vah_y)
                        if _va_mask.any():
                            _va_x = [_x0_ts + pd.Timedelta(seconds=float(d) * _max_w_s)
                                     for d in _den_norm[_va_mask]]
                            _va_y = list(_xg[_va_mask])
                            _va_xs = _va_x + [_x0_ts, _x0_ts]
                            _va_ys = _va_y + [_va_y[-1], _va_y[0]]
                            fig.add_trace(go.Scatter(
                                x=_va_xs, y=_va_ys,
                                mode='lines',
                                fill='toself',
                                fillcolor='rgba(0,200,100,0.28)',
                                line=dict(color='rgba(0,200,100,0.55)', width=1),
                                hoverinfo='skip', showlegend=False,
                                name='VA',
                            ), row=candle_row, col=1)

                        # POC-bin: жёлтая полоска
                        _poc_y = float(vp.get('poc', _xg[int(np.argmax(_den))]))
                        _poc_i = int(np.argmin(np.abs(_xg - _poc_y)))
                        _poc_w = _xs_right[_poc_i]
                        fig.add_shape(
                            type='line',
                            x0=_x0_ts, x1=_poc_w,
                            y0=_poc_y, y1=_poc_y,
                            line=dict(color='rgba(255,220,0,0.95)', width=3),
                            row=candle_row, col=1,
                        )
                except Exception:
                    pass

            # ── HVN: полосы (жёлтый tint), ограничены [vp_x0, vp_x1] ────────
            if toggles.get('hvn', False) or _force_show_vp:
                hvn_lvls  = vp.get('hvn_levels', [])
                hvn_wdths = vp.get('hvn_widths', [])
                for i, lvl in enumerate(hvn_lvls):
                    w = hvn_wdths[i] if i < len(hvn_wdths) else 0.0
                    if w > 0:
                        fig.add_shape(
                            type='rect',
                            x0=vp_x0, x1=vp_x1,
                            y0=lvl - w / 2.0, y1=lvl + w / 2.0,
                            fillcolor='rgba(255, 215, 0, 0.16)',
                            line=dict(color='rgba(255, 215, 0, 0.70)', width=1, dash='dot'),
                            layer='below',
                            row=candle_row, col=1,
                        )
                    fig.add_annotation(
                        x=vp_x1, y=lvl, text=f"HVN {lvl:.4f}",
                        showarrow=False, xanchor='left', xshift=4,
                        font=dict(size=9, color='#FFD700'),
                        row=candle_row, col=1,
                    )

            # ── LVN: полосы (белый tint) ────────────────────────────────────
            if toggles.get('lvn', False) or _force_show_vp:
                lvn_lvls  = vp.get('lvn_levels', [])
                lvn_wdths = vp.get('lvn_widths', [])
                for i, lvl in enumerate(lvn_lvls):
                    w = lvn_wdths[i] if i < len(lvn_wdths) else 0.0
                    if w > 0:
                        fig.add_shape(
                            type='rect',
                            x0=vp_x0, x1=vp_x1,
                            y0=lvl - w / 2.0, y1=lvl + w / 2.0,
                            fillcolor='rgba(255, 255, 255, 0.14)',
                            line=dict(color='rgba(255, 255, 255, 0.70)', width=1, dash='dot'),
                            layer='below',
                            row=candle_row, col=1,
                        )
                    fig.add_annotation(
                        x=vp_x1, y=lvl, text=f"LVN {lvl:.4f}",
                        showarrow=False, xanchor='left', xshift=4,
                        font=dict(size=9, color='#FFFFFF'),
                        row=candle_row, col=1,
                    )

            # ── VAH / VAL: dashed линии (только в выделенном диапазоне) ─────
            if toggles.get('vah', False) or _force_show_vp:
                fig.add_trace(go.Scatter(
                    x=[vp_x0, vp_x1], y=[vp['vah'], vp['vah']],
                    mode='lines', line=dict(color='rgba(255,150,150,0.85)', width=1.5, dash='dash'),
                    name=f"VAH {vp['vah']:.4f}", hoverinfo='name+y', showlegend=False,
                ), row=candle_row, col=1)
                fig.add_annotation(x=vp_x1, y=vp['vah'], text=f"VAH {vp['vah']:.4f}",
                                    showarrow=False, xanchor='left', xshift=4,
                                    font=dict(size=10, color='#FF9696'),
                                    row=candle_row, col=1)
            if toggles.get('val', False) or _force_show_vp:
                fig.add_trace(go.Scatter(
                    x=[vp_x0, vp_x1], y=[vp['val'], vp['val']],
                    mode='lines', line=dict(color='rgba(150,255,150,0.85)', width=1.5, dash='dash'),
                    name=f"VAL {vp['val']:.4f}", hoverinfo='name+y', showlegend=False,
                ), row=candle_row, col=1)
                fig.add_annotation(x=vp_x1, y=vp['val'], text=f"VAL {vp['val']:.4f}",
                                    showarrow=False, xanchor='left', xshift=4,
                                    font=dict(size=10, color='#96FF96'),
                                    row=candle_row, col=1)

            # ── POC: жирная линия (argmax KDE) ──────────────────────────────
            if toggles.get('poc', False) or _force_show_vp:
                fig.add_trace(go.Scatter(
                    x=[vp_x0, vp_x1], y=[vp['poc'], vp['poc']],
                    mode='lines', line=dict(color='rgba(255,220,0,0.95)', width=2),
                    name=f"POC {vp['poc']:.4f}", hoverinfo='name+y', showlegend=False,
                ), row=candle_row, col=1)
                fig.add_annotation(x=vp_x1, y=vp['poc'], text=f"POC {vp['poc']:.4f}",
                                    showarrow=False, xanchor='left', xshift=4,
                                    font=dict(size=10, color='#FFDC00'),
                                    row=candle_row, col=1)

    # ── 8. BOS + 9. FVG (общая функция find_bos_ob_fvg) ─────────────────────
    if toggles.get('bos', False) or toggles.get('fvg', False):
        bos_list, ob_list, fvg_list = _sv2_bos_ob_fvg(hash(_h), _pkl)

        if toggles.get('bos', False):
            # BOS — горизонтальные линии от x0 до x1 на уровне y
            for b in bos_list:
                _color = '#00E676' if b.get('dir') == 'up' else '#FF1744'
                _dash = 'solid' if b.get('strong') else 'dot'
                _txt = ('CHoCH' if b.get('choch') else 'BOS') + ('↑' if b['dir'] == 'up' else '↓')
                fig.add_trace(go.Scatter(
                    x=[b['x0'], b['x1']], y=[b['y'], b['y']],
                    mode='lines', line=dict(color=_color, width=2, dash=_dash),
                    hoverinfo='name', name=_txt, showlegend=False,
                ), row=candle_row, col=1)
                fig.add_annotation(x=b['x1'], y=b['y'], text=_txt,
                                    showarrow=False, xanchor='left', xshift=4,
                                    font=dict(size=9, color=_color),
                                    row=candle_row, col=1)

        if toggles.get('fvg', False):
            # FVG — цветные прямоугольники
            for fv in fvg_list:
                _color = 'rgba(0,230,118,0.18)' if fv['dir'] == 'up' else 'rgba(255,23,68,0.18)'
                _line = 'rgba(0,230,118,0.6)' if fv['dir'] == 'up' else 'rgba(255,23,68,0.6)'
                fig.add_shape(
                    type='rect',
                    x0=fv['x0'], x1=fv['x1'], y0=fv['y0'], y1=fv['y1'],
                    line=dict(color=_line, width=1),
                    fillcolor=_color, layer='above',
                    row=candle_row, col=1,
                )

    # ── 10. Balance Zone (ML detector) — фиолетовые rect'ы ─────────────────
    if toggles.get('balance', False):
        bz_df = _sv2_balance_zones(hash(_h), _pkl, tf, symbol)
        if bz_df is None:
            # tf не поддерживается (Renko) или модели нет — показать caption ниже
            pass
        elif not len(bz_df):
            pass
        else:
            xs_bz, ys_bz = [], []
            for _, z in bz_df.iterrows():
                xs_bz.extend([z['start_ts'], z['end_ts'], z['end_ts'],
                              z['start_ts'], z['start_ts'], None])
                ys_bz.extend([z['range_low'], z['range_low'], z['range_high'],
                              z['range_high'], z['range_low'], None])
            fig.add_trace(go.Scatter(
                x=xs_bz, y=ys_bz, mode='lines',
                line=dict(color='rgba(180,100,255,0.85)', width=1.5),
                fill='toself', fillcolor='rgba(180,100,255,0.08)',
                hoverinfo='skip', showlegend=False, name='balance_zone',
            ), row=candle_row, col=1)
            # POC внутри каждой зоны (тонкая жёлтая)
            for _, z in bz_df.iterrows():
                fig.add_trace(go.Scatter(
                    x=[z['start_ts'], z['end_ts']],
                    y=[z['poc'], z['poc']],
                    mode='lines',
                    line=dict(color='rgba(255,220,0,0.6)', width=1, dash='dot'),
                    hoverinfo='skip', showlegend=False,
                ), row=candle_row, col=1)

    # ── 7. CVD subplot: Delta bars + CVD line + NOFI per CVD.txt research ──
    # Delta candles (D_open=0, D_close=delta, sign-colored bars) на основной оси,
    # накопительная CVD-линия + NOFI ribbon на secondary_y.
    if use_cvd_subplot:
        cvd_pkg = _sv2_cvd_full(hash(_h), _pkl)
        if cvd_pkg is not None and 'cvd' in cvd_pkg:
            cvd_arr   = cvd_pkg['cvd']
            delta_arr = cvd_pkg.get('delta')
            nofi_arr  = cvd_pkg.get('nofi')

            # Delta histogram per bar (D_close): зелёный если +, красный если −.
            if delta_arr is not None:
                _colors = ['rgba(0,230,118,0.55)' if d >= 0 else 'rgba(255,23,68,0.55)'
                           for d in delta_arr]
                fig.add_trace(go.Bar(
                    x=df['timestamp'], y=delta_arr,
                    marker=dict(color=_colors, line=dict(width=0)),
                    name='Δ', hoverinfo='y+name', showlegend=False,
                    opacity=0.85,
                ), row=cvd_row, col=1, secondary_y=False)

            # CVD-линия (накопительная) на secondary_y
            fig.add_trace(go.Scatter(
                x=df['timestamp'], y=cvd_arr,
                mode='lines', line=dict(color='#7DD3FC', width=1.8),
                name='CVD', hoverinfo='y+name', showlegend=False,
            ), row=cvd_row, col=1, secondary_y=True)

            # NOFI на secondary_y (тонкая жёлтая dotted)
            if nofi_arr is not None:
                # NOFI ∈ [-1, +1] — масштабируем к диапазону cvd для отображения
                _cmax = np.nanmax(np.abs(cvd_arr)) if len(cvd_arr) else 1.0
                fig.add_trace(go.Scatter(
                    x=df['timestamp'], y=nofi_arr * _cmax * 0.5,
                    mode='lines', line=dict(color='rgba(255,193,7,0.55)', width=1, dash='dot'),
                    name='NOFI', hoverinfo='y+name', showlegend=False,
                ), row=cvd_row, col=1, secondary_y=True)

            fig.add_hline(y=0, line=dict(color='rgba(200,200,200,0.3)', width=1, dash='dot'),
                           row=cvd_row, col=1)

    # ── 11. CVD Divergences (Classical / Hidden / Absorption) per research ─
    # Соединительные линии: pivot price[i1] → price[i2] на главном чарте +
    # (если subplot активен) cvd[i1] → cvd[i2] на CVD subplot.
    # Стиль:  classical=solid, hidden=dash, absorption=dot
    # Цвет:   bull=#00C853 (зелёный), bear=#D50000 (красный)
    if toggles.get('cvd_div', False):
        cvd_divs = _sv2_cvd_divergences(
            hash(_h), _pkl,
            swing_len=toggles.get('cvd_div_swing_len', 2),
            atr_filter_mult=toggles.get('cvd_div_atr_mult', 0.5),
            cvd_z_threshold=toggles.get('cvd_div_z', 0.30),
            absorption_cvd_z=toggles.get('cvd_div_absorption_z', 2.0),
        )
        _style = {'classical': 'solid', 'hidden': 'dash', 'absorption': 'dot'}
        _color = {'bull': '#00C853', 'bear': '#D50000'}
        _label = {'classical': 'CLS', 'hidden': 'HID', 'absorption': 'ABS'}
        for ev in cvd_divs.get('events', []):
            k, d = ev['kind'], ev['direction']
            col  = _color[d]
            dash = _style[k]
            arrow = '↑' if d == 'bull' else '↓'
            # Price-сторона: главный чарт
            fig.add_trace(go.Scatter(
                x=[ev['ts0'], ev['ts1']], y=[ev['p0'], ev['p1']],
                mode='lines+markers',
                line=dict(color=col, width=2, dash=dash),
                marker=dict(size=8, color=col, symbol='circle',
                             line=dict(color='white', width=1)),
                name=f"{_label[k]} {d} {arrow}",
                hovertemplate=(f"<b>CVD Div {k.upper()} {d.upper()}</b><br>"
                               f"p0={ev['p0']:.4f}<br>p1={ev['p1']:.4f}<br>"
                               f"c0={ev['c0']:.2f}<br>c1={ev['c1']:.2f}<extra></extra>"),
                showlegend=False,
            ), row=candle_row, col=1)
            fig.add_annotation(
                x=ev['ts1'], y=ev['p1'],
                text=f"<b>{_label[k]}{arrow}</b>",
                showarrow=False, xanchor='left', xshift=6, yshift=8,
                font=dict(size=10, color=col),
                bgcolor='rgba(0,0,0,0.5)',
                row=candle_row, col=1,
            )
            # CVD-сторона: subplot (если активен) на secondary_y
            if use_cvd_subplot:
                fig.add_trace(go.Scatter(
                    x=[ev['ts0'], ev['ts1']], y=[ev['c0'], ev['c1']],
                    mode='lines+markers',
                    line=dict(color=col, width=2, dash=dash),
                    marker=dict(size=6, color=col, symbol='diamond',
                                 line=dict(color='white', width=1)),
                    showlegend=False, hoverinfo='skip',
                ), row=cvd_row, col=1, secondary_y=True)

    # ── 12. Market Phase (Balance/Imbalance) — corrections > auto ────────────
    # Приоритет рендера:
    #   1. Если для (symbol, tf) есть corrections в окне → используем их.
    #   2. Иначе — auto-детекция через detect_market_phases_research.
    # BALANCE = синий tint, IMBALANCE_UP = зелёный, IMBALANCE_DOWN = красный.
    _phase_segments = []   # будет заполнено для editor'а под чартом
    _phase_source   = 'none'
    if toggles.get('phase', False):
        from phase_markup import load_corrected_phases

        _t0 = pd.to_datetime(df['timestamp'].iloc[0], utc=True)
        _t1 = pd.to_datetime(df['timestamp'].iloc[-1], utc=True)
        _corr_df = load_corrected_phases(symbol=symbol, tf=tf)
        if len(_corr_df):
            _corr_df = _corr_df[
                (pd.to_datetime(_corr_df['end_ts'],   utc=True) >= _t0) &
                (pd.to_datetime(_corr_df['start_ts'], utc=True) <= _t1)
            ].reset_index(drop=True)

        if len(_corr_df):
            # CORRECTIONS — приоритет
            _phase_source = 'corrected'
            for _, r in _corr_df.iterrows():
                _phase_segments.append({
                    'kind': r['kind'],
                    'ts_start': pd.to_datetime(r['start_ts'], utc=True),
                    'ts_end':   pd.to_datetime(r['end_ts'],   utc=True),
                    'mean_t_stat': float(r.get('t_stat', 0.0)) if pd.notna(r.get('t_stat', 0.0)) else 0.0,
                    'id': r['id'],
                })
        else:
            # AUTO-детектор
            _phase_source = 'auto'
            ph_out = _sv2_market_phases(
                hash(_h), _pkl,
                L_min=toggles.get('phase_L_min', 20),
                L_max=toggles.get('phase_L_max', 80),
                t_stat_threshold=toggles.get('phase_t_threshold', 3.0),
                min_segment_bars=toggles.get('phase_min_segment', 5),
            )
            for seg in ph_out.get('segments', []):
                _phase_segments.append({
                    'kind': seg['kind'],
                    'ts_start': pd.to_datetime(seg['ts_start'], utc=True),
                    'ts_end':   pd.to_datetime(seg['ts_end'],   utc=True),
                    'mean_t_stat': float(seg.get('mean_t_stat', 0.0)),
                    'id': None,
                })

        _phase_color = {
            'balance':        'rgba(64, 156, 255, 0.10)',
            'imbalance_up':   'rgba(0, 200, 100, 0.12)',
            'imbalance_down': 'rgba(255, 50, 80, 0.12)',
        }
        _phase_label = {
            'balance':        '⚖ BAL',
            'imbalance_up':   '🟢 IMB↑',
            'imbalance_down': '🔴 IMB↓',
        }
        for seg in _phase_segments:
            kind = seg['kind']
            if kind not in _phase_color:
                continue
            fig.add_vrect(
                x0=seg['ts_start'], x1=seg['ts_end'],
                fillcolor=_phase_color[kind],
                line=dict(width=0),
                layer='below',
                row=candle_row, col=1,
            )
            try:
                _span = pd.Timestamp(seg['ts_end']) - pd.Timestamp(seg['ts_start'])
                mid_ts = pd.Timestamp(seg['ts_start']) + _span / 2
            except Exception:
                mid_ts = seg['ts_start']
            fig.add_annotation(
                x=mid_ts, y=df['high'].max(),
                text=(f"{_phase_label[kind]} · t={seg['mean_t_stat']:.1f}"
                      if seg.get('mean_t_stat') else _phase_label[kind]),
                showarrow=False, yanchor='top',
                font=dict(size=9,
                          color={'balance': '#7DD3FC',
                                 'imbalance_up': '#00E676',
                                 'imbalance_down': '#FF5577'}[kind]),
                bgcolor='rgba(0,0,0,0.45)',
                row=candle_row, col=1,
            )

    # ── 13. 80% Rule events (Auction Market Theory) ──────────────────────────
    # Маркер на trigger-баре + стрелка к target-уровню (VAL или VAH).
    if toggles.get('rule80', False):
        rule80_events = _sv2_80pct_rule(
            hash(_h), _pkl,
            va_window=toggles.get('rule80_va_window', 50),
            va_pct=toggles.get('rule80_va_pct', 0.70),
            hold_bars=toggles.get('rule80_hold', 2),
        )
        for ev in rule80_events:
            col = '#00E676' if ev['direction'] == 'bullish' else '#FF1744'
            arrow = '↑' if ev['direction'] == 'bullish' else '↓'
            # Маркер trigger
            fig.add_trace(go.Scatter(
                x=[ev['trigger_ts']], y=[float(df['close'].iloc[ev['trigger_idx']])],
                mode='markers',
                marker=dict(size=14, color=col, symbol='star',
                             line=dict(color='white', width=1.5)),
                name=f"80% {ev['direction']}",
                hovertemplate=(f"<b>80% Rule {ev['direction'].upper()}</b><br>"
                               f"target={ev['target']:.4f}<br>"
                               f"VAH={ev['va_high']:.4f}, VAL={ev['va_low']:.4f}<extra></extra>"),
                showlegend=False,
            ), row=candle_row, col=1)
            # Линия от trigger до target
            fig.add_trace(go.Scatter(
                x=[ev['trigger_ts'], ev['trigger_ts']],
                y=[float(df['close'].iloc[ev['trigger_idx']]), ev['target']],
                mode='lines',
                line=dict(color=col, width=1.5, dash='dot'),
                showlegend=False, hoverinfo='skip',
            ), row=candle_row, col=1)
            # Target-маркер
            fig.add_trace(go.Scatter(
                x=[ev['trigger_ts']], y=[ev['target']],
                mode='markers',
                marker=dict(size=8, color=col, symbol='triangle-up' if ev['direction'] == 'bullish' else 'triangle-down',
                             line=dict(color='white', width=1)),
                showlegend=False, hoverinfo='skip',
            ), row=candle_row, col=1)
            # Подпись
            fig.add_annotation(
                x=ev['trigger_ts'], y=float(df['close'].iloc[ev['trigger_idx']]),
                text=f"<b>80%{arrow}</b>",
                showarrow=False, xanchor='left', xshift=8, yshift=12,
                font=dict(size=10, color=col),
                bgcolor='rgba(0,0,0,0.5)',
                row=candle_row, col=1,
            )

    # ── 14. Structural Balance Zones (HH/HL/LL/LH cluster) — Wyckoff/SMC ────
    # Sliding-window K swings: если highs кластеризуются у одной линии, lows у
    # другой, и range ≤ N×ATR → горизонтальный balance. Бирюзовый прямоугольник.
    if toggles.get('struct_bal', False):
        sb_zones = _sv2_struct_balance(
            hash(_h), _pkl,
            swing_len=toggles.get('struct_bal_swing_len', 2),
            min_swings=toggles.get('struct_bal_min_swings', 4),
            max_range_atr=toggles.get('struct_bal_max_range_atr', 3.0),
            cluster_spread_frac=toggles.get('struct_bal_cluster_frac', 0.40),
        )
        for z in sb_zones:
            # Rect — main zone (range_low → range_high)
            fig.add_shape(
                type='rect',
                x0=z['start_ts'], x1=z['end_ts'],
                y0=z['range_low'], y1=z['range_high'],
                fillcolor='rgba(0, 200, 200, 0.10)',
                line=dict(color='rgba(0, 200, 200, 0.85)', width=1.5, dash='dot'),
                layer='below',
                row=candle_row, col=1,
            )
            # Highs cluster band (тонкая полоса у потолка)
            fig.add_shape(
                type='rect',
                x0=z['start_ts'], x1=z['end_ts'],
                y0=z['highs_cluster_bot'], y1=z['highs_cluster_top'],
                fillcolor='rgba(0, 200, 200, 0.18)',
                line=dict(width=0),
                layer='below',
                row=candle_row, col=1,
            )
            # Lows cluster band (тонкая полоса у пола)
            fig.add_shape(
                type='rect',
                x0=z['start_ts'], x1=z['end_ts'],
                y0=z['lows_cluster_bot'], y1=z['lows_cluster_top'],
                fillcolor='rgba(0, 200, 200, 0.18)',
                line=dict(width=0),
                layer='below',
                row=candle_row, col=1,
            )
            # Подпись (в правом верхнем углу зоны)
            fig.add_annotation(
                x=z['end_ts'], y=z['range_high'],
                text=f"<b>🧱 STR-BAL</b> · {z['n_swings']} sw "
                     f"({z['n_highs']}H/{z['n_lows']}L)",
                showarrow=False, xanchor='left', yanchor='bottom',
                xshift=2, yshift=2,
                font=dict(size=9, color='#00C8C8'),
                bgcolor='rgba(0,0,0,0.55)',
                row=candle_row, col=1,
            )

    # ── 15. SAVED POSITIONS (TradingView-style Long/Short markers) ──────────
    # Causal refresh статусов: open позиции для (symbol, tf) проверяются по df.
    try:
        from screener_drawings import refresh_all_positions, load_positions, load_drawings
        refresh_all_positions(symbol, tf, df)  # auto-update open positions
        _positions = load_positions(symbol=symbol, tf=tf)
    except Exception:
        _positions = None
    _last_ts_in_chart = pd.to_datetime(df['timestamp'].iloc[-1], utc=True)
    if _positions is not None and len(_positions):
        for _, pos in _positions.iterrows():
            _dir   = pos['direction']
            _entry = float(pos['entry'])
            _sl    = float(pos['sl_price'])
            _tp    = float(pos['tp_price'])
            _sts   = pos['status']
            _start = pd.to_datetime(pos['start_ts'], utc=True)
            _exit_ts = (pd.to_datetime(pos['exit_ts'], utc=True)
                        if pd.notna(pos.get('exit_ts')) else _last_ts_in_chart)
            # Цветовая схема: profit zone (entry→tp) зелёная, loss zone (entry→sl) красная
            _col_status = {'open': '#FFC107', 'won': '#00E676',
                           'lost': '#FF1744', 'closed_manual': '#9E9E9E'}.get(_sts, '#FFC107')
            # Profit zone
            fig.add_shape(
                type='rect',
                x0=_start, x1=_exit_ts,
                y0=min(_entry, _tp), y1=max(_entry, _tp),
                fillcolor='rgba(0, 230, 118, 0.10)',
                line=dict(color='rgba(0, 230, 118, 0.55)', width=1, dash='dot'),
                layer='below',
                row=candle_row, col=1,
            )
            # Loss zone
            fig.add_shape(
                type='rect',
                x0=_start, x1=_exit_ts,
                y0=min(_entry, _sl), y1=max(_entry, _sl),
                fillcolor='rgba(255, 23, 68, 0.10)',
                line=dict(color='rgba(255, 23, 68, 0.55)', width=1, dash='dot'),
                layer='below',
                row=candle_row, col=1,
            )
            # Entry line (белая)
            fig.add_shape(
                type='line', x0=_start, x1=_exit_ts,
                y0=_entry, y1=_entry,
                line=dict(color='white', width=1.5),
                row=candle_row, col=1,
            )
            # Direction badge + status в правом верхнем углу
            _arrow = '↑' if _dir == 'LONG' else '↓'
            _status_emoji = {'open': '⏳', 'won': '✅', 'lost': '❌',
                              'closed_manual': '🔒'}.get(_sts, '⏳')
            fig.add_annotation(
                x=_exit_ts, y=_entry,
                text=f"<b>{_dir} {_arrow}</b> · {_status_emoji} {_sts.upper()}",
                showarrow=False, xanchor='left', yanchor='middle',
                xshift=4,
                font=dict(size=10, color=_col_status),
                bgcolor='rgba(0,0,0,0.6)',
                row=candle_row, col=1,
            )

    # ── 16. SAVED DRAWINGS (заметки в виде прямоугольников) ─────────────────
    try:
        _drawings = load_drawings(symbol=symbol, tf=tf)
    except Exception:
        _drawings = None
    if _drawings is not None and len(_drawings):
        for _, dr in _drawings.iterrows():
            _color = dr.get('color') or '#FFA726'
            # Конвертируем hex → rgba для fill
            try:
                _r = int(_color[1:3], 16); _g = int(_color[3:5], 16); _b = int(_color[5:7], 16)
                _fill = f'rgba({_r},{_g},{_b},0.12)'
                _line = f'rgba({_r},{_g},{_b},0.85)'
            except Exception:
                _fill = 'rgba(255,167,38,0.12)'
                _line = 'rgba(255,167,38,0.85)'
            fig.add_shape(
                type='rect',
                x0=pd.to_datetime(dr['start_ts'], utc=True),
                x1=pd.to_datetime(dr['end_ts'],   utc=True),
                y0=float(dr['price_low']), y1=float(dr['price_high']),
                fillcolor=_fill,
                line=dict(color=_line, width=1.5),
                layer='below',
                row=candle_row, col=1,
            )
            if dr.get('label'):
                fig.add_annotation(
                    x=pd.to_datetime(dr['end_ts'], utc=True),
                    y=float(dr['price_high']),
                    text=f"<b>{dr['label']}</b>",
                    showarrow=False, xanchor='left', yanchor='bottom',
                    xshift=2, yshift=2,
                    font=dict(size=9, color=_color),
                    bgcolor='rgba(0,0,0,0.55)',
                    row=candle_row, col=1,
                )

    # ── Layout ──────────────────────────────────────────────────────────────
    n_active = sum(1 for k, v in toggles.items() if k in _TOOLBOX_DEFAULT and v)
    fig.update_layout(
        height=720 if not use_cvd_subplot else 820,
        template='plotly_dark',
        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
        xaxis_rangeslider_visible=False, showlegend=False,
        margin=dict(l=10, r=80, t=30, b=10),
        title=f"{symbol} {tf}{'  · Renko' if is_renko else ''}  ·  активно: {n_active}/15",
        # uirevision: НЕ зависит от len(df) — зум сохраняется при online refresh
        uirevision=f"sv2_{symbol}_{tf}",
        # 📦 VP Range активен → 'select' mode + только X-направление
        dragmode=('select' if vp_range_active else 'pan'),
        selectdirection=('h' if vp_range_active else 'd'),
        hovermode='x unified',
    )
    fig.update_xaxes(gridcolor='#1f2937', fixedrange=False)
    fig.update_yaxes(gridcolor='#1f2937', fixedrange=False)

    # ── Strategy Live: зона + VP + сделка (как инструмент TradingView) ──
    if strategy_signals:
        _ns = len(df)
        _ts = df["timestamp"]
        _n_sig = len(strategy_signals)
        _MAXRICH = 15   # богатую отрисовку (зона/VP/позиция) — последним N сигналам
        _VP_LAST = 2    # полную VP-гистограмму — последним 2 зонам (перф)
        _lx, _ly, _sx, _sy = [], [], [], []   # мелкие маркеры входа

        def _ix_ts(_i):
            _i = int(max(0, min(_ns - 1, _i)))
            return _ts.iloc[_i]

        for _ix, _sg in enumerate(strategy_signals):
            _bi = _sg.get("bar_idx")
            if _bi is None or _bi < 0 or _bi >= _ns:
                continue
            _tx = _ts.iloc[_bi]
            _dr = str(_sg.get("direction", "")).upper()
            _islong = _dr.startswith("L")
            _en = _sg.get("entry"); _sl = _sg.get("sl_price"); _tp = _sg.get("tp_price")
            if _en is not None:
                (_lx if _islong else _sx).append(_tx)
                (_ly if _islong else _sy).append(_en)

            if _ix < _n_sig - _MAXRICH:
                continue   # для старых сигналов — только маркер

            try:
                _zs = int(_sg.get("zone_start_idx", -1))
                _ze = int(_sg.get("zone_end_idx", -1))
                _rh = _sg.get("range_high"); _rl = _sg.get("range_low")
                _hc = _sg.get("hvn_center"); _hw = _sg.get("hvn_width")
                _vah = _sg.get("vah"); _val = _sg.get("val"); _poc = _sg.get("poc")
                # ── 1) Структурная зона баланса (рамка) ──
                if 0 <= _zs <= _ze < _ns and _rh is not None and _rl is not None:
                    _z0 = _ts.iloc[_zs]; _z1 = _ts.iloc[_ze]
                    fig.add_shape(type="rect", x0=_z0, x1=_z1, y0=_rl, y1=_rh,
                        line=dict(color="rgba(120,144,156,0.55)", width=1),
                        fillcolor="rgba(120,144,156,0.07)", layer="below",
                        row=candle_row, col=1)
                    # Value Area + POC внутри зоны
                    if _vah is not None and _val is not None:
                        fig.add_shape(type="rect", x0=_z0, x1=_z1, y0=_val, y1=_vah,
                            line=dict(width=0), fillcolor="rgba(33,150,243,0.07)",
                            layer="below", row=candle_row, col=1)
                    if _poc is not None:
                        fig.add_shape(type="line", x0=_z0, x1=_z1, y0=_poc, y1=_poc,
                            line=dict(color="rgba(255,82,82,0.7)", width=1, dash="dot"),
                            row=candle_row, col=1)
                    # ── 2) Зона HVN (объёмное ядро) ──
                    if _hc is not None and _hw:
                        fig.add_shape(type="rect", x0=_z0, x1=_z1,
                            y0=_hc - _hw / 2.0, y1=_hc + _hw / 2.0,
                            line=dict(color="rgba(255,152,0,0.6)", width=1),
                            fillcolor="rgba(255,152,0,0.18)", layer="below",
                            row=candle_row, col=1)
                    # ── 2b) Фиксированный VP внутри зоны (последние N) ──
                    if _ix >= _n_sig - _VP_LAST:
                        try:
                            from smc_core import compute_volume_profile_research as _cvp
                            _zdf = df.iloc[_zs:_ze + 1]
                            _vp = _cvp(_zdf, n_points=80, samples_per_bar=20,
                                       use_kde=False, smooth_win=3)
                            if _vp is not None:
                                _gx = _vp.get("kde_x"); _gd = _vp.get("kde_density")
                                if _gx is not None and _gd is not None and len(_gd):
                                    _dmax = float(max(_gd)) or 1.0
                                    _span = (_z1 - _z0)
                                    _dxp = (float(_gx[1]) - float(_gx[0])) if len(_gx) > 1 else 0.0
                                    for _k in range(len(_gx)):
                                        _d = float(_gd[_k])
                                        if _d <= 0:
                                            continue
                                        _pc = float(_gx[_k])
                                        _w = _span * (_d / _dmax) * 0.5
                                        fig.add_shape(type="rect", x0=_z0, x1=_z0 + _w,
                                            y0=_pc - _dxp / 2.0, y1=_pc + _dxp / 2.0,
                                            line=dict(width=0), fillcolor="rgba(0,150,136,0.22)",
                                            layer="below", row=candle_row, col=1)
                        except Exception:
                            pass
                # ── 3) Сделка «позиция» + forward-sim исхода (TP/SL/таймаут + R) ──
                if _en is not None and _sl is not None and _tp is not None:
                    _islong_t = _dr.startswith("L")
                    _sl_d = abs(_en - _sl)
                    _tpr = _sg.get("tp_r_expected") or (abs(_tp - _en) / _sl_d if _sl_d > 0 else 0.0)
                    _hi_arr = df["high"].values; _lo_arr = df["low"].values; _cl_arr = df["close"].values
                    _exit_i = None; _exit_r = 0.0; _exit_kind = "open"
                    _cap = min(_ns - 1, _bi + 400)
                    for _j in range(_bi + 1, _cap + 1):
                        _hj = float(_hi_arr[_j]); _lj = float(_lo_arr[_j])
                        _hit_sl = (_lj <= _sl) if _islong_t else (_hj >= _sl)
                        _hit_tp = (_hj >= _tp) if _islong_t else (_lj <= _tp)
                        if _hit_sl:                       # пессимизм: SL раньше TP на одном баре
                            _exit_i = _j; _exit_r = -1.0; _exit_kind = "SL"; break
                        if _hit_tp:
                            _exit_i = _j; _exit_r = float(_tpr); _exit_kind = "TP"; break
                    if _exit_i is None:                   # таймаут → нереализованный R
                        _exit_i = _cap
                        _lc = float(_cl_arr[_cap])
                        _exit_r = (((_lc - _en) if _islong_t else (_en - _lc)) / _sl_d) if _sl_d > 0 else 0.0
                    _xe = _tx; _xf = _ix_ts(_exit_i)
                    fig.add_shape(type="rect", x0=_xe, x1=_xf, y0=min(_en, _tp), y1=max(_en, _tp),
                        line=dict(width=0), fillcolor="rgba(0,230,118,0.13)", layer="below", row=candle_row, col=1)
                    fig.add_shape(type="rect", x0=_xe, x1=_xf, y0=min(_en, _sl), y1=max(_en, _sl),
                        line=dict(width=0), fillcolor="rgba(255,82,82,0.13)", layer="below", row=candle_row, col=1)
                    for _yv, _cv in ((_en, "#ffd54f"), (_sl, "#ff5252"), (_tp, "#00e676")):
                        fig.add_shape(type="line", x0=_xe, x1=_xf, y0=_yv, y1=_yv,
                            line=dict(color=_cv, width=1.5), row=candle_row, col=1)
                    # маркер выхода + сколько R взяла сделка
                    _excol = "#00e676" if _exit_r > 0 else ("#ff5252" if _exit_r < 0 else "#90a4ae")
                    _exy = float(_tp if _exit_kind == "TP" else (_sl if _exit_kind == "SL" else _cl_arr[_exit_i]))
                    fig.add_trace(go.Scatter(x=[_ix_ts(_exit_i)], y=[_exy], mode="markers+text",
                        marker=dict(symbol="circle", size=7, color=_excol),
                        text=[f"{_exit_r:+.1f}R"], textposition="top center",
                        textfont=dict(size=10, color=_excol), showlegend=False),
                        row=candle_row, col=1)
            except Exception:
                pass

        # мелкие маркеры входа поверх всего
        if _lx:
            fig.add_trace(go.Scatter(x=_lx, y=_ly, mode="markers", name="LONG entry",
                marker=dict(symbol="triangle-up", size=9, color="#00e676",
                            line=dict(width=1, color="#0a5d2a"))),
                row=candle_row, col=1)
        if _sx:
            fig.add_trace(go.Scatter(x=_sx, y=_sy, mode="markers", name="SHORT entry",
                marker=dict(symbol="triangle-down", size=9, color="#ff5252",
                            line=dict(width=1, color="#7a0000"))),
                row=candle_row, col=1)

    chart_event = st.plotly_chart(
        fig, use_container_width=True, key=chart_key,
        on_select=('rerun' if vp_range_active else 'ignore'),
        selection_mode=('box' if vp_range_active else None),
        config={
            'scrollZoom': True, 'displayModeBar': True,
            'displaylogo': False, 'doubleClick': False,
            'modeBarButtonsToRemove': (['lasso2d'] if vp_range_active
                                        else ['lasso2d', 'select2d']),
        },
    )

    # ── VP Range: извлечение выделения + статус-banner + Clear ──────────────
    if vp_range_active:
        def _extract_box_xrange(ev):
            """Извлекает [x_min, x_max] из chart_event selection.box."""
            try:
                sel = ev.get('selection') if isinstance(ev, dict) else getattr(ev, 'selection', None)
                if not sel: return None
                box = sel.get('box') if isinstance(sel, dict) else getattr(sel, 'box', None)
                if not box: return None
                b = box[0] if isinstance(box, list) else box
                x_arr = b.get('x') if isinstance(b, dict) else getattr(b, 'x', None)
                if not x_arr or len(x_arr) < 2: return None
                return [str(x) for x in x_arr]
            except Exception:
                return None

        _new_box = (_extract_box_xrange(chart_event)
                    or _extract_box_xrange(st.session_state.get(chart_key)))
        if _new_box is not None:
            try:
                _ts0_new = pd.to_datetime(min(_new_box), utc=True)
                _ts1_new = pd.to_datetime(max(_new_box), utc=True)
                _new_sel = (_ts0_new, _ts1_new)
                if _selection_ts != _new_sel:
                    st.session_state[vp_box_key] = _new_sel
                    st.rerun()
            except Exception:
                pass

        sb_cols = st.columns([4, 1])
        if _vp_in_range:
            _n_bars_sel = int(((df['timestamp'] >= vp_x0) & (df['timestamp'] <= vp_x1)).sum())
            sb_cols[0].success(
                f"📦 **VP Range активен** · нарисованы POC + VAH + VAL + HVN + LVN "
                f"внутри выделения · {_n_bars_sel} баров · "
                f"`{pd.Timestamp(vp_x0).strftime('%Y-%m-%d %H:%M')}` → "
                f"`{pd.Timestamp(vp_x1).strftime('%Y-%m-%d %H:%M')}`"
            )
        else:
            sb_cols[0].info(
                "📦 **VP Range активен** · натяни прямоугольник на графике (drag по X) — "
                "автоматически нарисуются POC + VAH + VAL + HVN + LVN внутри выделения. "
                "Параметры (VA%, γ_height, prom) настраиваются в Toolbox выше."
            )
        if sb_cols[1].button("🗑 Сбросить", key=f"clr_{vp_box_key}", use_container_width=True):
            for _k in (vp_box_key, chart_key):
                if _k in st.session_state:
                    del st.session_state[_k]
            st.rerun()

    # ── 📍 POSITIONS PANEL (Long/Short TradingView-style markup) ────────────
    from screener_drawings import (
        save_position, delete_position, load_positions as _load_pos,
        save_drawing, delete_drawing, load_drawings as _load_dr,
        DRAWING_COLORS,
    )
    _pos_key = f"pos_{symbol}_{tf}_{chart_key}"
    _dr_key  = f"dr_{symbol}_{tf}_{chart_key}"
    _last_close = float(df['close'].iloc[-1])
    _last_ts_iso = pd.to_datetime(df['timestamp'].iloc[-1], utc=True)

    with st.expander(
        f"📍 **Positions & Drawings** · {symbol} {tf} · "
        f"saved: {len(_positions) if _positions is not None else 0} pos, "
        f"{len(_drawings) if _drawings is not None else 0} drawings",
        expanded=False,
    ):
        st.caption(
            "Marks сохраняются в `screener_positions.parquet` / `screener_drawings.parquet`. "
            "Статус позиций обновляется автоматически (causal walk через bars after entry)."
        )

        # ── ADD POSITION form ──────────────────────────────────────────────
        st.markdown("##### ➕ Add Position")
        pf_cols = st.columns([1.2, 1, 1, 1, 1.5, 2])
        _new_dir = pf_cols[0].selectbox("Direction", ['LONG', 'SHORT'],
                                          key=f"{_pos_key}_dir")
        _new_entry = pf_cols[1].number_input("Entry", value=float(_last_close),
                                              format="%.6f", key=f"{_pos_key}_e")
        # Дефолты SL/TP — ±1% от entry
        if _new_dir == 'LONG':
            _def_sl = _new_entry * 0.99; _def_tp = _new_entry * 1.02
        else:
            _def_sl = _new_entry * 1.01; _def_tp = _new_entry * 0.98
        _new_sl = pf_cols[2].number_input("SL", value=float(_def_sl),
                                            format="%.6f", key=f"{_pos_key}_sl")
        _new_tp = pf_cols[3].number_input("TP", value=float(_def_tp),
                                            format="%.6f", key=f"{_pos_key}_tp")
        _new_ts = pf_cols[4].text_input("Entry bar (UTC, blank=last)",
                                          value='', placeholder=str(_last_ts_iso)[:19],
                                          key=f"{_pos_key}_ts")
        _new_comment = pf_cols[5].text_input("Comment", value='',
                                                key=f"{_pos_key}_c")
        if st.button("💾 Save position", key=f"{_pos_key}_save",
                     type='primary'):
            try:
                _entry_ts = (pd.to_datetime(_new_ts, utc=True)
                             if _new_ts.strip() else _last_ts_iso)
                save_position(symbol, tf, _new_dir,
                                _new_entry, _new_sl, _new_tp,
                                _entry_ts, comment=_new_comment)
                st.success(f"✅ Position saved · {_new_dir} @ {_new_entry:.4f}")
                st.rerun()
            except ValueError as e:
                st.error(f"Invalid: {e}")
            except Exception as e:
                st.error(f"Save error: {e}")

        # ── EXISTING POSITIONS list ────────────────────────────────────────
        if _positions is not None and len(_positions):
            st.markdown("##### 📋 Positions")
            for _, p in _positions.iterrows():
                pcol = st.columns([1, 1, 1, 1, 1, 1.5, 1, 0.8])
                _sts_emoji = {'open': '⏳', 'won': '✅', 'lost': '❌',
                                'closed_manual': '🔒'}.get(p['status'], '⏳')
                pcol[0].write(f"**{p['direction']}** {_sts_emoji}")
                pcol[1].write(f"E: {p['entry']:.4f}")
                pcol[2].write(f"SL: {p['sl_price']:.4f}")
                pcol[3].write(f"TP: {p['tp_price']:.4f}")
                pcol[4].write(f"`{p['status']}`")
                _exit_str = (str(p['exit_ts'])[:16] if pd.notna(p.get('exit_ts')) else '—')
                pcol[5].write(f"exit: {_exit_str}")
                pcol[6].write((p.get('comment') or '')[:18])
                if pcol[7].button("🗑", key=f"{_pos_key}_del_{p['id']}"):
                    delete_position(p['id'])
                    st.rerun()

        # ── ADD DRAWING via box-select hint ────────────────────────────────
        st.markdown("---")
        st.markdown("##### ➕ Add Drawing (form)")
        df_cols = st.columns([1.5, 1.5, 1, 1, 1.2, 1.2, 1.5])
        _d_label = df_cols[0].text_input("Label", value='', key=f"{_dr_key}_l")
        _color_name = df_cols[1].selectbox("Color",
                                             list(DRAWING_COLORS.keys()),
                                             key=f"{_dr_key}_clr")
        _d_low  = df_cols[2].number_input("Low",  value=float(_last_close * 0.98),
                                            format="%.6f", key=f"{_dr_key}_pl")
        _d_high = df_cols[3].number_input("High", value=float(_last_close * 1.02),
                                            format="%.6f", key=f"{_dr_key}_ph")
        _d_start = df_cols[4].text_input("Start UTC",
                                            value=str(df['timestamp'].iloc[max(0, len(df)-50)])[:19],
                                            key=f"{_dr_key}_st")
        _d_end = df_cols[5].text_input("End UTC",
                                          value=str(_last_ts_iso)[:19],
                                          key=f"{_dr_key}_et")
        if df_cols[6].button("💾 Save drawing", key=f"{_dr_key}_save",
                              type='primary'):
            try:
                save_drawing(symbol, tf,
                              pd.to_datetime(_d_start, utc=True),
                              pd.to_datetime(_d_end,   utc=True),
                              price_low=_d_low, price_high=_d_high,
                              label=_d_label,
                              color=DRAWING_COLORS[_color_name])
                st.success(f"✅ Drawing saved · {_d_label or '(no label)'}")
                st.rerun()
            except Exception as e:
                st.error(f"Save error: {e}")

        # ── EXISTING DRAWINGS list ─────────────────────────────────────────
        if _drawings is not None and len(_drawings):
            st.markdown("##### 📋 Drawings")
            for _, d in _drawings.iterrows():
                dcol = st.columns([2, 1, 1, 1.5, 1.5, 0.8])
                dcol[0].write(f"**{d.get('label') or '(no label)'}**")
                dcol[1].write(f"L: {d['price_low']:.4f}")
                dcol[2].write(f"H: {d['price_high']:.4f}")
                dcol[3].write(f"`{str(d['start_ts'])[:16]}`")
                dcol[4].write(f"`{str(d['end_ts'])[:16]}`")
                if dcol[5].button("🗑", key=f"{_dr_key}_del_{d['id']}"):
                    delete_drawing(d['id'])
                    st.rerun()

    # ── 🌊 Market Phase Editor: правка границ + класса + Save corrections ────
    # Виден только если toggle 'phase' активен. Использует _phase_segments,
    # собранный в блоке #12 (либо corrections, либо auto-детектор).
    if toggles.get('phase', False) and _phase_segments:
        from phase_markup import (
            replace_corrections_in_window, reset_corrections_in_window,
            count_corrections_by_kind,
        )

        _editor_key = f"phase_editor_{symbol}_{tf}_{chart_key}"
        _t0 = pd.to_datetime(df['timestamp'].iloc[0], utc=True)
        _t1 = pd.to_datetime(df['timestamp'].iloc[-1], utc=True)

        with st.expander(
            f"📝 **Фазы в окне** · источник: "
            f"{'✏️ corrected' if _phase_source == 'corrected' else '🤖 auto-detector'} · "
            f"{len(_phase_segments)} сегментов",
            expanded=False,
        ):
            st.caption(
                "Правь границы (start/end), класс (kind), удаляй ненужные. "
                "💾 Save corrections — сохранит все строки как `phase_corrections.parquet`. "
                "🗑 Reset to auto — удалит corrections, вернёт auto-детектор."
            )
            # Готовим DataFrame для st.data_editor — только нужные колонки
            _ed_rows = []
            for s in _phase_segments:
                _ed_rows.append({
                    'kind': s['kind'],
                    'start_ts': pd.to_datetime(s['ts_start'], utc=True),
                    'end_ts':   pd.to_datetime(s['ts_end'],   utc=True),
                    'original_kind': s['kind'] if _phase_source == 'auto' else None,
                })
            _ed_df = pd.DataFrame(_ed_rows)
            edited = st.data_editor(
                _ed_df,
                num_rows='dynamic',
                use_container_width=True,
                column_config={
                    'kind': st.column_config.SelectboxColumn(
                        "Класс",
                        options=['balance', 'imbalance_up', 'imbalance_down'],
                        required=True,
                    ),
                    'start_ts': st.column_config.DatetimeColumn(
                        "Начало (UTC)", required=True, step=60,
                    ),
                    'end_ts': st.column_config.DatetimeColumn(
                        "Конец (UTC)", required=True, step=60,
                    ),
                    'original_kind': st.column_config.TextColumn(
                        "Был (auto)", disabled=True, width='small',
                    ),
                },
                hide_index=True,
                key=_editor_key,
            )

            # Сводка по corrections (на всём датасете)
            _ccnt = count_corrections_by_kind()
            st.caption(
                f"📊 Всего corrections в БД: "
                f"⚖ {_ccnt['balance']} | 🟢 {_ccnt['imbalance_up']} | 🔴 {_ccnt['imbalance_down']}"
            )

            ec1, ec2, ec3 = st.columns([1, 1, 4])
            if ec1.button("💾 Save corrections", key=f"save_{_editor_key}",
                          use_container_width=True, type='primary'):
                try:
                    new_phases = []
                    for _, r in edited.iterrows():
                        kind = r.get('kind')
                        s_ts = r.get('start_ts')
                        e_ts = r.get('end_ts')
                        if (kind in ('balance', 'imbalance_up', 'imbalance_down') and
                            pd.notna(s_ts) and pd.notna(e_ts) and s_ts < e_ts):
                            new_phases.append({
                                'start_ts': pd.to_datetime(s_ts, utc=True),
                                'end_ts':   pd.to_datetime(e_ts, utc=True),
                                'kind': kind,
                                'source': 'corrected',
                                'original_kind': r.get('original_kind'),
                            })
                    n = replace_corrections_in_window(
                        symbol, tf, _t0, _t1, new_phases, df_full=df,
                    )
                    st.success(f"✅ Сохранено {n} фаз в phase_corrections.parquet")
                    st.rerun()
                except Exception as e:
                    st.error(f"Save error: {e}")

            if ec2.button("🗑 Reset to auto", key=f"reset_{_editor_key}",
                          use_container_width=True):
                n_del = reset_corrections_in_window(symbol, tf, _t0, _t1)
                st.info(f"Удалено {n_del} corrections в окне. Auto-детектор активен.")
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  ВКЛАДКА: 📈 Crypto Candles — все 82 крипто-пары с Toolbox
# ══════════════════════════════════════════════════════════════════════════════
if "Crypto Candles" in tab:
    from balance_zone_markup import list_available_symbols
    from bt_base import CRYPTO_SYMBOLS as _SV2_CRYPTO

    st.header("📈 Crypto Candles — 82 пары · Toolbox 11 инструментов")
    st.caption("Каждый инструмент включается отдельно в Toolbox ниже для валидации.")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        sv2_c_sym = st.selectbox("Symbol:", _SV2_CRYPTO,
                                  index=0, key="sv2_crypto_sym")
    with c2:
        sv2_c_tf = st.selectbox("ТФ:", ["15m", "1h", "1D"], index=0, key="sv2_crypto_tf")
    with c3:
        sv2_c_nbars = st.number_input("Баров:", 200, 5000, 1500, 100, key="sv2_crypto_nbars")

    toggles_c = _render_toolbox("sv2_crypto")

    df_c = _sv2_load_ohlcv_crypto(sv2_c_sym, sv2_c_tf, int(sv2_c_nbars))
    if df_c.empty:
        st.error(f"Нет данных для {sv2_c_sym} {sv2_c_tf}")
        st.stop()
    _last_ts_c = pd.to_datetime(df_c['timestamp'].iloc[-1], utc=True)
    _age_c = pd.Timestamp.utcnow() - _last_ts_c
    _age_str_c = (f"{_age_c.days}d {_age_c.seconds // 3600}h" if _age_c.days > 0
                   else f"{_age_c.seconds // 60}m {_age_c.seconds % 60}s")
    st.caption(f"📊 {sv2_c_sym} {sv2_c_tf}  ·  баров: {len(df_c)}  ·  "
               f"last: {_last_ts_c.strftime('%Y-%m-%d %H:%M UTC')}  ·  age: {_age_str_c}")
    _render_chart_with_tools(
        df_c, sv2_c_sym, sv2_c_tf, toggles_c,
        chart_key=f"sv2_crypto_chart_{sv2_c_sym}_{sv2_c_tf}_{sv2_c_nbars}",
    )
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  ВКЛАДКА: 💹 Forex Candles — 12 forex-пар с Toolbox
# ══════════════════════════════════════════════════════════════════════════════
if "Forex Candles" in tab:
    from bt_base import FOREX_SYMBOLS as _SV2_FOREX, DATA_DIR as _SV2_DATA_DIR

    st.header("💹 Forex Candles — 12 пар · Toolbox 11 инструментов")
    st.caption("EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD, EURGBP, "
               "EURJPY, GBPJPY, AUDJPY, XAUUSD (золото).")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        sv2_f_sym = st.selectbox("Symbol:", _SV2_FOREX, index=0, key="sv2_forex_sym")
    with c2:
        sv2_f_tf = st.selectbox("ТФ:", ["15m", "1h", "1D"], index=0, key="sv2_forex_tf")
    with c3:
        sv2_f_nbars = st.number_input("Баров:", 200, 5000, 1500, 100, key="sv2_forex_nbars")

    toggles_f = _render_toolbox("sv2_forex")

    # Online loader: parquet + yfinance splice (определён в общем блоке выше)
    df_f = _sv2_load_ohlcv_forex_online(sv2_f_sym, sv2_f_tf, int(sv2_f_nbars))
    if df_f.empty:
        st.error(f"Нет данных для {sv2_f_sym} {sv2_f_tf}")
        st.stop()
    _last_ts_f = pd.to_datetime(df_f['timestamp'].iloc[-1], utc=True)
    _age_f = pd.Timestamp.utcnow() - _last_ts_f
    _age_str_f = f"{_age_f.days}d {_age_f.seconds // 3600}h" if _age_f.days > 0 \
                  else f"{_age_f.seconds // 60}m"
    st.caption(f"📊 {sv2_f_sym} {sv2_f_tf}  ·  баров: {len(df_f)}  ·  "
               f"last: {_last_ts_f.strftime('%Y-%m-%d %H:%M UTC')}  ·  age: {_age_str_f}")
    _render_chart_with_tools(
        df_f, sv2_f_sym, sv2_f_tf, toggles_f,
        chart_key=f"sv2_forex_chart_{sv2_f_sym}_{sv2_f_tf}_{sv2_f_nbars}",
    )
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  ВКЛАДКА: 🧱 Renko — все 94 пары (82 crypto + 12 forex) с Toolbox
# ══════════════════════════════════════════════════════════════════════════════
if "Renko" in tab and "ML" not in tab:
    from bt_base import (CRYPTO_SYMBOLS as _SV2_C2, FOREX_SYMBOLS as _SV2_F2,
                         DATA_DIR as _SV2_DD, build_renko)

    st.header("🧱 Renko — 94 пары · Toolbox 11 инструментов")
    st.caption("Renko строится из 1m свечей. Box size = box_pct × close (по умолчанию 0.1%).")

    _SV2_ALL = list(_SV2_C2) + list(_SV2_F2)

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        sv2_r_sym = st.selectbox("Symbol:", _SV2_ALL, index=0, key="sv2_renko_sym")
    with c2:
        sv2_r_box = st.number_input("Box %:", 0.05, 5.0, 0.10, 0.05,
                                     format="%.2f", key="sv2_renko_box",
                                     help="Размер кирпича в % от цены")
    with c3:
        sv2_r_nbars = st.number_input("Min 1m баров:", 1000, 50000, 10000, 1000,
                                       key="sv2_renko_nbars",
                                       help="Сколько 1m свечей грузить для построения Renko")

    toggles_r = _render_toolbox("sv2_renko")

    @st.cache_data(ttl=600, show_spinner="Строю Renko…")
    def _sv2_load_renko(sym: str, box_pct: float, n_bars_1m: int) -> pd.DataFrame:
        # 1m данные: crypto в DATA_DIR/{sym}_1m.parquet, forex в cache_forex
        if sym in set(_SV2_F2):
            fname = sym[:3] + '_' + sym[3:]
            p = _SV2_DD / 'cache_forex' / f'{fname}_1m.parquet'
            if not p.exists():
                # forex 1m часто отсутствует — fallback на 3m
                p = _SV2_DD / 'cache_forex' / f'{fname}_3m.parquet'
                if not p.exists():
                    return pd.DataFrame()
        else:
            p = _SV2_DD / f'{sym}_1m.parquet'
            if not p.exists():
                p2 = _SV2_DD / 'cache' / f'{sym}_1m.parquet'
                if p2.exists():
                    p = p2
                else:
                    return pd.DataFrame()
        df_1m = pd.read_parquet(p)
        if 'timestamp' in df_1m.columns:
            df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'], utc=True)
        df_1m = df_1m.sort_values('timestamp').reset_index(drop=True)
        if n_bars_1m > 0 and len(df_1m) > n_bars_1m:
            df_1m = df_1m.tail(n_bars_1m).reset_index(drop=True)
        # box size = box_pct × последняя цена
        _close_now = float(df_1m['close'].iloc[-1])
        box_size = max(1e-8, _close_now * box_pct / 100.0)
        try:
            df_renko = build_renko(df_1m, box_size)
        except Exception as _re:
            st.error(f"build_renko failed: {_re}")
            return pd.DataFrame()
        return df_renko

    df_r = _sv2_load_renko(sv2_r_sym, float(sv2_r_box), int(sv2_r_nbars))
    if df_r.empty:
        st.error(f"Нет данных для {sv2_r_sym} (нужен 1m parquet) или Renko build упал")
        st.stop()
    st.caption(f"🧱 {sv2_r_sym}  ·  Renko-баров: {len(df_r)}  ·  box: {sv2_r_box}%")
    _render_chart_with_tools(
        df_r, sv2_r_sym, f"Renko {sv2_r_box}%", toggles_r,
        chart_key=f"sv2_renko_chart_{sv2_r_sym}_{sv2_r_box}_{sv2_r_nbars}",
        is_renko=True,
    )
    st.stop()
# ==============================================================================
#  TAB: Strategy Live -- live detect_signals overlay for any bt_* strategy
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def _sv2_list_strategies():
    from pathlib import Path as _P
    base = _P(__file__).parent
    out = []
    for fp in sorted(base.glob("bt_*.py")):
        if fp.name == "bt_base.py":
            continue
        try:
            txt = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if "def detect_signals" in txt:
            out.append(fp.stem)
    return out


def _sv2_run_strategy(mod_name, df, tf, symbol):
    import importlib, inspect
    try:
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, "detect_signals", None)
        if fn is None:
            return [], "no detect_signals"
        params = list(inspect.signature(fn).parameters)
        kwargs = {}
        if "sym" in params:
            kwargs["sym"] = symbol
        elif "symbol" in params:
            kwargs["symbol"] = symbol
        if "btc_returns" in params:
            kwargs["btc_returns"] = None
        # MTF: если стратегия принимает htf_dfs и объявила старшие ТФ зон —
        # подгружаем их (1h/4h), чтобы скринер показывал то же, что бэктест.
        if "htf_dfs" in params:
            _zone_tfs = getattr(mod, "_MTF_ZONE_TFS", [])
            if _zone_tfs and getattr(mod, "_MTF_MODE", False) and hasattr(mod, "_load_tf"):
                _isfx = symbol in getattr(mod, "_FOREX_SYMS", set())
                _hd = {}
                for _h in _zone_tfs:
                    try:
                        _hdf = mod._load_tf(symbol, _h, _isfx)
                        if _hdf is not None and len(_hdf):
                            _hd[_h] = _hdf
                    except Exception:
                        pass
                kwargs["htf_dfs"] = _hd
        sigs = fn(df, tf, **kwargs)
        return (list(sigs) if sigs else []), None
    except Exception as e:
        import traceback
        return [], f"{e}\n{traceback.format_exc()[-700:]}"


if "Strategy Live" in tab:
    from bt_base import CRYPTO_SYMBOLS as _SV2_SC, FOREX_SYMBOLS as _SV2_SF

    st.header("\U0001F4E1 Strategy Live — сигналы стратегий в реальном времени")
    st.caption("Пара + ТФ + стратегия → её detect_signals() считается с нуля на видимых данных и рисуется на графике. bare-core → сигналов много (норма).")

    _SV2_ALL_S = list(_SV2_SC) + list(_SV2_SF)
    _forex_set_s = set(_SV2_SF)

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        sv2_s_sym = st.selectbox("Symbol:", _SV2_ALL_S, index=0, key="sv2_strat_sym")
    _is_fx_s = sv2_s_sym in _forex_set_s
    with c2:
        _tf_opts_s = (["3m", "15m", "1h", "1D"] if _is_fx_s else ["1m", "15m", "1h", "1D"])
        sv2_s_tf = st.selectbox("ТФ:", _tf_opts_s, index=1, key="sv2_strat_tf")
    with c3:
        sv2_s_nbars = st.number_input("Баров:", 200, 5000, 1500, 100, key="sv2_strat_nbars")
    with c4:
        sv2_s_maxsig = st.number_input("Макс сигналов:", 10, 5000, 200, 10,
                                        key="sv2_strat_maxsig",
                                        help="Рисуем последние N сигналов.")

    _strats = _sv2_list_strategies()
    if not _strats:
        st.error("Нет стратегий bt_*.py с detect_signals")
        st.stop()
    sv2_s_strat = st.selectbox(
        "Стратегия (bt_*):", _strats,
        format_func=lambda m: m.replace("bt_", "").replace("_Bar", "").replace("_Renko", " (Renko)"),
        key="sv2_strat_pick",
    )

    toggles_s = _render_toolbox("sv2_strat")

    if _is_fx_s:
        df_s = _sv2_load_ohlcv_forex_online(sv2_s_sym, sv2_s_tf, int(sv2_s_nbars))
    else:
        df_s = _sv2_load_ohlcv_crypto(sv2_s_sym, sv2_s_tf, int(sv2_s_nbars))
    if df_s is None or df_s.empty:
        st.error(f"Нет данных для {sv2_s_sym} {sv2_s_tf}")
        st.stop()
    df_s = df_s.reset_index(drop=True)

    _last_ts_s = pd.to_datetime(df_s["timestamp"].iloc[-1], utc=True)
    _age_s = pd.Timestamp.utcnow() - _last_ts_s
    _age_str_s = (f"{_age_s.days}d {_age_s.seconds // 3600}h" if _age_s.days > 0
                   else f"{_age_s.seconds // 60}m")
    st.caption(f"\U0001F4CA {sv2_s_sym} {sv2_s_tf}  ·  баров: {len(df_s)}  ·  "
               f"last: {_last_ts_s.strftime('%Y-%m-%d %H:%M UTC')}  ·  age: {_age_str_s}")

    if sv2_s_strat.endswith("_Renko"):
        st.info("⚠️ Renko-стратегия: detect_signals считается на свечном df — сигналы могут быть некорректны.")

    with st.spinner(f"Считаю сигналы {sv2_s_strat}…"):
        _signals, _serr = _sv2_run_strategy(sv2_s_strat, df_s, sv2_s_tf, sv2_s_sym)
    if _serr:
        st.warning(f"⚠️ {sv2_s_strat}: {_serr}")

    _n_total = len(_signals)
    _signals_sorted = sorted(_signals, key=lambda s: s.get("bar_idx", 0))
    _maxn = int(sv2_s_maxsig)
    _signals_draw = _signals_sorted[-_maxn:] if _n_total > _maxn else _signals_sorted
    _n_long = sum(1 for s in _signals_draw
                  if str(s.get("direction", s.get("pattern_type", ""))).upper().startswith("L"))
    _n_short = len(_signals_draw) - _n_long
    m1, m2, m3 = st.columns(3)
    m1.metric("Сигналов всего", _n_total)
    m2.metric("LONG", _n_long)
    m3.metric("SHORT", _n_short)

    _render_chart_with_tools(
        df_s, sv2_s_sym, sv2_s_tf, toggles_s,
        chart_key=f"sv2_strat_chart_{sv2_s_sym}_{sv2_s_tf}_{sv2_s_nbars}_{sv2_s_strat}",
        strategy_signals=_signals_draw,
    )

    with st.expander(f"\U0001F4CB Таблица сигналов ({len(_signals_draw)} из {_n_total})", expanded=False):
        if _signals_draw:
            _rows = []
            for s in _signals_draw:
                _bi = s.get("bar_idx")
                _ts = (pd.to_datetime(df_s["timestamp"].iloc[_bi])
                       if (_bi is not None and 0 <= _bi < len(df_s)) else None)
                _rows.append({
                    "bar": _bi, "time": _ts,
                    "dir": s.get("direction", s.get("pattern_type", "")),
                    "entry": s.get("entry"), "sl": s.get("sl_price"),
                    "key_level": s.get("key_level"), "pattern": s.get("pattern"),
                })
            st.dataframe(pd.DataFrame(_rows), use_container_width=True, height=320)
        else:
            st.info("Сигналов нет на текущем окне.")
    st.stop()




# ══════════════════════════════════════════════════════════════════════════════
#  ВКЛАДКА 17z: 🧠 ML Zone Lab — единая разметка + обучение модели (M1 backbone)
#  Заменит 17b/c/d/e после M4. Сейчас работает параллельно.
# ══════════════════════════════════════════════════════════════════════════════
if "ML Zone Lab" in tab:
    from balance_zone_markup import (
        list_available_symbols, load_ohlcv_bars,
        compute_volume_profile, compute_zone_stats,
        save_zone, load_saved_zones, delete_zone,
        save_detected_as_negative, load_negative_zones, delete_negative_zone,
        save_corrected_zone, load_corrected_zones, delete_corrected_zone,
    )
    from detect_balance_zones import detect_balance_zones_ml, ML_MODEL_PATH

    st.header("🧠 ML Zone Lab — единая разметка + обучение")
    st.caption(
        "Одна вкладка для всего: разметка positive (📐), правка границ детектор-зон (🔧), "
        "пометка false-positive (❌). Действия НЕ перезагружают страницу. "
        "Кеш данных, retrain модели — здесь же."
    )

    # ── Cached loaders ────────────────────────────────────────────────────────
    @st.cache_data(ttl=600, show_spinner=False)
    def _zl_load_ohlcv(sym, tf_, n_bars_):
        d = load_ohlcv_bars(sym, tf_, n_bars=n_bars_)
        if d.empty: return d
        d = d.reset_index(drop=True)
        d['timestamp'] = pd.to_datetime(d['timestamp'], utc=True)
        return d

    @st.cache_data(ttl=600, show_spinner="Сканирую ML…")
    def _zl_detect(sym, tf_, n_bars_, proba_, density_, _det_bump):
        d = _zl_load_ohlcv(sym, tf_, n_bars_)
        if d.empty: return pd.DataFrame()
        return detect_balance_zones_ml(d, tf=tf_, symbol=sym,
                                       min_proba=proba_, max_per_1k_bars=density_)

    @st.cache_data(ttl=600, show_spinner=False)
    def _zl_load_pos(sym, tf_, _b):  return load_saved_zones(symbol=sym, tf=tf_)
    @st.cache_data(ttl=600, show_spinner=False)
    def _zl_load_neg(sym, tf_, _b):  return load_negative_zones(symbol=sym, tf=tf_)
    @st.cache_data(ttl=600, show_spinner=False)
    def _zl_load_corr(sym, tf_, _b): return load_corrected_zones(symbol=sym, tf=tf_)

    def _zl_bump(kind: str):
        k = f'zl_{kind}_bump'
        st.session_state[k] = st.session_state.get(k, 0) + 1

    # ── CONTROLS (top sticky) ─────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
    with c1:
        zl_market = st.radio("Рынок:", ["crypto", "forex"], horizontal=True, key="zl_market")
    with c2:
        zl_tf = st.selectbox("ТФ:", ["15m", "1h", "1D"], index=0, key="zl_tf")
    zl_symbols = list_available_symbols(zl_tf, src=zl_market)
    if not zl_symbols:
        st.warning(f"Нет parquet для {zl_market} {zl_tf}.")
        st.stop()
    with c3:
        _def_sym = "BTCUSDT" if "BTCUSDT" in zl_symbols else zl_symbols[0]
        zl_symbol = st.selectbox("Symbol:", zl_symbols,
                                  index=zl_symbols.index(_def_sym), key="zl_symbol")
    with c4:
        zl_nbars = st.number_input("Баров:", 500, 5000, 2000, 100, key="zl_nbars")

    c5, c6, c7 = st.columns(3)
    with c5:
        zl_proba = st.slider("Min proba (ML):", 0.10, 0.80, 0.30, 0.05, key="zl_proba")
    with c6:
        _def_density = {'15m': 12, '1h': 14, '1D': 19}.get(zl_tf, 12)
        zl_density = st.number_input("Зон на 1k:", 5, 50, _def_density, key="zl_density")
    with c7:
        zl_minbars = st.number_input("Min n_bars:", 5, 300, 15, key="zl_minbars")

    # ── DATA LOAD (cached) ────────────────────────────────────────────────────
    df_zl = _zl_load_ohlcv(zl_symbol, zl_tf, int(zl_nbars))
    if df_zl.empty:
        st.error("Нет данных")
        st.stop()

    pos_bump  = st.session_state.get('zl_pos_bump', 0)
    neg_bump  = st.session_state.get('zl_neg_bump', 0)
    corr_bump = st.session_state.get('zl_corr_bump', 0)
    det_bump  = st.session_state.get('zl_det_bump', 0)

    has_ml = ML_MODEL_PATH.exists()
    if has_ml:
        detected_zl = _zl_detect(zl_symbol, zl_tf, int(zl_nbars),
                                  float(zl_proba), float(zl_density), det_bump)
        if len(detected_zl):
            detected_zl = detected_zl[detected_zl['n_bars'] >= int(zl_minbars)].reset_index(drop=True)
            detected_zl = detected_zl.sort_values('start_ts').reset_index(drop=True)
    else:
        detected_zl = pd.DataFrame()

    pos_all  = _zl_load_pos(zl_symbol, zl_tf, pos_bump)
    neg_all  = _zl_load_neg(zl_symbol, zl_tf, neg_bump)
    corr_all = _zl_load_corr(zl_symbol, zl_tf, corr_bump)

    # Filter by window
    _t_min = pd.to_datetime(df_zl['timestamp'].iloc[0], utc=True)
    _t_max = pd.to_datetime(df_zl['timestamp'].iloc[-1], utc=True)
    def _zl_in_window(zdf):
        if not len(zdf): return zdf
        return zdf[
            (pd.to_datetime(zdf['end_ts'], utc=True) >= _t_min) &
            (pd.to_datetime(zdf['start_ts'], utc=True) <= _t_max)
        ].reset_index(drop=True)
    pos_w  = _zl_in_window(pos_all)
    neg_w  = _zl_in_window(neg_all)
    corr_w = _zl_in_window(corr_all)

    # Status per detected zone (IoU-based matching)
    def _zl_has_iou_match(z_row, others_df, iou_min=0.30):
        if not len(others_df): return False
        zs = pd.to_datetime(z_row['start_ts'], utc=True)
        ze = pd.to_datetime(z_row['end_ts'], utc=True)
        for _, o in others_df.iterrows():
            os_ = pd.to_datetime(o['start_ts'], utc=True)
            oe = pd.to_datetime(o['end_ts'], utc=True)
            inter = max(0.0, (min(ze, oe) - max(zs, os_)).total_seconds())
            union = (max(ze, oe) - min(zs, os_)).total_seconds()
            if union > 0 and inter / union >= iou_min:
                return True
        return False

    if len(detected_zl):
        # Cache IoU matching в session_state — O(N×M) при rerun слишком дорого.
        # Ключ зависит от ВСЕХ bumps → авто-invalidates после любого save/delete.
        _status_cache_key = (
            f"zl_stat_{zl_symbol}_{zl_tf}_{zl_nbars}_"
            f"d{det_bump}_p{pos_bump}_n{neg_bump}_c{corr_bump}"
        )
        cached_statuses = st.session_state.get(_status_cache_key)
        if cached_statuses is not None and len(cached_statuses) == len(detected_zl):
            statuses = cached_statuses
        else:
            corr_orig_ids = set(corr_w['original_id'].astype(str)) if len(corr_w) else set()
            statuses = []
            for _, z in detected_zl.iterrows():
                if _zl_has_iou_match(z, pos_w):
                    statuses.append('positive')
                elif _zl_has_iou_match(z, neg_w):
                    statuses.append('negative')
                elif z['id'] in corr_orig_ids:
                    statuses.append('corrected')
                else:
                    statuses.append('pending')
            st.session_state[_status_cache_key] = statuses
        detected_zl = detected_zl.copy()
        detected_zl['_status'] = statuses

    # ── STATUS BAR (model + dataset overview) ────────────────────────────────
    # Cache total counts с bump invalidation
    @st.cache_data(ttl=600, show_spinner=False)
    def _zl_total_pos(_b):  return len(load_saved_zones())
    @st.cache_data(ttl=600, show_spinner=False)
    def _zl_total_neg(_b):  return len(load_negative_zones())
    @st.cache_data(ttl=600, show_spinner=False)
    def _zl_total_corr(_b): return len(load_corrected_zones())
    n_pos_total  = _zl_total_pos(pos_bump)
    n_neg_total  = _zl_total_neg(neg_bump)
    n_corr_total = _zl_total_corr(corr_bump)
    pending_n    = int((detected_zl['_status'] == 'pending').sum()) if len(detected_zl) else 0

    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("Detected", len(detected_zl))
    s2.metric("Pending",  pending_n)
    s3.metric("✅ Pos в окне",  f"{len(pos_w)} / {n_pos_total}")
    s4.metric("❌ Neg в окне",  f"{len(neg_w)} / {n_neg_total}")
    s5.metric("🔧 Corr в окне", f"{len(corr_w)} / {n_corr_total}")
    s6.metric("Model", "OK" if has_ml else "❌ нет")

    # ── MODE SELECTOR ─────────────────────────────────────────────────────────
    st.markdown("---")
    zl_mode = st.radio(
        "**Режим работы:**",
        ["📐 Markup (новая зона)", "🔧 Refine (правка границ)",
         "❌ Negative (мусор)", "🎯 Active Learning (uncertain)"],
        horizontal=True, key="zl_mode",
    )

    # ── MAIN CHART (общий для всех режимов) ──────────────────────────────────
    import plotly.graph_objects as go
    fig_zl = go.Figure()
    fig_zl.add_trace(go.Candlestick(
        x=df_zl['timestamp'], open=df_zl['open'], high=df_zl['high'],
        low=df_zl['low'], close=df_zl['close'],
        increasing_line_color='#00E676', decreasing_line_color='#FF1744', name='OHLC',
    ))

    # Labeled positives (синие solid)
    if len(pos_w):
        xs, ys = [], []
        for _, z in pos_w.iterrows():
            xs.extend([z['start_ts'], z['end_ts'], z['end_ts'],
                       z['start_ts'], z['start_ts'], None])
            ys.extend([z['range_low'], z['range_low'], z['range_high'],
                       z['range_high'], z['range_low'], None])
        fig_zl.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines',
            line=dict(color='rgba(100,150,255,0.95)', width=1.8),
            fill='toself', fillcolor='rgba(100,150,255,0.05)',
            hoverinfo='skip', showlegend=False, name='positive_labeled',
        ))

    # Detected zones по статусам (color-coded)
    if len(detected_zl) and '_status' in detected_zl.columns:
        for status_, color_, fill_ in [
            ('pending',   'rgba(255,200,0,0.85)',   'rgba(255,200,0,0.05)'),
            ('positive',  'rgba(0,230,118,0.95)',   'rgba(0,230,118,0.07)'),
            ('negative',  'rgba(255,50,50,0.95)',   'rgba(255,50,50,0.07)'),
            ('corrected', 'rgba(180,100,255,0.95)', 'rgba(180,100,255,0.07)'),
        ]:
            sub = detected_zl[detected_zl['_status'] == status_]
            if not len(sub): continue
            xs, ys = [], []
            for _, z in sub.iterrows():
                xs.extend([z['start_ts'], z['end_ts'], z['end_ts'],
                           z['start_ts'], z['start_ts'], None])
                ys.extend([z['range_low'], z['range_low'], z['range_high'],
                           z['range_high'], z['range_low'], None])
            fig_zl.add_trace(go.Scatter(
                x=xs, y=ys, mode='lines',
                line=dict(color=color_, width=1.4),
                fill='toself', fillcolor=fill_,
                hoverinfo='skip', showlegend=False, name=status_,
            ))

    # ── Chart rendering — режим Markup автоматически включает box-select ────
    is_markup_mode = zl_mode.startswith('📐')

    fig_zl.update_layout(
        height=620, template='plotly_dark',
        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
        xaxis_rangeslider_visible=False, showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),
        title=(f"{zl_symbol} {zl_tf} · 🟦 labeled positives  "
               f"🟡 pending  🟢 confirmed  🔴 rejected  🟣 corrected"),
        # uirevision: pan/zoom сохраняется пока symbol/tf/n_bars не меняется
        uirevision=f"zl_chart_{zl_symbol}_{zl_tf}_{zl_nbars}",
        # Markup mode → 'select' (box-select сразу активен, не надо включать в toolbar)
        # Другие режимы → 'pan' (привычное перетаскивание)
        dragmode=('select' if is_markup_mode else 'pan'),
        # Ограничиваем box-selection только осью X (по времени) — как в 17b
        selectdirection='h' if is_markup_mode else 'd',
    )
    fig_zl.update_xaxes(gridcolor='#1f2937')
    fig_zl.update_yaxes(gridcolor='#1f2937')

    # Markup mode: chart_key содержит pos_bump чтобы после Save новой зоны
    # полностью пересоздался chart (это очищает Plotly box-select rectangle).
    # Refine/Negative/AL: chart_key стабильный — traces обновляются через
    # _status (новые цвета зон), Plotly не дёргает полную пересборку.
    if is_markup_mode:
        _zl_chart_key = (
            f"zl_main_chart_{zl_symbol}_{zl_tf}_{zl_nbars}_M"
            f"_p{pos_bump}"
        )
    else:
        _zl_chart_key = (
            f"zl_main_chart_{zl_symbol}_{zl_tf}_{zl_nbars}_{zl_mode[:3]}"
        )
    chart_event = st.plotly_chart(
        fig_zl, use_container_width=True,
        key=_zl_chart_key,
        on_select=("rerun" if is_markup_mode else "ignore"),
        # selection_mode как строка (надёжнее чем list — как в 17b)
        selection_mode=("box" if is_markup_mode else None),
        config={
            'scrollZoom':       True,
            'displayModeBar':   True,
            'displaylogo':      False,
            'doubleClick':      False,
            'modeBarButtonsToRemove': ['lasso2d'] if is_markup_mode else ['lasso2d', 'select2d'],
        },
    )

    # ── ACTION PANEL (per mode) — M1 SKELETON, real handlers в следующих шагах ──
    st.markdown("---")

    if is_markup_mode:
        st.subheader("📐 Markup — выдели зону на графике box-select'ом")

        # ── Извлечение выделения из event с кешем в session_state ─────────
        zl_box_key = f"zl_box_{zl_symbol}_{zl_tf}"

        def _zl_extract_box(ev):
            try:
                sel = ev.get('selection') if isinstance(ev, dict) else getattr(ev, 'selection', None)
                if not sel: return None
                box = sel.get('box') if isinstance(sel, dict) else getattr(sel, 'box', None)
                if not box: return None
                b = box[0] if isinstance(box, list) else box
                x_arr = b.get('x') if isinstance(b, dict) else getattr(b, 'x', None)
                if not x_arr or len(x_arr) < 2: return None
                return [str(x) for x in x_arr]
            except Exception:
                return None

        # Fallback: после rerun chart_event может быть пустым, читаем из state
        _new_box = (_zl_extract_box(chart_event)
                    or _zl_extract_box(st.session_state.get(_zl_chart_key)))
        if _new_box is not None:
            st.session_state[zl_box_key] = _new_box
        _cached_box = st.session_state.get(zl_box_key)

        # Debug-панель (вспомогательная — если save не работает, видно что приходит)
        with st.expander("🔧 DEBUG (если save не работает)", expanded=False):
            st.write({
                'chart_event_type': type(chart_event).__name__,
                'chart_event_has_selection': bool(
                    (chart_event.get('selection') if isinstance(chart_event, dict)
                     else getattr(chart_event, 'selection', None))
                ),
                'state_chart_key': _zl_chart_key,
                'state_has_chart': _zl_chart_key in st.session_state,
                'extracted_box_new': _new_box,
                'cached_box': _cached_box,
            })
            if st.button("🗑 Сбросить cache box", key=f"zl_clr_box_{zl_symbol}_{zl_tf}"):
                for k in (zl_box_key, _zl_chart_key):
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()

        zl_ts0 = zl_ts1 = None
        zl_vp = zl_stats = None
        _parse_err = None
        if _cached_box is not None:
            try:
                _ts0 = pd.to_datetime(min(_cached_box), utc=True)
                _ts1 = pd.to_datetime(max(_cached_box), utc=True)
                zl_ts0 = max(_ts0, _t_min)
                zl_ts1 = min(_ts1, _t_max)
                _mask = (df_zl['timestamp'] >= zl_ts0) & (df_zl['timestamp'] <= zl_ts1)
                _df_range = df_zl[_mask].copy()
                if len(_df_range) >= 2:
                    zl_vp = compute_volume_profile(_df_range)
                    zl_stats = compute_zone_stats(_df_range, df_zl)
            except Exception as _e:
                _parse_err = repr(_e)

        if zl_vp and zl_stats:
            mm1, mm2, mm3, mm4, mm5 = st.columns(5)
            mm1.metric("Bars",  zl_stats['n_bars'])
            mm2.metric("Range", f"{zl_stats['range_low']:.4f} – {zl_stats['range_high']:.4f}")
            mm3.metric("ATR",   f"{zl_stats['atr_avg']:.4f}")
            mm4.metric("POC",   f"{zl_vp['poc']:.4f}")
            mm5.metric("VAH/VAL", f"{zl_vp['vah']:.4f} / {zl_vp['val']:.4f}")

            bcol1, bcol2 = st.columns([3, 1])
            if bcol1.button("💾 Save as POSITIVE zone (твой пример balance)",
                            type="primary", use_container_width=True,
                            key=f"zl_save_pos_{zl_symbol}_{zl_tf}"):
                try:
                    _before_n = len(load_saved_zones(symbol=zl_symbol, tf=zl_tf))
                    _zid = save_zone(zl_symbol, zl_tf, zl_ts0, zl_ts1, zl_vp, zl_stats)
                    _after_n = len(load_saved_zones(symbol=zl_symbol, tf=zl_tf))
                    _zl_bump('pos')
                    # Очистить ВСЁ что связано с box-selection (кеш + plotly state)
                    for _k in (zl_box_key, _zl_chart_key):
                        if _k in st.session_state:
                            del st.session_state[_k]
                    st.success(
                        f"✅ Saved positive id=`{_zid[:8]}` · "
                        f"для {zl_symbol} {zl_tf}: {_before_n} → {_after_n}"
                    )
                    st.rerun()
                except Exception as _se:
                    import traceback
                    st.error(f"❌ save_zone failed: {type(_se).__name__}: {_se}")
                    st.code(traceback.format_exc())

            if bcol2.button("🗑 Сбросить выделение", use_container_width=True,
                            key=f"zl_clear_box_{zl_symbol}_{zl_tf}"):
                if zl_box_key in st.session_state:
                    del st.session_state[zl_box_key]
                st.rerun()
        else:
            if _cached_box and _parse_err:
                st.warning(f"Box есть, но parse failed: {_parse_err}")
            elif _cached_box and not zl_vp:
                st.warning("Box есть, но VP не строится (выделение <2 баров?). Расширь рамку.")
            else:
                st.info("👆 В toolbar графика включи **Box Select** (квадратик) → "
                        "тяни рамку от свечи A до свечи B → "
                        "тут появятся метрики и кнопка Save.")

    elif zl_mode.startswith('🔧'):
        st.subheader("🔧 Refine — двигай границы detector-зон")

        if not has_ml:
            st.warning("Модель не обучена. Refine недоступен.")
        elif not len(detected_zl) or '_status' not in detected_zl.columns:
            st.info("Нет detected зон в этом окне (поправь Min proba/density сверху).")
        else:
            pending_zl = detected_zl[detected_zl['_status'] == 'pending'].reset_index(drop=True)
            if not len(pending_zl):
                st.success(f"✅ Все {len(detected_zl)} detected зон в окне обработаны.")
            else:
                # ── Current zone selection ───────────────────────────────────
                _rf_idx_key = f"zl_rf_idx_{zl_symbol}_{zl_tf}"
                if _rf_idx_key not in st.session_state:
                    st.session_state[_rf_idx_key] = 0
                cur_i = min(int(st.session_state[_rf_idx_key]), len(pending_zl) - 1)
                cur_z = pending_zl.iloc[cur_i]
                cur_zid = str(cur_z['id'])

                _bars_ts = df_zl['timestamp'].values
                _cur_start = pd.to_datetime(cur_z['start_ts'], utc=True)
                _cur_end   = pd.to_datetime(cur_z['end_ts'], utc=True)
                orig_l = int(np.argmin(np.abs(_bars_ts - np.datetime64(_cur_start))))
                orig_r = int(np.argmin(np.abs(_bars_ts - np.datetime64(_cur_end))))

                # ── Navigation ───────────────────────────────────────────────
                nv1, nv2, nv3, nv4 = st.columns([1, 4, 1, 1])
                with nv1:
                    if st.button("⏮ Prev", disabled=cur_i == 0,
                                 use_container_width=True, key=f"zl_rf_prev"):
                        st.session_state[_rf_idx_key] = max(0, cur_i - 1)
                        st.rerun()
                nv2.markdown(
                    f"**Zone {cur_i + 1}/{len(pending_zl)}** · "
                    f"`{cur_zid[:6]}` · POC `{cur_z['poc']:.2f}` · "
                    f"{int(cur_z['n_bars'])} bars · "
                    f"{pd.Timestamp(_cur_start):%Y-%m-%d %H:%M} → {pd.Timestamp(_cur_end):%H:%M}"
                )
                with nv3:
                    if st.button("⏭ Next", disabled=cur_i >= len(pending_zl) - 1,
                                 use_container_width=True, key=f"zl_rf_next"):
                        st.session_state[_rf_idx_key] = min(len(pending_zl) - 1, cur_i + 1)
                        st.rerun()
                with nv4:
                    if st.button("🔄 Reset", use_container_width=True, key=f"zl_rf_reset"):
                        for k in [f"zl_rf_ls_{cur_zid}", f"zl_rf_ln_{cur_zid}",
                                  f"zl_rf_rs_{cur_zid}", f"zl_rf_rn_{cur_zid}"]:
                            if k in st.session_state:
                                del st.session_state[k]
                        st.rerun()

                # ── Sliders: LEFT + RIGHT с number_input для точности ───────
                _l_min, _l_max = 0, len(df_zl) - 2
                _r_min, _r_max = 1, len(df_zl) - 1

                ls_k = f"zl_rf_ls_{cur_zid}"
                ln_k = f"zl_rf_ln_{cur_zid}"
                rs_k = f"zl_rf_rs_{cur_zid}"
                rn_k = f"zl_rf_rn_{cur_zid}"

                if ls_k not in st.session_state:
                    st.session_state[ls_k] = orig_l
                    st.session_state[ln_k] = orig_l
                if rs_k not in st.session_state:
                    st.session_state[rs_k] = orig_r
                    st.session_state[rn_k] = orig_r

                # Кламп
                st.session_state[ls_k] = int(np.clip(st.session_state[ls_k], _l_min, _l_max))
                st.session_state[ln_k] = int(np.clip(st.session_state[ln_k], _l_min, _l_max))
                st.session_state[rs_k] = int(np.clip(st.session_state[rs_k], _r_min, _r_max))
                st.session_state[rn_k] = int(np.clip(st.session_state[rn_k], _r_min, _r_max))

                def _sync_ls():
                    st.session_state[ln_k] = int(st.session_state[ls_k])
                def _sync_ln():
                    st.session_state[ls_k] = int(np.clip(st.session_state[ln_k], _l_min, _l_max))
                def _sync_rs():
                    st.session_state[rn_k] = int(st.session_state[rs_k])
                def _sync_rn():
                    st.session_state[rs_k] = int(np.clip(st.session_state[rn_k], _r_min, _r_max))

                sl1, sl2 = st.columns(2)
                with sl1:
                    cL1, cL2 = st.columns([4, 1])
                    cL1.slider("◀ LEFT (bar)", _l_min, _l_max, step=1,
                               key=ls_k, on_change=_sync_ls)
                    cL2.number_input("точно", _l_min, _l_max, step=1,
                                     key=ln_k, on_change=_sync_ln,
                                     label_visibility='collapsed')
                with sl2:
                    cR1, cR2 = st.columns([4, 1])
                    cR1.slider("RIGHT ▶ (bar)", _r_min, _r_max, step=1,
                               key=rs_k, on_change=_sync_rs)
                    cR2.number_input("точно", _r_min, _r_max, step=1,
                                     key=rn_k, on_change=_sync_rn,
                                     label_visibility='collapsed')

                new_l = int(st.session_state[ls_k])
                new_r = int(st.session_state[rs_k])
                if new_r <= new_l:
                    st.warning("RIGHT должна быть правее LEFT")
                    new_r = min(_r_max, new_l + 1)

                new_ts0 = pd.to_datetime(df_zl['timestamp'].iloc[new_l], utc=True)
                new_ts1 = pd.to_datetime(df_zl['timestamp'].iloc[new_r], utc=True)

                _df_range_new = df_zl.iloc[new_l:new_r + 1].copy()
                new_vp = compute_volume_profile(_df_range_new) if len(_df_range_new) >= 2 else None
                new_stats = compute_zone_stats(_df_range_new, df_zl) if new_vp else {}

                delta_s = new_l - orig_l
                delta_e = new_r - orig_r

                rm1, rm2, rm3, rm4, rm5 = st.columns(5)
                rm1.metric("Δ start", f"{delta_s:+d}")
                rm2.metric("Δ end",   f"{delta_e:+d}")
                rm3.metric("n_bars",  f"{new_r - new_l + 1}",
                           delta=f"{(new_r - new_l + 1) - int(cur_z['n_bars']):+d}")
                if new_vp:
                    rm4.metric("POC", f"{new_vp['poc']:.2f}",
                               delta=f"{new_vp['poc'] - float(cur_z['poc']):+.2f}")
                    rm5.metric("range",
                               f"{new_vp['range_low']:.0f}–{new_vp['range_high']:.0f}")

                # ── Mini-preview chart (focused) ──────────────────────────
                import plotly.graph_objects as go
                _zoom_l = max(0, min(orig_l, new_l) - 40)
                _zoom_r = min(len(df_zl) - 1, max(orig_r, new_r) + 40)
                fig_prev = go.Figure()
                fig_prev.add_trace(go.Candlestick(
                    x=df_zl['timestamp'], open=df_zl['open'], high=df_zl['high'],
                    low=df_zl['low'], close=df_zl['close'],
                    increasing_line_color='#00E676', decreasing_line_color='#FF1744',
                ))
                # Original border (grey dashed)
                fig_prev.add_shape(type='rect',
                                   x0=cur_z['start_ts'], x1=cur_z['end_ts'],
                                   y0=cur_z['range_low'], y1=cur_z['range_high'],
                                   line=dict(color='rgba(180,180,180,0.7)', width=1, dash='dot'),
                                   fillcolor='rgba(0,0,0,0)', layer='above')
                # New zone (golden bright)
                if new_vp:
                    fig_prev.add_shape(type='rect',
                                       x0=new_ts0, x1=new_ts1,
                                       y0=new_vp['range_low'], y1=new_vp['range_high'],
                                       line=dict(color='rgba(255,200,0,0.95)', width=2.5),
                                       fillcolor='rgba(255,200,0,0.08)', layer='above')
                    for lvl, c, dash in [
                        (new_vp['poc'], 'rgba(255,220,0,0.95)', 'solid'),
                        (new_vp['vah'], 'rgba(255,150,150,0.8)', 'dash'),
                        (new_vp['val'], 'rgba(150,255,150,0.8)', 'dash'),
                    ]:
                        fig_prev.add_shape(type='line',
                                           x0=new_ts0, x1=new_ts1, y0=lvl, y1=lvl,
                                           line=dict(color=c, width=1.5, dash=dash),
                                           layer='above')

                _preview_zoom_key = f"zl_rf_prev_zoom_{cur_zid}"
                _preview_kwargs = {}
                if _preview_zoom_key not in st.session_state:
                    _preview_kwargs['xaxis_range'] = [
                        df_zl['timestamp'].iloc[_zoom_l],
                        df_zl['timestamp'].iloc[_zoom_r],
                    ]
                    st.session_state[_preview_zoom_key] = True

                fig_prev.update_layout(
                    height=420, template='plotly_dark',
                    paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                    xaxis_rangeslider_visible=False, showlegend=False,
                    margin=dict(l=10, r=10, t=20, b=10),
                    uirevision=cur_zid,
                    dragmode='pan',
                    **_preview_kwargs,
                )
                fig_prev.update_xaxes(gridcolor='#1f2937')
                fig_prev.update_yaxes(gridcolor='#1f2937')
                st.plotly_chart(fig_prev, use_container_width=True,
                                key=f"zl_rf_prev_{cur_zid}",
                                config={'scrollZoom': True, 'displayModeBar': True,
                                        'displaylogo': False, 'doubleClick': False,
                                        'modeBarButtonsToRemove': ['lasso2d', 'select2d']})

                # ── Verdict buttons ──────────────────────────────────────
                def _orig_vp(z):
                    return {'poc': float(z['poc']), 'vah': float(z['vah']),
                            'val': float(z['val']), 'va_width': float(z['va_width']),
                            'total_volume': float(z['total_volume']),
                            'hvn_levels': [], 'lvn_levels': [],
                            'range_high': float(z['range_high']),
                            'range_low': float(z['range_low'])}
                def _orig_st(z):
                    return {'n_bars': int(z['n_bars']),
                            'range_high': float(z['range_high']),
                            'range_low': float(z['range_low']),
                            'range_width': float(z['range_width']),
                            'atr_avg': float(z['atr_avg']),
                            'swing_touches_high': int(z['swing_touches_high']),
                            'swing_touches_low': int(z['swing_touches_low']),
                            'swing_touches_total': int(z['swing_touches_total'])}

                bn1, bn2, bn3, bn4 = st.columns(4)
                if bn1.button("✅ Save corrected", type="primary",
                              use_container_width=True, key="zl_rf_save",
                              disabled=(new_vp is None)):
                    save_corrected_zone(cur_z.to_dict(), new_ts0, new_ts1,
                                         new_vp, new_stats, verdict='corrected')
                    _zl_bump('corr')
                    st.rerun()
                if bn2.button("✓ Already correct", use_container_width=True, key="zl_rf_alr"):
                    save_corrected_zone(cur_z.to_dict(),
                                         pd.to_datetime(cur_z['start_ts'], utc=True),
                                         pd.to_datetime(cur_z['end_ts'], utc=True),
                                         _orig_vp(cur_z), _orig_st(cur_z),
                                         verdict='already_correct')
                    _zl_bump('corr')
                    st.rerun()
                if bn3.button("❌ Not a zone", use_container_width=True, key="zl_rf_no"):
                    save_corrected_zone(cur_z.to_dict(),
                                         pd.to_datetime(cur_z['start_ts'], utc=True),
                                         pd.to_datetime(cur_z['end_ts'], utc=True),
                                         _orig_vp(cur_z), _orig_st(cur_z),
                                         verdict='not_a_zone')
                    _zl_bump('corr')
                    st.rerun()
                if bn4.button("⏭ Skip", use_container_width=True, key="zl_rf_sk"):
                    st.session_state[_rf_idx_key] = min(len(pending_zl) - 1, cur_i + 1)
                    st.rerun()

                # ── Verdicts log (undo последних 10) ─────────────────────
                if len(corr_w):
                    with st.expander(f"📕 Verdicts log в окне ({len(corr_w)}) — undo"):
                        for _, c in corr_w.tail(10).iloc[::-1].iterrows():
                            cl1, cl2, cl3, cl4, cl5 = st.columns([2, 1, 1, 1, 1])
                            cl1.caption(f"`{str(c['original_id'])[:6]}` · **{c['verdict']}** · "
                                        f"{pd.Timestamp(c['corrected_at']):%H:%M:%S}")
                            cl2.caption(f"Δs:{int(c['delta_start_bars']):+d}")
                            cl3.caption(f"Δe:{int(c['delta_end_bars']):+d}")
                            cl4.caption(f"IoU:{c['iou_with_original']:.2f}")
                            if cl5.button("↩ undo", key=f"zl_rf_undo_{c['id']}"):
                                delete_corrected_zone(c['id'])
                                _zl_bump('corr')
                                st.rerun()

    elif zl_mode.startswith('❌'):
        st.subheader("❌ Negative — быстрая разметка pending зон детектора")

        if not has_ml:
            st.warning("Модель не обучена. Negative режим недоступен.")
        elif not len(detected_zl) or '_status' not in detected_zl.columns:
            st.info("Нет detected зон в окне.")
        else:
            pending_zl_neg = detected_zl[detected_zl['_status'] == 'pending'] \
                .sort_values('start_ts').reset_index(drop=True)
            if not len(pending_zl_neg):
                st.success(f"✅ Все {len(detected_zl)} зон в окне уже размечены.")
            else:
                st.markdown(f"**🟡 Pending: {len(pending_zl_neg)}** — пройдись и каждой "
                            "поставь вердикт. Зоны помеченные ❌ — false-positive для "
                            "обучения модели (hard-negative mining, weight=5×).")

                for _idx, _z in pending_zl_neg.iterrows():
                    nz_id = str(_z['id'])
                    cz1, cz2, cz3, cz4, cz5, cz6 = st.columns([2, 1, 1, 1, 1, 1])
                    cz1.caption(f"`{nz_id[:6]}` · "
                                f"{pd.Timestamp(_z['start_ts']):%Y-%m-%d %H:%M} → "
                                f"{pd.Timestamp(_z['end_ts']):%H:%M}")
                    cz2.caption(f"{int(_z['n_bars'])} bars")
                    cz3.caption(f"POC {_z['poc']:.2f}")
                    cz4.caption(f"H/L: {_z['range_high']:.0f}/{_z['range_low']:.0f}")
                    if cz5.button("❌ Not zone", key=f"zl_neg_no_{nz_id}",
                                  use_container_width=True):
                        save_detected_as_negative(_z.to_dict())
                        _zl_bump('neg')
                        st.rerun()
                    if cz6.button("✓ Confirm", key=f"zl_neg_yes_{nz_id}",
                                  use_container_width=True, type="primary"):
                        # Сохраняем как positive — детектор-зона подтверждена пользователем
                        _vp_like = {'poc': _z['poc'], 'vah': _z['vah'], 'val': _z['val'],
                                    'va_width': _z['va_width'],
                                    'hvn_levels': [], 'lvn_levels': [],
                                    'total_volume': _z['total_volume'],
                                    'range_high': _z['range_high'],
                                    'range_low': _z['range_low']}
                        _st_like = {'n_bars': int(_z['n_bars']),
                                    'range_high': float(_z['range_high']),
                                    'range_low': float(_z['range_low']),
                                    'range_width': float(_z['range_width']),
                                    'atr_avg': float(_z['atr_avg']),
                                    'swing_touches_high': int(_z['swing_touches_high']),
                                    'swing_touches_low': int(_z['swing_touches_low']),
                                    'swing_touches_total': int(_z['swing_touches_total'])}
                        save_zone(zl_symbol, zl_tf,
                                  _z['start_ts'], _z['end_ts'],
                                  _vp_like, _st_like)
                        _zl_bump('pos')
                        st.rerun()

                # Undo последних negatives в окне
                if len(neg_w):
                    with st.expander(f"📕 Уже помеченные ❌ в окне ({len(neg_w)}) — undo"):
                        for _, _zn in neg_w.tail(15).iloc[::-1].iterrows():
                            ncl1, ncl2, ncl3 = st.columns([3, 1, 1])
                            ncl1.caption(f"`{str(_zn['id'])[:6]}` · "
                                         f"{pd.Timestamp(_zn['start_ts']):%Y-%m-%d %H:%M} → "
                                         f"{pd.Timestamp(_zn['end_ts']):%H:%M} · "
                                         f"{int(_zn['n_bars'])} bars")
                            ncl2.caption(f"POC {_zn['poc']:.2f}")
                            if ncl3.button("↩ undo", key=f"zl_neg_undo_{_zn['id']}"):
                                delete_negative_zone(_zn['id'])
                                _zl_bump('neg')
                                st.rerun()

    elif zl_mode.startswith('🎯'):
        st.subheader("🎯 Active Learning — модель показывает наименее уверенные кандидаты")
        st.caption(
            "Каждая разметка тут даёт ×5-10 больше информации модели чем случайная "
            "(maximum information gain). 20 разметок здесь ≈ 100-200 случайных."
        )

        if not has_ml:
            st.warning("Модель не обучена. Active Learning недоступен.")
        elif not len(detected_zl) or 'proba' not in detected_zl.columns:
            st.info("Нет detected зон с proba. Перезагрузи (Min proba снизь до 0.10 для большего пула).")
        else:
            # Uncertainty = расстояние от 0.5. Берём ТОЛЬКО pending (без разметки).
            pending_al = detected_zl[detected_zl['_status'] == 'pending'].copy()
            if not len(pending_al):
                st.success(f"✅ Все {len(detected_zl)} зон в окне обработаны.")
            else:
                pending_al['_uncertainty'] = (pending_al['proba'] - 0.5).abs()
                pending_al = pending_al.sort_values('_uncertainty').reset_index(drop=True)
                top_n = min(20, len(pending_al))
                st.markdown(f"**Топ-{top_n} наименее уверенных** (proba ближе к 0.5 = "
                            f"модель не знает 'balance или нет'):")

                for _idx, _z in pending_al.head(top_n).iterrows():
                    al_id = str(_z['id'])
                    cz1, cz2, cz3, cz4, cz5, cz6, cz7 = st.columns([2, 1, 1, 1, 1, 1, 1])
                    cz1.caption(f"`{al_id[:6]}` · "
                                f"{pd.Timestamp(_z['start_ts']):%Y-%m-%d %H:%M} → "
                                f"{pd.Timestamp(_z['end_ts']):%H:%M}")
                    cz2.caption(f"{int(_z['n_bars'])} bars")
                    cz3.caption(f"POC {_z['poc']:.2f}")
                    # Uncertainty bar visual
                    cz4.caption(f"proba **{_z['proba']:.2f}**")
                    cz5.caption(f"unc {_z['_uncertainty']:.3f}")
                    if cz6.button("❌", key=f"zl_al_no_{al_id}",
                                  use_container_width=True, help="Not a zone"):
                        save_detected_as_negative(_z.to_dict())
                        _zl_bump('neg')
                        st.rerun()
                    if cz7.button("✓", key=f"zl_al_yes_{al_id}",
                                  use_container_width=True, type="primary",
                                  help="Confirm as balance zone"):
                        _vp_l = {'poc': _z['poc'], 'vah': _z['vah'], 'val': _z['val'],
                                 'va_width': _z['va_width'],
                                 'hvn_levels': [], 'lvn_levels': [],
                                 'total_volume': _z['total_volume'],
                                 'range_high': _z['range_high'],
                                 'range_low': _z['range_low']}
                        _st_l = {'n_bars': int(_z['n_bars']),
                                 'range_high': float(_z['range_high']),
                                 'range_low': float(_z['range_low']),
                                 'range_width': float(_z['range_width']),
                                 'atr_avg': float(_z['atr_avg']),
                                 'swing_touches_high': int(_z['swing_touches_high']),
                                 'swing_touches_low': int(_z['swing_touches_low']),
                                 'swing_touches_total': int(_z['swing_touches_total'])}
                        save_zone(zl_symbol, zl_tf,
                                  _z['start_ts'], _z['end_ts'], _vp_l, _st_l)
                        _zl_bump('pos')
                        st.rerun()

                st.caption(f"📊 Распределение proba pending зон: "
                           f"min={pending_al['proba'].min():.2f} · "
                           f"med={pending_al['proba'].median():.2f} · "
                           f"max={pending_al['proba'].max():.2f}")

    # ── TRAINING PANEL (общий для всех режимов) ─────────────────────────────
    st.markdown("---")
    with st.expander("🔄 **Training panel** — переобучить модель на свежей разметке",
                     expanded=False):
        from balance_zone_markup import (
            ZONES_PARQUET, NEGATIVES_PARQUET, CORRECTED_PARQUET,
        )
        import json
        from datetime import datetime as _dt

        tp1, tp2, tp3, tp4 = st.columns(4)
        tp1.metric("✅ Positives", n_pos_total)
        tp2.metric("❌ Negatives", n_neg_total)
        tp3.metric("🔧 Corrected", n_corr_total)
        if ML_MODEL_PATH.exists():
            _model_mtime = _dt.fromtimestamp(ML_MODEL_PATH.stat().st_mtime)
            tp4.metric("Model trained", f"{_model_mtime:%d.%m %H:%M}")
        else:
            tp4.metric("Model", "❌ no model")

        st.markdown(
            "Запускает `python train_zone_classifier.py`. Использует:\n"
            f"- `{ZONES_PARQUET.name}` ({n_pos_total} positives)\n"
            f"- `{NEGATIVES_PARQUET.name}` ({n_neg_total} hand-negatives, weight=5×)\n"
            f"- + random negatives с IoU<0.10 (weight=1×)\n"
            "После обучения автоматически обновится модель и detected зоны."
        )

        if st.button("🚀 Retrain model NOW",
                     type="primary", key="zl_retrain_btn"):
            import subprocess, os
            _train_script = Path(__file__).parent / 'train_zone_classifier.py'
            if not _train_script.exists():
                st.error(f"❌ Не найден {_train_script}")
            else:
                with st.spinner("Тренировка модели… (30-90 сек)"):
                    try:
                        _env = {**os.environ,
                                'PYTHONIOENCODING': 'utf-8',
                                'PYTHONUTF8': '1'}
                        _result = subprocess.run(
                            [str(Path(sys.executable)), str(_train_script)],
                            cwd=str(Path(__file__).parent),
                            capture_output=True, text=True, timeout=180,
                            env=_env, encoding='utf-8', errors='replace',
                        )
                    except subprocess.TimeoutExpired:
                        st.error("❌ Timeout (>180s)")
                        _result = None
                    except Exception as _re:
                        st.error(f"❌ {type(_re).__name__}: {_re}")
                        _result = None

                if _result is not None:
                    if _result.returncode == 0:
                        st.success("✅ Модель переобучена и сохранена!")
                        # Bump det cache → новые зоны при следующем рендере
                        st.session_state['zl_det_bump'] = \
                            st.session_state.get('zl_det_bump', 0) + 1
                        with st.expander("📄 Лог обучения", expanded=True):
                            st.code(_result.stdout[-3000:] if _result.stdout else '(empty)')
                        st.rerun()
                    else:
                        st.error(f"❌ Скрипт упал с кодом {_result.returncode}")
                        with st.expander("📄 stderr", expanded=True):
                            st.code(_result.stderr[-3000:] if _result.stderr else '(empty)')
                        if _result.stdout:
                            with st.expander("📄 stdout"):
                                st.code(_result.stdout[-3000:])

    st.stop()

asset, is_renko, src = "BTCUSDT", "Renko" in tab, "crypto"
if tab not in _LAB_TABS:
    if   "BTC"   in tab: asset = "BTCUSDT"
    elif "Альты" in tab:
        asset = st.selectbox("Выберите актив:", LIST_ALTS_MAIN)
    elif "Набор 2" in tab:
        asset = st.selectbox("Выберите актив:", LIST_ALTS_2)
    elif "Набор 3" in tab:
        asset = st.selectbox("Выберите актив:", LIST_ALTS_3)
    elif "Золото" in tab: asset, src = "GC=F",     "forex"
    elif "EUR"    in tab: asset, src = "EURUSD=X", "forex"
    elif "GBP"    in tab: asset, src = "GBPUSD=X", "forex"

# ─────────────────────────────────────────────────────────────────────────────
# 9. СКРИНЕР СИГНАЛОВ + ЖУРНАЛ СДЕЛОК
# ─────────────────────────────────────────────────────────────────────────────
import json, os, base64
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# ГРУППА 3: ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ─────────────────────────────────────────────────────────────────────────────

def take_trade_snapshot(fig, trade_id):
    """
    Сохраняет Plotly фигуру как PNG для Vision AI анализа.
    Требует: pip install kaleido
    Fallback: сохраняет HTML если kaleido недоступен.
    """
    snap_dir = "/opt/myscreener/snapshots"
    os.makedirs(snap_dir, exist_ok=True)
    filepath_png  = f"{snap_dir}/trade_{trade_id}.png"
    filepath_html = f"{snap_dir}/trade_{trade_id}.html"

    try:
        fig.write_image(filepath_png, width=1920, height=1080, scale=2)
        return filepath_png
    except Exception:
        # Fallback — kaleido не установлен
        try:
            fig.write_html(filepath_html)
            return filepath_html
        except:
            return None


def analyze_trade_with_claude(screenshot_path, trade_data, user_comment):
    """
    Vision AI анализ через Anthropic Claude API.
    Отправляет скриншот графика + данные сделки → получает структурированные фичи.

    Возвращает dict с фичами для Random Forest:
    {
        trend_direction: UP/DOWN/BALANCE,
        is_at_key_level: bool,
        is_inside_ftr: bool,
        cvd_divergence_type: ABSORPTION_BULL/ABSORPTION_BEAR/EXHAUSTION_BULL/EXHAUSTION_BEAR/NONE,
        dpoc_position: ABOVE/BELOW/AT,
        liquidity_sweep: bool,
        trade_logic_rating: 1-10,
        ai_comment: str
    }
    """
    if not screenshot_path or not os.path.exists(screenshot_path):
        return None

    # Читаем картинку только если это PNG
    if not screenshot_path.endswith('.png'):
        return None

    try:
        with open(screenshot_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode('utf-8')
    except:
        return None

    symbol   = trade_data.get('symbol', '')
    entry    = trade_data.get('entry', 0)
    sl       = trade_data.get('sl', 0)
    tp       = trade_data.get('tp', 0)
    tf       = trade_data.get('tf', '15m')
    direction = trade_data.get('direction', '')

    prompt = (
        f"Ты профессиональный SMC трейдер-аналитик. Анализируй скриншот графика.\n\n"
        f"Данные сделки: {symbol} | {tf} | {direction}\n"
        f"Вход: {entry} | Стоп: {sl} | Тейк: {tp}\n"
        f"Комментарий трейдера: \"{user_comment}\"\n\n"
        f"На графике: свечи, CVD (кумулятивная дельта), Volume, FTR зоны (зелёные=Demand, красные=Supply), "
        f"жёлтая линия=dPOC, синие линии=VAH/VAL баланса, BOS метки.\n\n"
        f"Верни ТОЛЬКО валидный JSON без markdown:\n"
        f"{{\n"
        f"  \"trend_direction\": \"UP\"|\"DOWN\"|\"BALANCE\",\n"
        f"  \"is_at_key_level\": true|false,\n"
        f"  \"is_inside_ftr\": true|false,\n"
        f"  \"cvd_divergence_type\": \"ABSORPTION_BULL\"|\"ABSORPTION_BEAR\"|\"EXHAUSTION_BULL\"|\"EXHAUSTION_BEAR\"|\"NONE\",\n"
        f"  \"dpoc_position\": \"ABOVE\"|\"BELOW\"|\"AT\",\n"
        f"  \"liquidity_sweep\": true|false,\n"
        f"  \"trade_logic_rating\": <1-10>,\n"
        f"  \"ai_comment\": \"<краткий вывод по входу>\"\n"
        f"}}"
    )

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 500,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }]
            },
            timeout=30
        )
        raw = resp.json()['content'][0]['text']
        clean = raw.replace('```json','').replace('```','').strip()
        return json.loads(clean)
    except Exception as e:
        print(f"[VISION AI] Ошибка: {e}")
        return None


def collect_smc_snapshot(snap_df, snap_htf, t_symbol, t_tf, t_entry, t_sl, t_tp):
    """
    Собирает SMC фичи для снимка сделки — НОВЫЕ признаки из Групп 1+2.
    Дополняет старый market_snapshot.
    """
    smc = {}
    try:
        lc = float(snap_df['close'].iloc[-1])
        atr_s = float((snap_df['high'] - snap_df['low']).rolling(14).mean().iloc[-1])

        # 1. Структурный анализ (новые функции Группы 2)
        (in_bal, t_dir, key_high, key_low,
         vah, val, poc, bko, structure, kh_list, kl_list) = get_struct_levels(snap_df, tf=t_tf, sw=3)

        phase = 'BALANCE' if in_bal else ('TREND_UP' if t_dir == 1 else 'TREND_DOWN')
        smc['smc_phase']     = phase
        smc['key_high']      = round(float(key_high), 6) if not np.isnan(key_high) else None
        smc['key_low']       = round(float(key_low),  6) if not np.isnan(key_low)  else None

        # 2. Свип ликвидности (детекторы Группы 2)
        smc['sweep_key_high'] = bool(check_sweep(snap_df, key_high, 'bearish'))
        smc['sweep_key_low']  = bool(check_sweep(snap_df, key_low,  'bullish'))

        # 3. FTR — в зоне или нет (новые детекторы)
        zns = calc_ftr_zones(snap_df, **get_ftr_params(t_symbol))
        smc['in_demand_ftr'] = bool(check_price_in_demand_ftr(snap_df, zns, key_low))
        smc['in_supply_ftr'] = bool(check_price_in_supply_ftr(snap_df, zns, key_high))

        # 4. CVD дивергенция — тип (новый детектор)
        smc['cvd_div_type'] = check_cvd_divergence(snap_df) or 'NONE'

        # 5. dPOC позиция (якорный dPOC)
        dpoc_arr = calc_dynamic_poc(snap_df, tf=t_tf)
        dpoc_last = float(dpoc_arr[-1]) if not np.isnan(dpoc_arr[-1]) else lc
        if lc > dpoc_last * 1.002:
            smc['dpoc_position'] = 'ABOVE'
        elif lc < dpoc_last * 0.998:
            smc['dpoc_position'] = 'BELOW'
        else:
            smc['dpoc_position'] = 'AT'
        smc['dpoc_value'] = round(dpoc_last, 6)

        # 6. Breakout Retest детектор
        smc['is_bos_retest_long']  = bool(check_pullback_to_broken_key_high(snap_df, key_high))
        smc['is_bos_retest_short'] = bool(check_pullback_to_broken_key_low(snap_df, key_low))

        # 7. Структура рынка на HTF (если передан)
        if snap_htf is not None and not snap_htf.empty:
            (htf_bal, htf_dir, htf_kh, htf_kl, *_) = get_struct_levels(snap_htf, tf='1h', sw=3)
            smc['htf_phase_smc'] = 'BALANCE' if htf_bal else ('TREND_UP' if htf_dir==1 else 'TREND_DOWN')
        else:
            smc['htf_phase_smc'] = 'UNKNOWN'

    except Exception as e:
        smc['smc_error'] = str(e)

    return smc




TRADES_FILE = "trades.json"

def load_trades():
    if os.path.exists(TRADES_FILE):
        try:
            return json.load(open(TRADES_FILE, encoding='utf-8'))
        except: pass
    return []

def save_trades(trades):
    json.dump(trades, open(TRADES_FILE, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)

def scan_signal(symbol, tf_scan, src_scan):
    """
    v7: упрощённая логика — 3 типа сетапов, нет двойной фильтрации.
    Тип 1 — Reversal (зона + подтверждение)
    Тип 2 — Continuation (тренд + откат к уровню)
    Тип 3 — Liquidity Grab (sweep + возврат)
    Условие входа: HTF контекст + зона/уровень + 1 сильный триггер
    """
    try:
        d = fetch_main_data(symbol, tf_scan, False, src_scan)
        if d.empty or len(d) < 60:
            return None
        d = apply_order_flow(d)

        htf_tf = "1h" if tf_scan == "15m" else "1D"
        d_htf  = fetch_main_data(symbol, htf_tf, False, src_scan)
        if d_htf.empty or len(d_htf) < 30:
            return None
        d_htf = apply_order_flow(d_htf)

        # ── Структурный анализ HTF
        (in_bal_htf, t_dir_htf,
         key_high_htf, key_low_htf,
         vah_htf, val_htf, poc_htf,
         breakout_htf, structure_htf,
         kh_htf, kl_htf) = get_struct_levels(d_htf, tf=htf_tf, sw=3)

        ALTS_MAIN = ["XRPUSDT","XLMUSDT","SOLUSDT","TRXUSDT","BCHUSDT","LTCUSDT",
                     "AAVEUSDT","BNBUSDT","AVAXUSDT","ETCUSDT","LINKUSDT","DOGEUSDT",
                     "NEARUSDT","ATOMUSDT","UNIUSDT","POLUSDT","ADAUSDT","DOTUSDT",
                     "ETHUSDT","OPUSDT","FILUSDT","ARBUSDT"]
        ALTS_2 = ["CAKEUSDT","WOOUSDT","VETUSDT","ZILUSDT","STXUSDT","WAVESUSDT",
                  "ZRXUSDT","XTZUSDT","JASMYUSDT","ORDIUSDT","TONUSDT","INJUSDT",
                  "SEIUSDT","PEOPLEUSDT","FLOKIUSDT","YGGUSDT","GALAUSDT","RUNEUSDT",
                  "CRVUSDT","ICPUSDT","MASKUSDT","APEUSDT","SUIUSDT","APTUSDT",
                  "MANAUSDT","THETAUSDT","GMTUSDT","HBARUSDT","SANDUSDT","KSMUSDT"]
        zns = calc_ftr_zones(d, **get_ftr_params(symbol))

        last_close = float(d['close'].iloc[-1])
        last_open  = float(d['open'].iloc[-1])
        atr_val    = float((d['high'] - d['low']).rolling(14).mean().iloc[-1])

        # ── Структурный анализ LTF
        (in_bal, t_dir,
         key_high_ltf, key_low_ltf,
         vah, val, cur_poc,
         mkt_mode_struct, structure_ltf,
         kh_ltf, kl_ltf) = get_struct_levels(d, tf=tf_scan, sw=3)

        mkt_mode = mkt_mode_struct  # "UP"/"DOWN"/None
        bal_high = key_high_ltf if not np.isnan(key_high_ltf) else np.nan
        bal_low  = key_low_ltf  if not np.isnan(key_low_ltf)  else np.nan

        _all_prof = build_balance_profiles(d, tf_scan, max_balances=10)
        hist_pocs = [{'x0': p['x0'], 'x1': p['x1'], 'poc': p['poc']}
                     for p in _all_prof if not p.get('is_current', False)]

        # ── HTF контекст через СТРУКТУРУ (Key High / Key Low)
        if in_bal_htf:
            htf_context = "BALANCE"
            kh = key_high_htf if not np.isnan(key_high_htf) else vah_htf
            kl = key_low_htf  if not np.isnan(key_low_htf)  else val_htf
            range_size = kh - kl if (not np.isnan(kh) and not np.isnan(kl) and kh > kl) else 0
            price_pos  = (last_close - kl) / range_size if range_size > 0 else 0.5
            if price_pos > 0.75:   trade_dir = -1
            elif price_pos < 0.25: trade_dir = 1
            else:                  trade_dir = None
        else:
            htf_context = "TREND"
            trade_dir   = t_dir_htf

        # ── Базовые данные для триггеров
        abs_bull_r   = d['abs_bull'].iloc[-5:].any()
        abs_bear_r   = d['abs_bear'].iloc[-5:].any()
        dshift_bull  = d['delta_shift_bull'].iloc[-3:].any()
        dshift_bear  = d['delta_shift_bear'].iloc[-3:].any()
        dp           = float(d['delta_pressure'].iloc[-1])
        sweep_low    = float(d['low'].iloc[-1])  < float(d['low'].rolling(20).min().iloc[-2])
        sweep_high   = float(d['high'].iloc[-1]) > float(d['high'].rolling(20).max().iloc[-2])
        divs         = find_divergences(d)
        has_bull_div = any(dv['type']=='bull' and
                           len(d)-np.searchsorted(d['timestamp'].values,dv['x1'])<=10
                           for dv in divs)
        has_bear_div = any(dv['type']=='bear' and
                           len(d)-np.searchsorted(d['timestamp'].values,dv['x1'])<=10
                           for dv in divs)
        bos_list, _, fvg_list = find_bos_ob_fvg(d)
        has_bos_bull = any(b['dir']==1  and
                           len(d)-np.searchsorted(d['timestamp'].values,b['x1'])<=5
                           for b in bos_list)
        has_bos_bear = any(b['dir']==-1 and
                           len(d)-np.searchsorted(d['timestamp'].values,b['x1'])<=5
                           for b in bos_list)

        # ── ПСЕВДО-FOOTPRINT триггеры (приоритет по силе)
        # 1. Золотое поглощение (самый сильный — 10/10)
        gold_abs_bull = d['absorption_gold_bull'].iloc[-5:].any()
        gold_abs_bear = d['absorption_gold_bear'].iloc[-5:].any()
        # 2. Delta exhaustion (сильный)
        exhaust_bull  = d['exhaustion_bear'].iloc[-5:].any()  # медв.дельта поглощается = бычий
        exhaust_bear  = d['exhaustion_bull'].iloc[-5:].any()  # бычья дельта поглощается = медвежий
        # 3. Stacked imbalance (след крупного)
        stack_bull    = d['stacked_bull'].iloc[-5:].any()
        stack_bear    = d['stacked_bear'].iloc[-5:].any()
        # 4. Imbalance extreme (экстремальный дисбаланс)
        imbal_bull    = d['imbalance_bull'].iloc[-3:].any()
        imbal_bear    = d['imbalance_bear'].iloc[-3:].any()
        imbal_ext_bull = d['imbalance_extreme'].iloc[-3:].any() & (d['delta'].iloc[-3:] > 0).any()
        imbal_ext_bear = d['imbalance_extreme'].iloc[-3:].any() & (d['delta'].iloc[-3:] < 0).any()
        # 5. CVD pressure zones
        cvd_press_bull = d['cvd_pressure_bull'].iloc[-3:].any()
        cvd_press_bear = d['cvd_pressure_bear'].iloc[-3:].any()

        def get_fp_triggers(direction):
            """Возвращает footprint триггеры с приоритетом по силе."""
            t = []
            if direction == 1:
                if gold_abs_bull:   t.append("🥇 Gold Absorption")   # приоритет 1
                if exhaust_bull:    t.append("💥 Delta Exhaustion ↑") # приоритет 2
                if stack_bull:      t.append("📦 Stacked Bull")       # приоритет 3
                if imbal_ext_bull:  t.append("⚡ Extreme Imbalance ↑")
                if imbal_bull:      t.append("↑ Imbalance Bull")
                if cvd_press_bull:  t.append("〰 CVD Pressure ↑")
                if abs_bull_r:      t.append("Absorption ⭐")
                if dshift_bull or dp > 0: t.append("Delta shift ↑")
                if has_bull_div:    t.append("CVD Div Bull")
                if sweep_low:       t.append("Sweep Low ⚡")
                if has_bos_bull:    t.append("BOS ↑")
            else:
                if gold_abs_bear:   t.append("🥇 Gold Absorption")
                if exhaust_bear:    t.append("💥 Delta Exhaustion ↓")
                if stack_bear:      t.append("📦 Stacked Bear")
                if imbal_ext_bear:  t.append("⚡ Extreme Imbalance ↓")
                if imbal_bear:      t.append("↓ Imbalance Bear")
                if cvd_press_bear:  t.append("〰 CVD Pressure ↓")
                if abs_bear_r:      t.append("Absorption ⭐")
                if dshift_bear or dp < 0: t.append("Delta shift ↓")
                if has_bear_div:    t.append("CVD Div Bear")
                if sweep_high:      t.append("Sweep High ⚡")
                if has_bos_bear:    t.append("BOS ↓")
            return t

        # ════════════════════════════════════════════════════
        # ТИП 1 — REVERSAL: FTR зона + 1 триггер
        # ════════════════════════════════════════════════════
        setup_type = None
        signal_dir = None
        signals    = []
        ftr_zone   = None
        ftr_tc     = 0

        for z in zns:
            if not z['active']: continue
            # В тренде — только по тренду; в балансе — обе стороны
            if htf_context == "TREND" and z['dir'] != trade_dir: continue
            if htf_context == "BALANCE" and trade_dir and z['dir'] != trade_dir: continue

            inside = z['zl'] <= last_close <= z['zh']
            near   = min(abs(last_close-z['zl']),
                         abs(last_close-z['zh'])) / last_close * 100 < 0.8
            if not (inside or near): continue

            tc = z.get('touch_count', 0)
            if tc >= 3: continue

            # Фильтр зрелости зоны — цена должна была покинуть зону после формирования
            bars_since_zone = len(d) - z['i']
            if bars_since_zone < 5:
                continue
            z_slice = d['close'].iloc[z['i']+1:-1]
            if len(z_slice) > 0:
                was_outside = (z_slice > z['zh']).any() if z['dir']==1 else (z_slice < z['zl']).any()
                if not was_outside:
                    continue

            # Нужен 1 триггер из: absorption, delta shift, div, sweep, BOS
            triggers = []
            d_for_check = z['dir']
            if d_for_check == 1:
                if abs_bull_r:                            triggers.append("Absorption ⭐")
                if dshift_bull or dp > 0:                 triggers.append("Delta shift ↑")
                if has_bull_div:                          triggers.append("CVD Div Bull")
                if sweep_low:                             triggers.append("Sweep Low ⚡")
                if has_bos_bull:                          triggers.append("BOS ↑")
            else:
                if abs_bear_r:                            triggers.append("Absorption ⭐")
                if dshift_bear or dp < 0:                 triggers.append("Delta shift ↓")
                if has_bear_div:                          triggers.append("CVD Div Bear")
                if sweep_high:                            triggers.append("Sweep High ⚡")
                if has_bos_bear:                          triggers.append("BOS ↓")

            if not triggers: continue  # нет ни одного триггера

            setup_type = "Reversal"
            signal_dir = d_for_check
            ftr_zone   = z
            ftr_tc     = tc
            touch_str  = f"1-й тест ⭐" if tc==0 else (f"2-й тест 🔥" if tc==1 else f"3-й тест")
            dist = min(abs(last_close-z['zl']), abs(last_close-z['zh'])) / last_close * 100
            signals = [
                f"HTF: {htf_context} {'↑' if t_dir_htf==1 else '↓'}",
                f"{'Demand' if z['dir']==1 else 'Supply'} FTR {touch_str} ({dist:.2f}%)",
            ] + triggers
            break

        # ════════════════════════════════════════════════════
        # ТИП 2 — CONTINUATION: тренд + откат к VAL/VAH/POC
        # ════════════════════════════════════════════════════
        if setup_type is None and htf_context == "TREND":
            cont_dir = t_dir_htf
            at_level = False
            level_name = ""

            val_dist  = abs(last_close - val) / last_close * 100
            vah_dist  = abs(last_close - vah) / last_close * 100
            poc_dist  = abs(last_close - cur_poc) / last_close * 100 if not np.isnan(cur_poc) else 999
            val_htf_d = abs(last_close - val_htf) / last_close * 100
            vah_htf_d = abs(last_close - vah_htf) / last_close * 100

            # Shallow pullback: откат 0.3–0.8 ATR от последнего хая/лоя
            recent_high = float(d['high'].iloc[-10:].max())
            recent_low  = float(d['low'].iloc[-10:].min())
            pullback_bull = cont_dir == 1  and (recent_high - last_close) > atr_val * 0.3                                            and (recent_high - last_close) < atr_val * 1.5
            pullback_bear = cont_dir == -1 and (last_close - recent_low)  > atr_val * 0.3                                            and (last_close - recent_low)  < atr_val * 1.5

            if cont_dir == 1:
                if val_dist < 0.5:       at_level = True; level_name = "LTF VAL"
                elif val_htf_d < 0.8:    at_level = True; level_name = "HTF VAL"
                elif poc_dist < 0.4:     at_level = True; level_name = "POC"
                elif pullback_bull:      at_level = True; level_name = "Shallow Pullback"
            else:
                if vah_dist < 0.5:       at_level = True; level_name = "LTF VAH"
                elif vah_htf_d < 0.8:    at_level = True; level_name = "HTF VAH"
                elif poc_dist < 0.4:     at_level = True; level_name = "POC"
                elif pullback_bear:      at_level = True; level_name = "Shallow Pullback"

            if at_level:
                triggers = []
                if cont_dir == 1:
                    if abs_bull_r:               triggers.append("Absorption ⭐")
                    if dshift_bull or dp > 0:    triggers.append("Delta shift ↑")
                    if has_bull_div:             triggers.append("CVD Div Bull")
                    if sweep_low:                triggers.append("Sweep Low ⚡")
                else:
                    if abs_bear_r:               triggers.append("Absorption ⭐")
                    if dshift_bear or dp < 0:    triggers.append("Delta shift ↓")
                    if has_bear_div:             triggers.append("CVD Div Bear")
                    if sweep_high:               triggers.append("Sweep High ⚡")

                if triggers:
                    setup_type = "Continuation"
                    signal_dir = cont_dir
                    signals = [
                        f"HTF: TREND {'↑' if t_dir_htf==1 else '↓'}",
                        f"Откат к {level_name}",
                    ] + triggers

        # ════════════════════════════════════════════════════
        # ТИП 3 — LIQUIDITY GRAB: sweep + возврат
        # ════════════════════════════════════════════════════
        if setup_type is None:
            grab_dir = None
            prev_low20  = float(d['low'].rolling(20).min().iloc[-2])
            prev_high20 = float(d['high'].rolling(20).max().iloc[-2])

            # Полный sweep ИЛИ касание в пределах 0.3% (equal lows/highs)
            near_low  = abs(last_close - prev_low20)  / last_close * 100 < 0.3
            near_high = abs(last_close - prev_high20) / last_close * 100 < 0.3

            if (sweep_low or near_low)   and (trade_dir == 1  or trade_dir is None):
                grab_dir = 1
            elif (sweep_high or near_high) and (trade_dir == -1 or trade_dir is None):
                grab_dir = -1

            if grab_dir is not None:
                # Цена вернулась или разворачивается от sweep
                returned = (grab_dir == 1  and last_close > prev_low20) or                            (grab_dir == -1 and last_close < prev_high20)
                if returned:
                    triggers = []
                    if grab_dir == 1:
                        if abs_bull_r:            triggers.append("Absorption ⭐")
                        if dshift_bull or dp > 0: triggers.append("Delta shift ↑")
                        if has_bull_div:          triggers.append("CVD Div Bull")
                    else:
                        if abs_bear_r:            triggers.append("Absorption ⭐")
                        if dshift_bear or dp < 0: triggers.append("Delta shift ↓")
                        if has_bear_div:          triggers.append("CVD Div Bear")

                    if triggers:
                        setup_type = "Liq.Grab"
                        signal_dir = grab_dir
                        signals = [
                            f"HTF: {htf_context} {'↑' if t_dir_htf==1 else '↓'}",
                            f"Sweep {'Low' if grab_dir==1 else 'High'} ⚡ + возврат",
                        ] + triggers

        # ══════════════════════════════════════════════════
        # ══════════════════════════════════════════════════
        # ТИП 4 — BREAKOUT RETEST
        # Без триггеров — факт возврата достаточен
        # ══════════════════════════════════════════════════
        # Структурный breakout: цена вышла за Key High или Key Low
        _struct_bko = mkt_mode  # "UP"/"DOWN"/None
        # Breakout Retest через структурные Key High/Key Low
        if setup_type is None and _struct_bko is not None:
            kh_use = bal_high if not np.isnan(bal_high) else vah
            kl_use = bal_low  if not np.isnan(bal_low)  else val

            if not np.isnan(kh_use) and not np.isnan(kl_use):
                kh_dist = abs(last_close - kh_use) / last_close * 100
                kl_dist = abs(last_close - kl_use) / last_close * 100

                # Retest Key High сверху → шорт
                if _struct_bko == "UP" and kh_dist < 0.6:
                    was_above = (d['close'].iloc[-5:-1] > kh_use).any()
                    if was_above:
                        setup_type = "Breakout Retest"
                        signal_dir = -1
                        signals = [
                            f"HTF:{final_context} {'↑' if t_dir_htf==1 else '↓'}",
                            f"🔄 Retest Key High сверху ({kh_use:.4f})",
                        ]

                # Retest Key Low снизу → лонг
                elif _struct_bko == "DOWN" and kl_dist < 0.6:
                    was_below = (d['close'].iloc[-5:-1] < kl_use).any()
                    if was_below:
                        setup_type = "Breakout Retest"
                        signal_dir = 1
                        signals = [
                            f"HTF:{final_context} {'↑' if t_dir_htf==1 else '↓'}",
                            f"🔄 Retest Key Low снизу ({kl_use:.4f})",
                        ]

                # Retest POC после выхода из баланса
                elif not np.isnan(cur_poc):
                    poc_d_br = abs(last_close - cur_poc) / last_close * 100
                    if poc_d_br < 0.4:
                        br_dir = 1 if (abs_bull_r or dshift_bull) else (-1 if (abs_bear_r or dshift_bear) else t_dir_htf)
                        setup_type = "Breakout Retest"
                        signal_dir = br_dir
                        signals = [
                            f"HTF:{final_context} {'↑' if t_dir_htf==1 else '↓'}",
                            "🔄 Retest POC",
                        ]

        # ══════════════════════════════════════════════════
        # PRE-SIGNAL: радар без подтверждений
        # ══════════════════════════════════════════════════
        pre_signals = []
        d_vah = abs(last_close-vah)/last_close*100 if not np.isnan(vah) else 999
        d_val = abs(last_close-val)/last_close*100 if not np.isnan(val) else 999
        d_poc = abs(last_close-cur_poc)/last_close*100 if not np.isnan(cur_poc) else 999
        # PRE: уровни без ограничения по режиму рынка
        if d_vah < 0.5:
            mode_tag = f"[{mkt_mode}]" if mkt_mode else ""
            pre_signals.append(f"PRE: подход к VAH ({d_vah:.2f}%) {mode_tag}")
        if d_val < 0.5:
            mode_tag = f"[{mkt_mode}]" if mkt_mode else ""
            pre_signals.append(f"PRE: подход к VAL ({d_val:.2f}%) {mode_tag}")
        if d_poc < 0.4:
            pre_signals.append(f"PRE: у POC ({d_poc:.2f}%)")
        for z in zns:
            if not z['active']: continue
            z_dist = min(abs(last_close-z['zl']),abs(last_close-z['zh']))/last_close*100
            if z_dist < 0.5:
                pre_signals.append(f"PRE: {'Demand' if z['dir']==1 else 'Supply'} FTR ({z_dist:.2f}%)")
                break
        if not np.isnan(vah) and not np.isnan(val):
            if last_close > vah and d_vah < 0.6:
                if (d['close'].iloc[-5:-1] > vah).any():
                    pre_signals.append(f"PRE: Retest VAH сверху ({d_vah:.2f}%) ⚡")
            elif last_close < val and d_val < 0.6:
                if (d['close'].iloc[-5:-1] < val).any():
                    pre_signals.append(f"PRE: Retest VAL снизу ({d_val:.2f}%) ⚡")
        p_low20  = float(d['low'].rolling(20).min().iloc[-2])
        p_high20 = float(d['high'].rolling(20).max().iloc[-2])
        if float(d['low'].iloc[-1]) < p_low20 and last_close > p_low20:
            pre_signals.append("PRE: Sweep Low + возврат ⚡")
        elif float(d['high'].iloc[-1]) > p_high20 and last_close < p_high20:
            pre_signals.append("PRE: Sweep High + возврат ⚡")

        if setup_type is None:
            if pre_signals:
                return {
                    'symbol':       symbol,
                    'price':        last_close,
                    'score':        len(pre_signals),
                    'signals':      pre_signals,
                    'trade_dir':    0,
                    'context':      f"PRE | {final_context}",
                    'strategy':     "Pre-Signal",
                    'ftr_test':     0, 'tvx': [],
                    'suggested_sl': np.nan,
                    'suggested_tp': np.nan,
                    'rr':           0,
                    'phase_trans':  False,
                    'amt_hint':     amt.get('strategy_hint',''),
                    'timestamp':    str(d['timestamp'].iloc[-1])[:16],
                }
            return None
        if signal_dir is None:
            return None

        # Нет сетапа — пропускаем
        stop_dist    = atr_val * 1.0
        suggested_sl = last_close - stop_dist if signal_dir==1 else last_close + stop_dist

        tp_candidates = []
        if signal_dir == 1:
            if not np.isnan(vah)     and vah     > last_close: tp_candidates.append(vah)
            if not np.isnan(poc_htf) and poc_htf > last_close: tp_candidates.append(poc_htf)
            if not np.isnan(vah_htf) and vah_htf > last_close: tp_candidates.append(vah_htf)
            if setup_type == "Breakout Retest" and not np.isnan(bal_high):
                tp_candidates.append(bal_high)
            for z in zns:
                if z['active'] and z['dir']==-1 and z['zl']>last_close:
                    tp_candidates.append(z['zl']); break
            SH, _ = find_swings(d)
            above = [s['price'] for s in SH if s['price'] > last_close]
            if above: tp_candidates.append(min(above))
        else:
            if not np.isnan(val)     and val     < last_close: tp_candidates.append(val)
            if not np.isnan(poc_htf) and poc_htf < last_close: tp_candidates.append(poc_htf)
            if not np.isnan(val_htf) and val_htf < last_close: tp_candidates.append(val_htf)
            for z in zns:
                if z['active'] and z['dir']==1 and z['zh']<last_close:
                    tp_candidates.append(z['zh']); break
            _, SL_sw = find_swings(d)
            below = [s['price'] for s in SL_sw if s['price'] < last_close]
            if below: tp_candidates.append(max(below))

        best_tp = None; best_rr = 0.0
        for tp in tp_candidates:
            tpd = abs(tp - last_close)
            if tpd < atr_val * 0.7: continue  # минимум 0.7 ATR до цели
            rr_tmp = tpd / stop_dist if stop_dist > 0 else 0
            if rr_tmp > best_rr: best_rr = rr_tmp; best_tp = tp

        # R/R фильтр убран — показываем все сетапы
        # Для Clean Retest рекомендуем R/R >= 1.8 в сообщении
        if best_tp is None:
            # Нет TP цели — используем ATR*2 как запасной
            best_tp  = last_close + stop_dist * 2 if signal_dir == 1 else last_close - stop_dist * 2
            best_rr  = 2.0

        # ИСПРАВЛЕНИЕ 4: взвешенный score
        trigger_weights = {
            "🥇 Gold Absorption":   3,
            "💥 Delta Exhaustion":  3,
            "📦 Stacked":           2,
            "⚡ Extreme Imbalance": 2,
            "Absorption ⭐":        2,
            "CVD Div":              2,
            "Sweep":                2,
            "Imbalance":            1,
            "CVD Pressure":         1,
            "Delta shift":          1,
            "BOS":                  1,
        }
        weighted_score = 3  # базовые очки за HTF контекст
        for sig in signals:
            for key, w in trigger_weights.items():
                if key in sig:
                    weighted_score += w
                    break
        if phase_transition and setup_type in ("Reversal","Liq.Grab","Clean Retest"):
            weighted_score += 2  # бонус за переход фаз

        return {
            'symbol':       symbol,
            'price':        last_close,
            'score':        weighted_score,
            'signals':      signals,
            'trade_dir':    signal_dir,
            'context':      f"HTF:{final_context}",
            'strategy':     setup_type,
            'ftr_test':     ftr_tc,
            'phase_trans':  phase_transition,
            'amt_hint':     amt.get('strategy_hint',''),
            'tvx':          [s for s in signals if any(t in s for t in
                              ["Absorption","Delta","CVD","Sweep","BOS","Retest"])],
            'suggested_sl': round(suggested_sl, 6),
            'suggested_tp': round(best_tp, 6),
            'rr':           round(best_rr, 1),
            'timestamp':    str(d['timestamp'].iloc[-1])[:16],
        }
    except Exception as e:
        return None


def scan_potential(symbol, tf_scan, src_scan, max_dist_pct=2.0):
    """
    Ищет активы где цена ещё НЕ в зоне, но приближается к ней.
    max_dist_pct — максимальное расстояние до зоны в %.
    Возвращает список потенциальных зон с направлением и расстоянием.
    """
    try:
        d = fetch_main_data(symbol, tf_scan, False, src_scan)
        if d.empty or len(d) < 60:
            return None
        d = apply_order_flow(d)

        ALTS_MAIN = ["XRPUSDT","XLMUSDT","SOLUSDT","TRXUSDT","BCHUSDT","LTCUSDT",
                     "AAVEUSDT","BNBUSDT","AVAXUSDT","ETCUSDT","LINKUSDT","DOGEUSDT",
                     "NEARUSDT","ATOMUSDT","UNIUSDT","POLUSDT","ADAUSDT","DOTUSDT",
                     "ETHUSDT","OPUSDT","FILUSDT","ARBUSDT"]
        ALTS_2 = ["CAKEUSDT","WOOUSDT","VETUSDT","ZILUSDT","STXUSDT","WAVESUSDT",
                  "ZRXUSDT","XTZUSDT","JASMYUSDT","ORDIUSDT","TONUSDT","INJUSDT",
                  "SEIUSDT","PEOPLEUSDT","FLOKIUSDT","YGGUSDT","GALAUSDT","RUNEUSDT",
                  "CRVUSDT","ICPUSDT","MASKUSDT","APEUSDT","SUIUSDT","APTUSDT",
                  "MANAUSDT","THETAUSDT","GMTUSDT","HBARUSDT","SANDUSDT","KSMUSDT"]

        zns = calc_ftr_zones(d, **get_ftr_params(symbol))

        last_close  = float(d['close'].iloc[-1])
        last_open   = float(d['open'].iloc[-1])

        # ── Скорость подхода: ATR-нормированное движение последних 3 баров
        atr14 = float((d['high'] - d['low']).rolling(14).mean().iloc[-1])
        move3 = float(d['close'].iloc[-1] - d['close'].iloc[-4])  # движение за 3 бара
        speed_atr = abs(move3) / atr14 if atr14 > 0 else 0  # >1 = быстрое движение

        # Направление движения (цена идёт к зоне или от неё)
        price_going_down = d['close'].iloc[-1] < d['close'].iloc[-3]
        price_going_up   = d['close'].iloc[-1] > d['close'].iloc[-3]

        # Контекст рынка
        (in_bal, t_dir, bal_high, bal_low, vah, val, cur_poc,
         mkt_mode, _str_sc, _kh_sc, _kl_sc) = get_struct_levels(d, tf=tf_scan, sw=3)
        _all_prof = build_balance_profiles(d, tf_scan, max_balances=10)
        hist_pocs = [{'x0': p['x0'], 'x1': p['x1'], 'poc': p['poc']}
                     for p in _all_prof if not p.get('is_current', False)]
        context = "BALANCE" if in_bal else "TREND"

        candidates = []

        # ── 1. FTR зоны в пределах max_dist_pct
        for z in zns:
            if not z['active']: continue
            touch = z.get('touch_count', 0)

            # Цена ВЫШЕ зоны → приближается к Demand снизу
            if last_close > z['zh']:
                dist_pct = (last_close - z['zh']) / last_close * 100
                if 0 < dist_pct <= max_dist_pct and z['dir'] == 1:
                    # Цена должна идти ВНИЗ к зоне
                    if not price_going_down: continue
                    speed_label = "🚀 Быстро" if speed_atr > 1.5 else ("➡️ Умеренно" if speed_atr > 0.5 else "🐢 Медленно")
                    scenario = "reversal" if touch == 0 else "retest"
                    candidates.append({
                        'zone_type':  'Demand FTR',
                        'trade_dir':  1,
                        'dist_pct':   round(dist_pct, 2),
                        'zone_lo':    z['zl'],
                        'zone_hi':    z['zh'],
                        'approach':   f"цена падает к зоне {speed_label}",
                        'speed_atr':  round(speed_atr, 2),
                        'touch':      touch,
                        'scenario':   scenario,
                    })

            # Цена НИЖЕ зоны → приближается к Supply сверху
            elif last_close < z['zl']:
                dist_pct = (z['zl'] - last_close) / last_close * 100
                if 0 < dist_pct <= max_dist_pct and z['dir'] == -1:
                    if not price_going_up: continue
                    speed_label = "🚀 Быстро" if speed_atr > 1.5 else ("➡️ Умеренно" if speed_atr > 0.5 else "🐢 Медленно")
                    scenario = "reversal" if touch == 0 else "retest"
                    candidates.append({
                        'zone_type':  'Supply FTR',
                        'trade_dir':  -1,
                        'dist_pct':   round(dist_pct, 2),
                        'zone_lo':    z['zl'],
                        'zone_hi':    z['zh'],
                        'approach':   f"цена растёт к зоне {speed_label}",
                        'speed_atr':  round(speed_atr, 2),
                        'touch':      touch,
                        'scenario':   scenario,
                    })

        # ── 2. VAH / VAL
        val_dist = (last_close - val) / last_close * 100
        vah_dist = (vah - last_close) / last_close * 100

        if 0 < val_dist <= max_dist_pct:
            candidates.append({
                'zone_type': 'VAL (Value Area Low)',
                'trade_dir': 1,
                'dist_pct':  round(val_dist, 2),
                'zone_lo':   val * 0.999,
                'zone_hi':   val * 1.001,
                'approach':  "цена падает к VAL → Long",
            })

        if 0 < vah_dist <= max_dist_pct:
            candidates.append({
                'zone_type': 'VAH (Value Area High)',
                'trade_dir': -1,
                'dist_pct':  round(vah_dist, 2),
                'zone_lo':   vah * 0.999,
                'zone_hi':   vah * 1.001,
                'approach':  "цена растёт к VAH → Short",
            })

        # ── 3. Текущий POC
        if not np.isnan(cur_poc):
            poc_dist = abs(last_close - cur_poc) / last_close * 100
            if 0.3 < poc_dist <= max_dist_pct:
                poc_dir = 1 if last_close < cur_poc else -1
                candidates.append({
                    'zone_type': 'POC',
                    'trade_dir': poc_dir,
                    'dist_pct':  round(poc_dist, 2),
                    'zone_lo':   cur_poc * 0.999,
                    'zone_hi':   cur_poc * 1.001,
                    'approach':  f"цена идёт {'вверх' if poc_dir==1 else 'вниз'} к POC",
                })

        # ── 4. Исторические POC
        for hp in hist_pocs[-3:]:
            hp_dist = abs(last_close - hp['poc']) / last_close * 100
            if 0.3 < hp_dist <= max_dist_pct:
                hp_dir = 1 if last_close < hp['poc'] else -1
                candidates.append({
                    'zone_type': 'Hist POC',
                    'trade_dir': hp_dir,
                    'dist_pct':  round(hp_dist, 2),
                    'zone_lo':   hp['poc'] * 0.999,
                    'zone_hi':   hp['poc'] * 1.001,
                    'approach':  "старый POC — зона возврата",
                })

        if not candidates:
            return None

        # ── Скорость движения (импульс к зоне)
        atr14  = float((d['high'] - d['low']).rolling(14).mean().iloc[-1])
        last3_range = float((d['high'].iloc[-3:].max() - d['low'].iloc[-3:].min()))
        momentum = last3_range / atr14 if atr14 > 0 else 1.0
        # momentum > 1.5 = цена летит, < 0.5 = еле движется

        # ── Определяем идёт ли цена К зоне или ОТ неё
        price_3bar_ago = float(d['close'].iloc[-4])
        price_direction = 1 if last_close > price_3bar_ago else -1  # куда идёт цена

        filtered = []
        for c in candidates:
            # Проверяем совпадение направления движения цены к зоне
            going_to_zone = False
            if c['trade_dir'] == 1 and price_direction == -1:
                # Demand зона снизу, цена падает к ней — правильно
                going_to_zone = True
            elif c['trade_dir'] == -1 and price_direction == 1:
                # Supply зона сверху, цена растёт к ней — правильно
                going_to_zone = True
            elif c['zone_type'] in ('POC', 'Hist POC'):
                going_to_zone = True  # POC — двунаправленный магнит

            c['going_to_zone'] = going_to_zone
            c['momentum']      = round(momentum, 2)
            filtered.append(c)

        # Берём ближайшую зону куда идёт цена
        going = [c for c in filtered if c['going_to_zone']]
        not_going = [c for c in filtered if not c['going_to_zone']]
        candidates_sorted = sorted(going, key=lambda x: x['dist_pct']) +                             sorted(not_going, key=lambda x: x['dist_pct'])

        if not candidates_sorted:
            return None

        best = candidates_sorted[0]
        trend_matches = (not in_bal and best['trade_dir'] == t_dir) or in_bal
        priority = "🔥 По тренду" if trend_matches else "⚠️ Против тренда"

        # Сценарий
        if best['trade_dir'] == 1:
            scenario = "reversal Long" if not in_bal else "bounce от VAL"
        else:
            scenario = "reversal Short" if not in_bal else "bounce от VAH"

        return {
            'symbol':     symbol,
            'price':      last_close,
            'context':    context,
            'priority':   priority,
            'scenario':   scenario,
            'momentum':   round(momentum, 2),
            'candidates': candidates_sorted[:3],
            'timestamp':  str(d['timestamp'].iloc[-1])[:16],
        }
    except:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 9. ОТРИСОВКА
# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════
#  ВКЛАДКА: СКРИНЕР СИГНАЛОВ
# ══════════════════════════════════════════════════════════════
if "Скринер" in tab:
    st.header("🔍 Скринер сигналов")
    ALL_SYMBOLS = (
        ["BTCUSDT"] +
        ["XRPUSDT","XLMUSDT","SOLUSDT","TRXUSDT","BCHUSDT","LTCUSDT",
         "AAVEUSDT","BNBUSDT","AVAXUSDT","ETCUSDT","LINKUSDT","DOGEUSDT",
         "NEARUSDT","ATOMUSDT","UNIUSDT","POLUSDT","ADAUSDT","DOTUSDT",
         "ETHUSDT","OPUSDT","FILUSDT","ARBUSDT"] +
        ["CAKEUSDT","WOOUSDT","VETUSDT","ZILUSDT","STXUSDT","WAVESUSDT",
         "ZRXUSDT","XTZUSDT","JASMYUSDT","ORDIUSDT","TONUSDT","INJUSDT",
         "SEIUSDT","PEOPLEUSDT","FLOKIUSDT","YGGUSDT","GALAUSDT","RUNEUSDT",
         "CRVUSDT","ICPUSDT","MASKUSDT","APEUSDT","SUIUSDT","APTUSDT",
         "MANAUSDT","THETAUSDT","GMTUSDT","HBARUSDT","SANDUSDT","KSMUSDT"] +
        ["ANKRUSDT","ASTRUSDT","RVNUSDT","GASUSDT","BOMEUSDT","ONTUSDT",
         "MEMEUSDT","ARUSDT","BANDUSDT","CKBUSDT","KAVAUSDT","BLURUSDT",
         "1000PEPEUSDT","GLMUSDT","1000BONKUSDT","PEPEUSDT","WIFUSDT",
         "POPCATUSDT","DYDXUSDT","SKLUSDT","AGIUSDT","IMXUSDT","LRCUSDT",
         "MUBARAKUSDT","SUSHIUSDT","XVSUSDT","MAGICUSDT","ZETAUSDT",
         "ENSUSDT","PENGUUSDT","COTIUSDT"]
    )
    scan_tf = st.radio("Таймфрейм скрининга:", ["15m","1h","1D"], horizontal=True, key="scan_tf")
    if st.button("▶ Запустить скрининг", key="run_scan"):
        results = []
        prog = st.progress(0)
        for idx_s, sym in enumerate(ALL_SYMBOLS):
            prog.progress((idx_s+1)/len(ALL_SYMBOLS))
            r = scan_signal(sym, scan_tf, "crypto")
            if r: results.append(r)
        prog.empty()
        results.sort(key=lambda x: x['score'], reverse=True)
        st.session_state['scan_results'] = results

    if 'scan_results' in st.session_state and st.session_state['scan_results']:
        res = st.session_state['scan_results']

        # ── Разделяем на PRE и ENTRY
        pre_res   = [r for r in res if r.get('strategy') == 'Pre-Signal']
        entry_res = [r for r in res if r.get('strategy') != 'Pre-Signal']

        total_str = f"ENTRY: {len(entry_res)} | PRE: {len(pre_res)}"
        st.success(f"Найдено: {total_str}")

        # ══════════════════════════════════════════
        # 🚨 ENTRY SIGNALS — подтверждённые входы
        # ══════════════════════════════════════════
        if entry_res:
            st.subheader(f"🚨 Сигналы входа ({len(entry_res)})")
            st.caption("Есть зона + триггер + контекст. Рассматривай как реальный вход.")
            for r in entry_res:
                dir_emoji  = "🟢 Long"  if r['trade_dir']== 1 else "🔴 Short"
                score_bar  = "🔥" * min(r['score']//2, 5)
                tvx_str    = " + ".join(r.get('tvx', []))
                strat      = r.get('strategy','')
                rr         = r.get('rr','?')
                with st.expander(
                    f"🚨 {r['symbol']}  {score_bar}  {dir_emoji}  "
                    f"[{strat}]  R/R 1:{rr}  —  {r['price']:.5f}"
                ):
                    st.markdown(f"**⚡ ТВХ:** `{tvx_str}`")
                    st.divider()
                    for sig in r['signals']:
                        st.markdown(f"✅ {sig}")
                    st.divider()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Entry", f"{r['price']:.5f}")
                    sl_val = r.get('suggested_sl')
                    tp_val = r.get('suggested_tp')
                    c2.metric("SL", f"{sl_val:.5f}" if sl_val and not (isinstance(sl_val, float) and np.isnan(sl_val)) else "—")
                    c3.metric("TP", f"{tp_val:.5f}" if tp_val and not (isinstance(tp_val, float) and np.isnan(tp_val)) else "—")
                    if r.get('amt_hint'):
                        st.caption(f"💡 {r['amt_hint']}")
                    st.caption(f"Время: {r['timestamp']}")
                    if st.button(f"📌 Открыть сделку по {r['symbol']}", key=f"open_{r['symbol']}"):
                        st.session_state['prefill_symbol'] = r['symbol']
                        st.session_state['prefill_dir']    = 'Long' if r['trade_dir']==1 else 'Short'
                        st.session_state['prefill_price']  = r['price']
                        st.session_state['prefill_sl']     = sl_val or 0.0
                        st.session_state['prefill_tp']     = tp_val or 0.0

        # ══════════════════════════════════════════
        # 🟡 PRE-SIGNALS — радар ситуаций
        # ══════════════════════════════════════════
        if pre_res:
            st.markdown("---")
            st.subheader(f"🟡 Радар ситуаций ({len(pre_res)})")
            st.caption("Цена рядом с зоной. Без подтверждений. Смотри сам — решай сам.")
            for r in pre_res:
                ctx = r.get('context','')
                with st.expander(
                    f"🟡 {r['symbol']}  |  {ctx}  |  {r['price']:.5f}"
                ):
                    for sig in r['signals']:
                        st.markdown(f"👁 {sig}")
                    if r.get('amt_hint'):
                        st.caption(f"💡 {r['amt_hint']}")
                    st.caption(f"Время: {r['timestamp']}")

    elif 'scan_results' in st.session_state:
        st.info("Сигналов не найдено — попробуй другой таймфрейм")
    st.stop()

# ══════════════════════════════════════════════════════════════
#  ВКЛАДКА: ЖУРНАЛ СДЕЛОК
# ══════════════════════════════════════════════════════════════
elif "Журнал" in tab:
    st.header("📒 Журнал сделок")
    trades = load_trades()

    # ── Форма новой сделки
    with st.expander("➕ Добавить сделку", expanded=('prefill_symbol' in st.session_state)):
        col1, col2 = st.columns(2)
        with col1:
            t_symbol = st.text_input("Пара", value=st.session_state.get('prefill_symbol','BTCUSDT'), key="t_sym")
            t_dir    = st.selectbox("Направление", ["Long","Short"],
                                    index=0 if st.session_state.get('prefill_dir','Long')=='Long' else 1, key="t_dir")
            t_tf     = st.selectbox("Таймфрейм", ["15m","1h","1D"], key="t_tf")
            t_entry  = st.number_input("Цена входа", value=float(st.session_state.get('prefill_price', 0.0)), format="%.5f", key="t_entry")
        with col2:
            t_sl     = st.number_input("Stop Loss", value=float(st.session_state.get('prefill_sl', 0.0)), format="%.5f", key="t_sl")
            t_tp     = st.number_input("Take Profit", value=float(st.session_state.get('prefill_tp', 0.0)), format="%.5f", key="t_tp")

            # Дата и время входа — для машины времени
            # Инициализируем дефолты только один раз
            from datetime import time as dt_time
            if 't_date_default' not in st.session_state:
                st.session_state['t_date_default'] = datetime.now().date()
            if 't_time_default' not in st.session_state:
                st.session_state['t_time_default'] = dt_time(datetime.now().hour, 0)

            st.date_input("Дата входа",
                          value=st.session_state['t_date_default'],
                          key="t_date_widget")
            st.time_input("Время входа (UTC)",
                          value=st.session_state['t_time_default'],
                          key="t_time_widget",
                          step=60)

            # Читаем текущее выбранное значение напрямую из session_state
            t_date = st.session_state.get('t_date_widget', st.session_state['t_date_default'])
            t_time = st.session_state.get('t_time_widget', st.session_state['t_time_default'])

            t_note = st.text_area("Заметка (почему вошёл — Vision AI прочитает это)", key="t_note",
                                  placeholder="Например: Вижу свип Key Low, бычья дивергенция CVD, цена в Demand FTR выше dPOC")

        if st.button("💾 Сохранить сделку", key="save_trade"):

            # ══════════════════════════════════════════════════════════════
            # ПОЛНЫЙ ПАЙПЛАЙН ЖУРНАЛА:
            # 1. Вычисляем timestamp момента входа ("Машина времени")
            # 2. Загружаем исторические данные СТРОГО ДО этого момента
            # 3. Строим полный график с FTR, dPOC, CVD, BOS
            # 4. Делаем скриншот → Vision AI Claude анализирует
            # 5. Собираем SMC фичи для Random Forest
            # ══════════════════════════════════════════════════════════════

            # Момент входа → Unix ms
            entry_dt = datetime.combine(t_date, t_time)
            entry_ts_ms = int(entry_dt.timestamp() * 1000)

            market_snapshot = {}
            try:
                # ── Машина времени: данные СТРОГО ДО момента входа
                snap_df = fetch_historical_data(t_symbol.upper(), t_tf,
                                                end_timestamp_ms=entry_ts_ms)
                if snap_df.empty:
                    # Fallback на текущие данные
                    snap_df = fetch_main_data(t_symbol.upper(), t_tf, False, "crypto")
                if not snap_df.empty:
                    snap_df = apply_order_flow(snap_df)
                    (in_bal_s, tdir_s, _khs2, _kls2, vah_s, val_s, poc_s,
                     _bko_s, _str_s, _kh_s, _kl_s) = get_struct_levels(snap_df, tf=t_tf, sw=3)
                    hpocs_s = []

                    # HTF контекст
                    htf_tf_s = "1h" if t_tf == "15m" else "1D"
                    snap_htf = fetch_main_data(t_symbol.upper(), htf_tf_s, False, "crypto")
                    htf_context_s = "unknown"
                    if not snap_htf.empty:
                        snap_htf = apply_order_flow(snap_htf)
                        (inbal_h, tdir_h, _khs3, _kls3, vah_h, val_h, poc_h,
                         _bko_h2, _str_h2, _kh_h2, _kl_h2) = get_struct_levels(snap_htf, tf=htf_tf_s, sw=3)
                        htf_context_s = "balance" if inbal_h else ("trend_up" if tdir_h==1 else "trend_down")

                    last = snap_df.iloc[-1]
                    atr_s = float((snap_df['high']-snap_df['low']).rolling(14).mean().iloc[-1])

                    # Дивергенции
                    divs_s  = find_divergences(snap_df)
                    has_div = len(divs_s) > 0 and (
                        len(snap_df) - np.searchsorted(
                            snap_df['timestamp'].values, divs_s[-1]['x1']
                        ) <= 10
                    )

                    # Absorption
                    has_abs_bull = bool(snap_df['abs_bull'].iloc[-5:].any())
                    has_abs_bear = bool(snap_df['abs_bear'].iloc[-5:].any())

                    # Delta slope
                    dt = snap_df['delta'].rolling(20).mean()
                    dslope = float(dt.iloc[-1]) - float(dt.iloc[-5])

                    # Liquidity sweep
                    sweep_low  = float(snap_df['low'].iloc[-1])  < float(snap_df['low'].rolling(20).min().iloc[-2])
                    sweep_high = float(snap_df['high'].iloc[-1]) > float(snap_df['high'].rolling(20).max().iloc[-2])

                    # FTR зоны
                    zns_s = calc_ftr_zones(snap_df, **get_ftr_params(t_symbol))
                    active_zones = [z for z in zns_s if z['active']]
                    in_ftr = any(
                        z['zl'] <= float(last['close']) <= z['zh']
                        for z in active_zones
                    )
                    ftr_touch = next(
                        (z.get('touch_count',0) for z in active_zones
                         if z['zl'] <= float(last['close']) <= z['zh']), -1
                    )

                    # ── Дополнительные расчёты для снимка
                    cl   = float(last['close'])
                    op   = float(last['open'])
                    hi   = float(last['high'])
                    lo   = float(last['low'])
                    vol  = float(last['volume'])
                    dlt  = float(last['delta'])

                    # Позиция цены внутри Value Area
                    va_range   = float(vah_s) - float(val_s)
                    price_pos  = (cl - float(val_s)) / va_range if va_range > 0 else 0.5

                    # Volume Profile позиция
                    poc_dist   = abs(cl - float(poc_s)) / cl * 100 if not np.isnan(poc_s) else 999
                    vah_dist   = abs(cl - float(vah_s)) / cl * 100
                    val_dist   = abs(cl - float(val_s)) / cl * 100

                    # CVD тренд и наклон
                    cvd_arr    = snap_df['cvd'].values
                    cvd_slope  = float(cvd_arr[-1]) - float(cvd_arr[-10]) if len(cvd_arr) >= 10 else 0
                    cvd_vs_price = (
                        1 if (cl > snap_df['close'].iloc[-10] and cvd_arr[-1] > cvd_arr[-10])
                        else -1 if (cl < snap_df['close'].iloc[-10] and cvd_arr[-1] < cvd_arr[-10])
                        else 0
                    )

                    # Объём относительно среднего
                    vm30  = float(snap_df['volume'].rolling(30).mean().iloc[-1])
                    vstd  = float(snap_df['volume'].rolling(30).std().iloc[-1])
                    vol_z = (vol - vm30) / vstd if vstd > 0 else 0  # z-score объёма

                    # Структура свечи
                    body_pct   = abs(cl - op) / (hi - lo) * 100 if (hi - lo) > 0 else 0
                    upper_wick = (hi - max(cl, op)) / (hi - lo) * 100 if (hi - lo) > 0 else 0
                    lower_wick = (min(cl, op) - lo) / (hi - lo) * 100 if (hi - lo) > 0 else 0
                    is_bull_candle = cl > op

                    # Импульс за N баров
                    impulse_3  = abs(cl - float(snap_df['close'].iloc[-3]))  / cl * 100
                    impulse_10 = abs(cl - float(snap_df['close'].iloc[-10])) / cl * 100

                    # BOS в последних барах
                    bos_s, _, fvg_s = find_bos_ob_fvg(snap_df)
                    has_bos_bull = any(b['dir']==1  and
                        len(snap_df)-np.searchsorted(snap_df['timestamp'].values,b['x1'])<=5
                        for b in bos_s)
                    has_bos_bear = any(b['dir']==-1 and
                        len(snap_df)-np.searchsorted(snap_df['timestamp'].values,b['x1'])<=5
                        for b in bos_s)
                    has_fvg_bull = any(f['dir']==1  for f in fvg_s[-3:])
                    has_fvg_bear = any(f['dir']==-1 for f in fvg_s[-3:])

                    # Delta trend direction (20-bar)
                    dt_arr     = snap_df['delta'].rolling(20).mean()
                    dslope5    = float(dt_arr.iloc[-1]) - float(dt_arr.iloc[-5])
                    dslope10   = float(dt_arr.iloc[-1]) - float(dt_arr.iloc[-10])

                    # Absorption count за 10 баров
                    abs_bull_count = int(snap_df['abs_bull'].iloc[-10:].sum())
                    abs_bear_count = int(snap_df['abs_bear'].iloc[-10:].sum())

                    # Количество активных FTR зон
                    n_demand_zones = sum(1 for z in active_zones if z['dir']==1)
                    n_supply_zones = sum(1 for z in active_zones if z['dir']==-1)

                    # Ближайшая FTR зона — расстояние и направление
                    nearest_zone_dist = 999
                    nearest_zone_dir  = 0
                    for z in active_zones:
                        d_z = min(abs(cl - z['zl']), abs(cl - z['zh'])) / cl * 100
                        if d_z < nearest_zone_dist:
                            nearest_zone_dist = d_z
                            nearest_zone_dir  = z['dir']

                    # Историческое POC — рядом ли
                    near_hist_poc = any(
                        abs(cl - hp['poc']) / cl * 100 < 0.5
                        for hp in hpocs_s
                    )

                    market_snapshot = {
                        # ── Контекст рынка
                        'htf_context':       htf_context_s,
                        'ltf_in_balance':    bool(in_bal_s),
                        'ltf_trend_dir':     int(tdir_s),
                        'price_pos_in_va':   round(price_pos, 3),

                        # ── Value Area уровни
                        'poc':               round(float(poc_s), 6) if not np.isnan(poc_s) else None,
                        'vah':               round(float(vah_s), 6),
                        'val':               round(float(val_s), 6),
                        'poc_dist_pct':      round(poc_dist, 3),
                        'vah_dist_pct':      round(vah_dist, 3),
                        'val_dist_pct':      round(val_dist, 3),
                        'near_hist_poc':     bool(near_hist_poc),

                        # ── CVD / Delta
                        'has_div_cvd':       bool(has_div),
                        'cvd_slope_10':      round(cvd_slope, 2),
                        'cvd_vs_price':      cvd_vs_price,
                        'delta_last':        round(dlt, 2),
                        'delta_slope_5':     round(dslope5, 2),
                        'delta_slope_10':    round(dslope10, 2),
                        'delta_pressure':    round(float(snap_df['delta_pressure'].iloc[-1]), 2),

                        # ── Absorption
                        'has_abs_bull':      bool(has_abs_bull),
                        'has_abs_bear':      bool(has_abs_bear),
                        'abs_bull_count_10': abs_bull_count,
                        'abs_bear_count_10': abs_bear_count,

                        # ── Объём
                        'vol_z_score':       round(vol_z, 2),
                        'vol_vs_mean':       round(vol / vm30 if vm30 > 0 else 1, 2),

                        # ── Структура свечи
                        'body_pct':          round(body_pct, 1),
                        'upper_wick_pct':    round(upper_wick, 1),
                        'lower_wick_pct':    round(lower_wick, 1),
                        'is_bull_candle':    bool(is_bull_candle),

                        # ── Импульс и волатильность
                        'impulse_3_pct':     round(impulse_3, 3),
                        'impulse_10_pct':    round(impulse_10, 3),
                        'atr':               round(atr_s, 6),
                        'atr_pct':           round(atr_s / cl * 100, 4),

                        # ── FTR зоны
                        'in_ftr_zone':       in_ftr,
                        'ftr_touch_count':   ftr_touch,
                        'n_demand_zones':    n_demand_zones,
                        'n_supply_zones':    n_supply_zones,
                        'nearest_zone_dist': round(nearest_zone_dist, 3),
                        'nearest_zone_dir':  nearest_zone_dir,

                        # ── Структура (BOS / FVG)
                        'has_bos_bull':      bool(has_bos_bull),
                        'has_bos_bear':      bool(has_bos_bear),
                        'has_fvg_bull':      bool(has_fvg_bull),
                        'has_fvg_bear':      bool(has_fvg_bear),

                        # ── Liquidity
                        'sweep_low':         bool(sweep_low),
                        'sweep_high':        bool(sweep_high),

                        # ── R/R
                        'rr_planned':        round(abs(t_tp - t_entry) / abs(t_entry - t_sl), 2)
                                             if t_sl and t_tp and t_entry and abs(t_entry-t_sl) > 0 else None,
                        'sl_atr_ratio':      round(abs(t_entry - t_sl) / atr_s, 2)
                                             if t_sl and atr_s > 0 else None,
                    }

                    # ── Новые SMC фичи (Группа 1+2 детекторы)
                    snap_htf_obj = None
                    try:
                        htf_tf_smc = "1h" if t_tf == "15m" else "1D"
                        snap_htf_obj = fetch_main_data(t_symbol.upper(), htf_tf_smc, False, "crypto")
                        if not snap_htf_obj.empty:
                            snap_htf_obj = apply_order_flow(snap_htf_obj)
                    except:
                        pass

                    smc_features = collect_smc_snapshot(
                        snap_df, snap_htf_obj,
                        t_symbol, t_tf, t_entry, t_sl, t_tp
                    )
                    market_snapshot.update(smc_features)

            except Exception as e:
                market_snapshot = {'error': str(e)}

            trade_id = len(trades) + 1

            # ── Строим исторический график и запускаем Vision AI
            vision_features = None
            screenshot_path = None
            try:
                with st.spinner("📊 Строю исторический график на момент входа..."):
                    # Используем build_journal_chart — полный график с FTR/dPOC/CVD/BOS
                    # строго на момент entry_ts_ms ("Машина времени")
                    journal_fig = build_journal_chart(
                        t_symbol.upper(), t_tf,
                        end_timestamp_ms=entry_ts_ms
                    )

                if journal_fig is not None:
                    screenshot_path = take_trade_snapshot(journal_fig, trade_id)
                    st.image(screenshot_path, caption=f"Скриншот на момент входа ({entry_dt.strftime('%Y-%m-%d %H:%M')})",
                             use_container_width=True) if screenshot_path and screenshot_path.endswith('.png') else None

                if screenshot_path and screenshot_path.endswith('.png'):
                    trade_data_for_ai = {
                        'symbol':    t_symbol,
                        'tf':        t_tf,
                        'direction': t_dir,
                        'entry':     t_entry,
                        'sl':        t_sl,
                        'tp':        t_tp,
                        'entry_time': entry_dt.strftime('%Y-%m-%d %H:%M'),
                    }
                    with st.spinner("🤖 Claude Vision анализирует график..."):
                        vision_features = analyze_trade_with_claude(
                            screenshot_path, trade_data_for_ai, t_note
                        )

                    if vision_features:
                        market_snapshot['vision_ai'] = vision_features
                        rating = vision_features.get('trade_logic_rating', '?')
                        comment = vision_features.get('ai_comment', '')
                        trend = vision_features.get('trend_direction', '?')
                        cvd_div = vision_features.get('cvd_divergence_type', '?')

                        col_ai1, col_ai2, col_ai3 = st.columns(3)
                        col_ai1.metric("🤖 Оценка логики", f"{rating}/10")
                        col_ai2.metric("📈 Тренд", trend)
                        col_ai3.metric("📊 CVD", cvd_div)
                        st.info(f"💬 Claude: *{comment}*")
                    else:
                        st.warning("⚠️ Vision AI не вернул данные (kaleido не установлен или ошибка API)")
                else:
                    st.caption("ℹ️ Скриншот не сделан — установите kaleido: `pip install kaleido`")
            except Exception as e:
                print(f"[VISION] Ошибка: {e}")
                st.warning(f"⚠️ Vision AI: {e}")

            new_trade = {
                'id':           trade_id,
                'symbol':       t_symbol.upper(),
                'direction':    t_dir,
                'tf':           t_tf,
                'entry':        t_entry,
                'sl':           t_sl,
                'tp':           t_tp,
                'note':         t_note,
                'opened_at':    entry_dt.strftime("%Y-%m-%d %H:%M"),  # реальное время входа
                'closed_at':    None,
                'exit_price':   None,
                'result':       None,
                'pnl_pct':      None,
                'pnl_r':        None,
                'screenshot':   screenshot_path,
                'snapshot':     market_snapshot,
            }
            trades.append(new_trade)
            save_trades(trades)
            for k in ['prefill_symbol','prefill_dir','prefill_price','prefill_sl','prefill_tp',
                      't_date_default', 't_time_default', 't_date_widget', 't_time_widget']:
                st.session_state.pop(k, None)
            st.success(f"Сделка сохранена! Снимок: {len(market_snapshot)} параметров")
            st.rerun()

    # ── Открытые сделки
    open_trades = [t for t in trades if t['result'] is None]
    closed_trades = [t for t in trades if t['result'] is not None]

    if open_trades:
        st.subheader(f"🟡 Открытые сделки ({len(open_trades)})")
        for t in open_trades:
            dir_col = "🟢" if t['direction']=='Long' else "🔴"
            with st.expander(f"{dir_col} #{t['id']} {t['symbol']} {t['direction']} @ {t['entry']} [{t['tf']}] — {t['opened_at']}"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Entry", f"{t['entry']:.5f}")
                c2.metric("SL", f"{t['sl']:.5f}" if t['sl'] else "—")
                c3.metric("TP", f"{t['tp']:.5f}" if t['tp'] else "—")
                if t['note']:
                    st.caption(f"📝 {t['note']}")
                st.markdown("**Закрыть сделку:**")
                cc1, cc2, cc3 = st.columns(3)
                exit_p  = cc1.number_input("Цена выхода", value=float(t['entry']), format="%.5f", key=f"exit_{t['id']}")
                result  = cc2.selectbox("Результат", ["win","loss"], key=f"res_{t['id']}")
                if cc3.button("✅ Закрыть", key=f"close_{t['id']}"):
                    # R:R по методологии TradingView (риск = 1% депозита)
                    sl   = t.get('sl') or t.get('suggested_sl')
                    tp   = t.get('tp') or t.get('suggested_tp')
                    risk_dist = abs(t['entry'] - sl) if sl and sl > 0 else None
                    if risk_dist and risk_dist > 0:
                        if t['direction'] == 'Long':
                            pnl_r = (exit_p - t['entry']) / risk_dist   # в R
                        else:
                            pnl_r = (t['entry'] - exit_p) / risk_dist
                    else:
                        # Fallback: считаем через % и конвертируем (1% риска = 1R)
                        pnl_pct = ((exit_p - t['entry']) / t['entry'] * 100) if t['direction']=='Long'                                   else ((t['entry'] - exit_p) / t['entry'] * 100)
                        pnl_r = pnl_pct / 1.0  # 1% риска → 1R
                    for tr in trades:
                        if tr['id'] == t['id']:
                            tr['closed_at']  = datetime.now().strftime("%Y-%m-%d %H:%M")
                            tr['exit_price'] = exit_p
                            tr['result']     = result
                            tr['pnl_r']      = round(pnl_r, 2)   # в R (1R = 1% депозита)
                            tr['pnl_pct']    = round(pnl_r, 2)   # совместимость
                    save_trades(trades)
                    r_str = f"{pnl_r:+.2f}R"
                    pct_str = f"≈ {pnl_r:+.1f}% депозита"
                    st.success(f"Сделка закрыта. {r_str} ({pct_str})")
                    st.rerun()

    # ── Статистика
    if closed_trades:
        st.subheader(f"📊 Статистика ({len(closed_trades)} закрытых)")
        wins  = [t for t in closed_trades if t['result']=='win']
        losses= [t for t in closed_trades if t['result']=='loss']
        wr    = len(wins)/len(closed_trades)*100
        avg_r   = np.mean([t.get('pnl_r', t.get('pnl_pct', 0)) for t in closed_trades
                           if t.get('pnl_r', t.get('pnl_pct')) is not None])
        total_r = sum([t.get('pnl_r', t.get('pnl_pct', 0)) for t in closed_trades
                       if t.get('pnl_r', t.get('pnl_pct')) is not None])
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Всего сделок", len(closed_trades))
        m2.metric("Winrate",      f"{wr:.1f}%")
        m3.metric("Wins/Losses",  f"{len(wins)}/{len(losses)}")
        m4.metric("Avg R",        f"{avg_r:+.2f}R")
        m5.metric("Total R",      f"{total_r:+.1f}R  ≈{total_r:+.1f}%")

        # ── ИИ: обучение и предсказание
        st.subheader("🤖 ИИ-анализ паттернов")

        # Три уровня качества данных:
        # 1. vision_trades — Vision AI смотрел на график → лучшие фичи
        # 2. rich_trades  — есть снимок рынка → хорошие фичи
        # 3. basic_trades — только базовые параметры
        vision_trades = [t for t in closed_trades
                         if (t.get('snapshot') or {}).get('vision_ai')]
        rich_trades   = [t for t in closed_trades
                         if t.get('snapshot') and not t['snapshot'].get('error')]
        basic_trades  = closed_trades

        has_vision = len(vision_trades) >= 5
        has_rich   = len(rich_trades)   >= 5
        has_basic  = len(basic_trades)  >= 5

        if has_vision:
            st.success(f"✅ Vision AI обучение: {len(vision_trades)} сделок с анализом графика")
        elif has_rich:
            st.success(f"✅ Полное обучение: {len(rich_trades)} сделок со снимком рынка")
        elif has_basic:
            st.warning(f"⚠️ Базовое обучение: {len(basic_trades)} сделок")
        else:
            st.info(f"Нужно минимум 5 закрытых сделок (сейчас: {len(basic_trades)})")

        if has_basic:
            try:
                from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                from sklearn.model_selection import cross_val_score
                import warnings; warnings.filterwarnings('ignore')

                def trade_to_features(t):
                    """
                    Feature Engineering — 3 уровня.
                    Приоритет: Vision AI фичи > SMC снимок > базовые параметры.
                    """
                    snap  = t.get('snapshot', {}) or {}
                    ai    = snap.get('vision_ai', {}) or {}
                    entry = float(t['entry']) if t['entry'] else 1
                    sl    = float(t['sl'])    if t['sl']    else 0
                    tp    = float(t['tp'])    if t['tp']    else 0
                    is_long = 1 if t['direction'] == 'Long' else 0

                    def g(key, default=0):
                        v = snap.get(key, default)
                        return float(v) if v is not None else float(default)

                    def ga(key, default=0):
                        """Получить фичу из Vision AI блока."""
                        v = ai.get(key, default)
                        return float(v) if v is not None else float(default)

                    return [
                        # ── Vision AI фичи (ПРИОРИТЕТ — Claude видел реальный график)
                        ga('trade_logic_rating', 5),                                     # Оценка логики 1-10
                        1 if ai.get('trend_direction') == 'UP'      else 0,              # AI: тренд вверх
                        1 if ai.get('trend_direction') == 'DOWN'    else 0,              # AI: тренд вниз
                        1 if ai.get('trend_direction') == 'BALANCE' else 0,              # AI: баланс
                        1 if ai.get('is_at_key_level')              else 0,              # AI: у ключевого уровня
                        1 if ai.get('is_inside_ftr')                else 0,              # AI: в FTR зоне
                        1 if ai.get('liquidity_sweep')              else 0,              # AI: свип ликвидности
                        1 if 'ABSORPTION' in str(ai.get('cvd_divergence_type',''))  else 0,  # AI: поглощение
                        1 if 'EXHAUSTION' in str(ai.get('cvd_divergence_type',''))  else 0,  # AI: истощение
                        1 if 'BULL' in str(ai.get('cvd_divergence_type',''))        else 0,  # AI: бычий CVD
                        {'ABOVE':1,'AT':0,'BELOW':-1}.get(str(ai.get('dpoc_position','AT')), 0),  # AI: dPOC позиция

                        is_long,
                        {'15m':0,'1h':1,'1D':2}.get(t['tf'], 0),
                        (tp - entry) / entry * 100 if entry else 0,
                        (entry - sl) / entry * 100 if entry else 0,
                        g('rr_planned'),
                        g('sl_atr_ratio'),

                        # ── HTF/LTF контекст
                        1 if snap.get('htf_context') == 'trend_up'   else 0,
                        1 if snap.get('htf_context') == 'trend_down' else 0,
                        1 if snap.get('htf_context') == 'balance'    else 0,
                        1 if snap.get('ltf_in_balance') else 0,
                        g('price_pos_in_va'),

                        # ── Value Area / POC
                        g('poc_dist_pct'),
                        g('vah_dist_pct'),
                        g('val_dist_pct'),
                        1 if snap.get('near_hist_poc') else 0,

                        # ── CVD / Delta
                        1 if snap.get('has_div_cvd')  else 0,
                        g('cvd_slope_10'),
                        g('cvd_vs_price'),
                        g('delta_last'),
                        g('delta_slope_5'),
                        g('delta_slope_10'),
                        g('delta_pressure'),

                        # ── Absorption
                        1 if snap.get('has_abs_bull') else 0,
                        1 if snap.get('has_abs_bear') else 0,
                        g('abs_bull_count_10'),
                        g('abs_bear_count_10'),

                        # ── Объём
                        g('vol_z_score'),
                        g('vol_vs_mean'),

                        # ── Структура свечи
                        g('body_pct'),
                        g('upper_wick_pct'),
                        g('lower_wick_pct'),
                        1 if snap.get('is_bull_candle') else 0,

                        # ── Импульс и волатильность
                        g('impulse_3_pct'),
                        g('impulse_10_pct'),
                        g('atr_pct'),

                        # ── FTR зоны
                        1 if snap.get('in_ftr_zone') else 0,
                        g('ftr_touch_count'),
                        g('n_demand_zones'),
                        g('n_supply_zones'),
                        g('nearest_zone_dist'),
                        g('nearest_zone_dir'),

                        # ── Структура
                        1 if snap.get('has_bos_bull') else 0,
                        1 if snap.get('has_bos_bear') else 0,
                        1 if snap.get('has_fvg_bull') else 0,
                        1 if snap.get('has_fvg_bear') else 0,

                        # ── Liquidity
                        1 if snap.get('sweep_low')  else 0,
                        1 if snap.get('sweep_high') else 0,

                        # ── НОВЫЕ SMC фичи (Группы 1+2)
                        1 if snap.get('smc_phase') == 'TREND_UP'   else 0,
                        1 if snap.get('smc_phase') == 'TREND_DOWN' else 0,
                        1 if snap.get('smc_phase') == 'BALANCE'    else 0,
                        1 if snap.get('sweep_key_high') else 0,
                        1 if snap.get('sweep_key_low')  else 0,
                        1 if snap.get('in_demand_ftr')  else 0,
                        1 if snap.get('in_supply_ftr')  else 0,
                        1 if snap.get('cvd_div_type') in ('BULL_ABSORPTION','BEAR_ABSORPTION') else 0,
                        1 if snap.get('cvd_div_type') in ('BULL_EXHAUSTION','BEAR_EXHAUSTION') else 0,
                        1 if snap.get('cvd_div_type','').startswith('BULL') else 0,
                        {'ABOVE':1,'AT':0,'BELOW':-1}.get(snap.get('dpoc_position','AT'), 0),
                        1 if snap.get('is_bos_retest_long')  else 0,
                        1 if snap.get('is_bos_retest_short') else 0,
                        1 if snap.get('htf_phase_smc') == 'TREND_UP'   else 0,
                        1 if snap.get('htf_phase_smc') == 'TREND_DOWN' else 0,
                        1 if snap.get('htf_phase_smc') == 'BALANCE'    else 0,

                        # ── Vision AI фичи (если есть)
                        float((snap.get('vision_ai') or {}).get('trade_logic_rating', 5)),
                        1 if (snap.get('vision_ai') or {}).get('is_at_key_level') else 0,
                        1 if (snap.get('vision_ai') or {}).get('is_inside_ftr') else 0,
                        1 if (snap.get('vision_ai') or {}).get('liquidity_sweep') else 0,
                        1 if (snap.get('vision_ai') or {}).get('cvd_divergence_present') else 0,
                    ]

                feature_names = [
                    # Vision AI фичи (первые 11 — приоритетные)
                    'AI Rating',
                    'AI Trend↑','AI Trend↓','AI Balance',
                    'AI Key Level','AI In FTR','AI Liq Sweep',
                    'AI Absorption','AI Exhaustion','AI Bull CVD','AI dPOC',
                    # Базовые (6)
                    'Direction','Timeframe','TP%','SL%','RR Planned','SL/ATR',
                    # HTF/LTF (5)
                    'HTF↑','HTF↓','HTF Balance','LTF Balance','Price in VA',
                    # Value Area (4)
                    'POC Dist%','VAH Dist%','VAL Dist%','Near Hist POC',
                    # CVD/Delta (7)
                    'CVD Div','CVD Slope10','CVD vs Price',
                    'Delta Last','Delta Slope5','Delta Slope10','Delta Pressure',
                    # Absorption (4)
                    'Abs Bull','Abs Bear','Abs Bull Count','Abs Bear Count',
                    # Объём (2)
                    'Vol Z-Score','Vol/Mean',
                    # Свеча (4)
                    'Body%','Upper Wick%','Lower Wick%','Bull Candle',
                    # Импульс (3)
                    'Impulse 3bar%','Impulse 10bar%','ATR%',
                    # FTR зоны (6)
                    'In FTR','FTR Touch#','Demand Zones','Supply Zones',
                    'Nearest Zone Dist','Nearest Zone Dir',
                    # BOS/FVG (4)
                    'BOS Bull','BOS Bear','FVG Bull','FVG Bear',
                    # Liquidity (2)
                    'Sweep Low','Sweep High',
                    # SMC фичи (16)
                    'SMC Trend↑','SMC Trend↓','SMC Balance',
                    'Sweep Key High','Sweep Key Low',
                    'In Demand FTR','In Supply FTR',
                    'CVD Absorption','CVD Exhaustion','CVD Bull Dir',
                    'dPOC Position',
                    'BOS Retest Long','BOS Retest Short',
                    'HTF SMC↑','HTF SMC↓','HTF SMC Balance',
                    # Vision AI (финальный блок, 5)
                    'AI Logic Rating','AI At Key Level','AI In FTR2',
                    'AI Liq Sweep2','AI CVD Div2',
                ]
                # Итого: 11+6+5+4+7+4+2+4+3+6+4+2+16+5 = 79

                # Приоритет: Vision AI → Rich → Basic
                trades_for_train = (vision_trades if has_vision else
                                    rich_trades   if has_rich  else
                                    basic_trades)
                X = [trade_to_features(t) for t in trades_for_train]
                y = [1 if t['result']=='win' else 0 for t in trades_for_train]

                clf = RandomForestClassifier(n_estimators=200, random_state=42,
                                             min_samples_leaf=2)
                clf.fit(X, y)

                # Cross-validation если достаточно данных
                if len(trades_for_train) >= 10:
                    scores = cross_val_score(clf, X, y, cv=min(5, len(trades_for_train)//2))
                    st.metric("CV точность модели", f"{scores.mean():.1%} ± {scores.std():.1%}")

                # ── Важность признаков
                st.markdown("**📊 Что влияет на успех твоих сделок:**")
                importances = clf.feature_importances_
                feat_imp = sorted(zip(feature_names, importances), key=lambda x:-x[1])
                for fn, imp in feat_imp[:10]:  # топ 10
                    if imp < 0.01: continue
                    bar   = "█" * int(imp * 40)
                    color = "🟢" if imp > 0.1 else ("🟡" if imp > 0.05 else "⚪")
                    st.markdown(f"{color} `{fn:<20}` {bar} {imp:.1%}")

                # ── Анализ паттернов побед vs поражений
                st.markdown("---")
                st.markdown("**🔍 Паттерны твоих сделок:**")
                wins  = [t for t in trades_for_train if t['result']=='win']
                losses= [t for t in trades_for_train if t['result']=='loss']
                if wins and losses:
                    w1, w2, w3, w4 = st.columns(4)
                    # Winrate по HTF контексту
                    for ctx in ['trend_up','trend_down','balance']:
                        ctx_trades = [t for t in trades_for_train
                                      if (t.get('snapshot') or {}).get('htf_context')==ctx]
                        if ctx_trades:
                            wr_ctx = sum(1 for t in ctx_trades if t['result']=='win') / len(ctx_trades)
                            lbl = {'trend_up':'↑ Тренд','trend_down':'↓ Тренд','balance':'⚖️ Баланс'}[ctx]
                            st.metric(f"WR {lbl}", f"{wr_ctx:.0%}", f"{len(ctx_trades)} сделок")

                # ── Предсказание для новой сделки
                st.markdown("---")
                st.markdown("**🔮 Оценить вероятность новой сделки:**")
                st.caption("Заполни параметры — ИИ оценит на основе твоей статистики")

                pa, pb, pc = st.columns(3)
                pred_dir    = pa.selectbox("Направление", ["Long","Short"], key="pred_dir")
                pred_tf2    = pb.selectbox("Таймфрейм",   ["15m","1h","1D"], key="pred_tf2")
                pred_ctx    = pc.selectbox("HTF контекст",
                                           ["trend_up","trend_down","balance"], key="pred_ctx")
                pd1, pd2, pd3 = st.columns(3)
                pred_entry2 = pd1.number_input("Entry", value=0.0, format="%.5f", key="pred_e2")
                pred_sl2    = pd2.number_input("SL",    value=0.0, format="%.5f", key="pred_sl2")
                pred_tp2    = pd3.number_input("TP",    value=0.0, format="%.5f", key="pred_tp2")

                # SMC фичи
                st.caption("SMC контекст (из текущего графика):")
                pf1, pf2, pf3, pf4 = st.columns(4)
                pred_div    = pf1.checkbox("CVD Div",      key="pred_div")
                pred_abs    = pf2.checkbox("Absorption",   key="pred_abs")
                pred_sweep  = pf3.checkbox("Liq Sweep",    key="pred_sweep")
                pred_ftr    = pf4.checkbox("In FTR Zone",  key="pred_ftr")
                pg1, pg2, pg3, pg4 = st.columns(4)
                pred_smc_up   = pg1.checkbox("SMC Trend↑",  key="pred_smc_up")
                pred_smc_dn   = pg2.checkbox("SMC Trend↓",  key="pred_smc_dn")
                pred_bos_ret  = pg3.checkbox("BOS Retest",  key="pred_bos_ret")
                pred_dpoc_ab  = pg4.selectbox("dPOC",       ["AT","ABOVE","BELOW"], key="pred_dpoc")

                # Vision AI оценка (если есть)
                st.caption("Vision AI (если анализировал):")
                ph1, ph2 = st.columns(2)
                pred_ai_rating = ph1.slider("AI Логика (1-10)", 1, 10, 5, key="pred_ai_r")
                pred_ai_ftr    = ph2.checkbox("AI: в FTR зоне", key="pred_ai_ftr")

                if st.button("🔮 Оценить", key="predict2"):
                    entry2 = pred_entry2 if pred_entry2 > 0 else 1
                    sl2    = pred_sl2
                    tp2    = pred_tp2

                    # Строим вектор признаков СТРОГО как trade_to_features()
                    feat_new = [[
                        # Vision AI фичи
                        float(pred_ai_rating),
                        1 if pred_ctx == 'trend_up'   else 0,   # AI trend up (используем HTF как proxy)
                        1 if pred_ctx == 'trend_down' else 0,
                        1 if pred_ctx == 'balance'    else 0,
                        0,                                       # AI at key level
                        1 if pred_ftr else 0,                    # AI in FTR
                        1 if pred_sweep else 0,                  # AI liq sweep
                        1 if pred_abs else 0,                    # AI absorption
                        0,                                       # AI exhaustion
                        1 if (pred_abs and pred_dir=='Long') else 0,  # AI bull CVD
                        {'AT':0,'ABOVE':1,'BELOW':-1}.get(pred_dpoc_ab, 0),  # AI dPOC

                        # Базовые
                        1 if pred_dir == 'Long' else 0,
                        {'15m':0,'1h':1,'1D':2}.get(pred_tf2, 0),
                        (tp2 - entry2) / entry2 * 100 if entry2 else 0,
                        (entry2 - sl2) / entry2 * 100 if entry2 else 0,
                        (tp2 - entry2) / abs(entry2 - sl2) if abs(entry2 - sl2) > 0 else 0,
                        0.0,  # sl_atr_ratio

                        # HTF/LTF контекст
                        1 if pred_ctx == 'trend_up'   else 0,
                        1 if pred_ctx == 'trend_down' else 0,
                        1 if pred_ctx == 'balance'    else 0,
                        0,   # ltf_in_balance
                        0.5, # price_pos_in_va

                        # Value Area
                        0.0, 0.0, 0.0, 0,  # poc/vah/val dist, near_hist_poc

                        # CVD / Delta
                        1 if pred_div else 0,
                        0.0, 0, 0.0, 0.0, 0.0, 0.0,

                        # Absorption
                        1 if (pred_abs and pred_dir=='Long')  else 0,
                        1 if (pred_abs and pred_dir=='Short') else 0,
                        0, 0,

                        # Объём
                        0.0, 1.0,

                        # Структура свечи
                        0.0, 0.0, 0.0, 0,

                        # Импульс
                        0.0, 0.0, 0.0,

                        # FTR зоны
                        1 if pred_ftr else 0,
                        0, 0, 0, 0.0, 0,

                        # BOS/FVG
                        0, 0, 0, 0,

                        # Liquidity
                        1 if (pred_sweep and pred_dir=='Long')  else 0,
                        1 if (pred_sweep and pred_dir=='Short') else 0,

                        # SMC фичи
                        1 if pred_smc_up else 0,
                        1 if pred_smc_dn else 0,
                        1 if (not pred_smc_up and not pred_smc_dn) else 0,
                        1 if (pred_sweep and pred_dir=='Short') else 0,
                        1 if (pred_sweep and pred_dir=='Long')  else 0,
                        1 if (pred_ftr and pred_dir=='Long')  else 0,
                        1 if (pred_ftr and pred_dir=='Short') else 0,
                        1 if pred_abs else 0,
                        0,
                        1 if (pred_abs and pred_dir=='Long') else 0,
                        {'AT':0,'ABOVE':1,'BELOW':-1}.get(pred_dpoc_ab, 0),
                        1 if pred_bos_ret else 0,
                        0,
                        1 if pred_smc_up else 0,
                        1 if pred_smc_dn else 0,
                        1 if (not pred_smc_up and not pred_smc_dn) else 0,

                        # Vision AI фичи (повтор для RF)
                        float(pred_ai_rating),
                        0,
                        1 if pred_ai_ftr else 0,
                        1 if pred_sweep else 0,
                        1 if pred_div else 0,
                    ]]

                    # Проверяем и дополняем до нужной размерности
                    n_expected = len(feature_names)
                    n_got      = len(feat_new[0])
                    if n_got < n_expected:
                        feat_new[0] += [0.0] * (n_expected - n_got)
                    elif n_got > n_expected:
                        feat_new[0] = feat_new[0][:n_expected]

                    prob  = clf.predict_proba(feat_new)[0][1]
                    col_p = "🟢" if prob >= 0.6 else ("🟡" if prob >= 0.45 else "🔴")
                    st.metric(f"{col_p} Вероятность успеха", f"{prob:.1%}")
                    if prob >= 0.6:
                        st.success("ИИ: сильный паттерн по твоей истории → входи")
                    elif prob >= 0.45:
                        st.warning("ИИ: неоднозначно → жди лучшего подтверждения")
                    else:
                        st.error("ИИ: такой паттерн у тебя чаще убыточен → пропусти")

                    # Подсказка агента по настройке скринера
                    st.markdown("---")
                    st.markdown("**🤖 Рекомендация агента:**")
                    if has_rich and len(rich_trades) >= 20:
                        # Анализируем при каком score были победы
                        win_snaps  = [(t.get('snapshot') or {}) for t in rich_trades if t['result']=='win']
                        loss_snaps = [(t.get('snapshot') or {}) for t in rich_trades if t['result']=='loss']
                        win_ftr  = sum(1 for s in win_snaps  if s.get('in_ftr_zone')) / max(len(win_snaps),1)
                        loss_ftr = sum(1 for s in loss_snaps if s.get('in_ftr_zone')) / max(len(loss_snaps),1)
                        win_div  = sum(1 for s in win_snaps  if s.get('has_div_cvd')) / max(len(win_snaps),1)
                        loss_div = sum(1 for s in loss_snaps if s.get('has_div_cvd')) / max(len(loss_snaps),1)
                        win_abs  = sum(1 for s in win_snaps  if s.get('has_abs_bull') or s.get('has_abs_bear')) / max(len(win_snaps),1)

                        recs = []
                        if win_ftr > loss_ftr + 0.2:
                            recs.append(f"✅ FTR зоны работают для тебя ({win_ftr:.0%} побед в зоне)")
                        if win_div > loss_div + 0.15:
                            recs.append(f"✅ CVD дивергенции увеличивают winrate ({win_div:.0%} побед с дивом)")
                        if win_abs > 0.6:
                            recs.append(f"✅ Absorption — сильный сигнал в твоей системе ({win_abs:.0%})")
                        if not recs:
                            recs.append("📊 Накапливай сделки — агент даст рекомендации после 20+ закрытых")
                        for rec in recs:
                            st.markdown(rec)
                    else:
                        n_needed = max(0, 20 - len(rich_trades))
                        st.info(f"Агент даст рекомендации по настройке скринера после {n_needed} ещё сделок со снимком")

            except ImportError:
                st.info("Установи sklearn: `pip install scikit-learn`")

        # История сделок
        with st.expander("📋 История закрытых сделок"):
            for t in reversed(closed_trades):
                res_icon = "✅" if t['result']=='win' else "❌"
                _r = t.get('pnl_r', t.get('pnl_pct'))
                pnl_str  = f"{_r:+.2f}R  (≈{_r:+.1f}%)" if _r is not None else "—"
                st.markdown(f"{res_icon} **#{t['id']} {t['symbol']}** {t['direction']} [{t['tf']}] "
                            f"Entry: {t['entry']} → Exit: {t.get('exit_price','—')} | "
                            f"PnL: `{pnl_str}` | {t.get('closed_at','—')}")
                if t.get('note'):
                    st.caption(f"📝 {t['note']}")
    st.stop()

# ══════════════════════════════════════════════════════════════
#  ВКЛАДКА: ПОТЕНЦИАЛЬНЫЕ СДЕЛКИ
# ══════════════════════════════════════════════════════════════
if "Радар" in tab:
    st.header("📡 Потенциальные сделки")
    st.markdown("Активы где цена **ещё не в зоне**, но приближается к ней.")

    ALL_SYMBOLS = (
        ["BTCUSDT"] +
        ["XRPUSDT","XLMUSDT","SOLUSDT","TRXUSDT","BCHUSDT","LTCUSDT",
         "AAVEUSDT","BNBUSDT","AVAXUSDT","ETCUSDT","LINKUSDT","DOGEUSDT",
         "NEARUSDT","ATOMUSDT","UNIUSDT","POLUSDT","ADAUSDT","DOTUSDT",
         "ETHUSDT","OPUSDT","FILUSDT","ARBUSDT"] +
        ["CAKEUSDT","WOOUSDT","VETUSDT","ZILUSDT","STXUSDT","WAVESUSDT",
         "ZRXUSDT","XTZUSDT","JASMYUSDT","ORDIUSDT","TONUSDT","INJUSDT",
         "SEIUSDT","PEOPLEUSDT","FLOKIUSDT","YGGUSDT","GALAUSDT","RUNEUSDT",
         "CRVUSDT","ICPUSDT","MASKUSDT","APEUSDT","SUIUSDT","APTUSDT",
         "MANAUSDT","THETAUSDT","GMTUSDT","HBARUSDT","SANDUSDT","KSMUSDT"] +
        ["ANKRUSDT","ASTRUSDT","RVNUSDT","GASUSDT","BOMEUSDT","ONTUSDT",
         "MEMEUSDT","ARUSDT","BANDUSDT","CKBUSDT","KAVAUSDT","BLURUSDT",
         "1000PEPEUSDT","GLMUSDT","1000BONKUSDT","PEPEUSDT","WIFUSDT",
         "POPCATUSDT","DYDXUSDT","SKLUSDT","AGIUSDT","IMXUSDT","LRCUSDT",
         "MUBARAKUSDT","SUSHIUSDT","XVSUSDT","MAGICUSDT","ZETAUSDT",
         "ENSUSDT","PENGUUSDT","COTIUSDT"]
    )

    col_tf, col_dist = st.columns(2)
    pot_tf   = col_tf.radio("Таймфрейм:", ["15m","1h","1D"],
                             horizontal=True, key="pot_tf")
    max_dist = col_dist.slider("Макс. расстояние до зоны (%)",
                                0.3, 5.0, 2.0, 0.1, key="pot_dist")

    if st.button("📡 Запустить радар", key="run_potential"):
        pot_results = []
        prog2 = st.progress(0)
        for idx_p, sym in enumerate(ALL_SYMBOLS):
            prog2.progress((idx_p+1)/len(ALL_SYMBOLS))
            r = scan_potential(sym, pot_tf, "crypto", max_dist_pct=max_dist)
            if r: pot_results.append(r)
        prog2.empty()

        # Сортируем: сначала по тренду, потом по расстоянию
        pot_results.sort(key=lambda x: (
            0 if "По тренду" in x['priority'] else 1,
            x['candidates'][0]['dist_pct']
        ))
        st.session_state['pot_results'] = pot_results

    if 'pot_results' in st.session_state and st.session_state['pot_results']:
        res = st.session_state['pot_results']
        st.success(f"Найдено активов: {len(res)}")

        # Разбиваем на по тренду и против тренда
        by_trend    = [r for r in res if "По тренду"    in r['priority']]
        against_trend = [r for r in res if "Против тренда" in r['priority']]

        if by_trend:
            st.subheader(f"🔥 По тренду ({len(by_trend)})")
            for r in by_trend:
                best_z = r['candidates'][0]
                dir_emoji = "🟢" if best_z['trade_dir']==1 else "🔴"
                ctx_emoji = "⚖️" if r['context']=="BALANCE" else "📈"
                with st.expander(
                    f"{dir_emoji} {r['symbol']}  {ctx_emoji} {r['context']}  "
                    f"— осталось {best_z['dist_pct']}%  до {best_z['zone_type']}"
                ):
                    for z in r['candidates']:
                        dir_e = "🟢 Long" if z['trade_dir']==1 else "🔴 Short"
                        arrow = "→ 🎯" if z.get('going_to_zone') else "↩ (не идёт)"
                        momentum_str = f"импульс: {'🚀' if z.get('momentum',1)>1.5 else ('🐢' if z.get('momentum',1)<0.5 else '➡️')} {z.get('momentum','?')}x ATR"
                        st.markdown(
                            f"**{z['zone_type']}** {dir_e} {arrow} — "
                            f"расстояние: `{z['dist_pct']}%` | "
                            f"зона: `{z['zone_lo']:.4f} – {z['zone_hi']:.4f}` | {momentum_str}"
                        )
                        st.caption(f"↳ {z['approach']}")
                    st.caption(
                        f"Сценарий: **{r.get('scenario','—')}** | "
                        f"Цена: {r['price']:.5f} | {r['timestamp']}"
                    )
                    if st.button(f"📌 Следить за {r['symbol']}", key=f"watch_{r['symbol']}"):
                        watches = st.session_state.get('watchlist', [])
                        if r['symbol'] not in watches:
                            watches.append(r['symbol'])
                            st.session_state['watchlist'] = watches
                            st.success(f"{r['symbol']} добавлен в список наблюдения")

        if against_trend:
            st.subheader(f"⚠️ Против тренда ({len(against_trend)}) — осторожно")
            for r in against_trend:
                best_z = r['candidates'][0]
                dir_emoji = "🟢" if best_z['trade_dir']==1 else "🔴"
                with st.expander(
                    f"{dir_emoji} {r['symbol']}  "
                    f"— осталось {best_z['dist_pct']}%  до {best_z['zone_type']}"
                ):
                    for z in r['candidates']:
                        dir_e = "🟢 Long" if z['trade_dir']==1 else "🔴 Short"
                        st.markdown(
                            f"**{z['zone_type']}** {dir_e} — "
                            f"расстояние: `{z['dist_pct']}%` | "
                            f"зона: `{z['zone_lo']:.4f} – {z['zone_hi']:.4f}`"
                        )
                        st.caption(f"↳ {z['approach']}")
                    st.caption(f"Цена: {r['price']:.5f} | {r['timestamp']}")

    elif 'pot_results' in st.session_state:
        st.info("Активов вблизи зон не найдено — попробуй увеличить расстояние")

    # ══════════════════════════════════════════════════════════════
    #  CONTEXT SCANNER — MTF анализ + FTR Absorption
    # ══════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🗺 Context Scanner — Decision Points (1D→1H→15m)")
    st.markdown("Ищет активы где цена идёт **к зоне FTR** и/или есть **Absorption** у зоны")

    ALL_CTX = list(dict.fromkeys(["BTCUSDT"] + LIST_ALTS_MAIN + LIST_ALTS_2 + LIST_ALTS_3))
    col_ctx1, col_ctx2 = st.columns(2)
    ctx_thresh   = col_ctx1.slider("Порог приближения (%)", 0.3, 3.0, 0.6, 0.1, key="ctx_thr2")
    ctx_tf_filt  = col_ctx2.multiselect("ТФ зон:", ["1D","1H"], default=["1D","1H"], key="ctx_tf2")

    if st.button("🗺 Запустить Context Scan", key="run_ctx2"):
        ctx_results = []
        prog3 = st.progress(0)
        for idx_c, sym in enumerate(ALL_CTX):
            prog3.progress((idx_c+1)/len(ALL_CTX))
            try:
                r = scan_context(sym)
                if not r: continue
                # Фильтруем зоны
                fz = [z for z in r['approaching'] if z['tf'] in ctx_tf_filt and z['dist'] <= ctx_thresh]
                has_abs = bool(r.get('ftr_abs'))
                if fz or has_abs:
                    r['approaching'] = fz[:2]
                    ctx_results.append(r)
            except: pass
        prog3.empty()
        # Сортируем: сначала с absorption, потом по близости
        ctx_results.sort(key=lambda x: (
            0 if x.get('ftr_abs') else 1,
            x['approaching'][0]['dist'] if x['approaching'] else 999
        ))
        st.session_state['ctx_results2'] = ctx_results

    if 'ctx_results2' in st.session_state and st.session_state['ctx_results2']:
        res = st.session_state['ctx_results2']
        abs_results  = [r for r in res if r.get('ftr_abs')]
        zone_results = [r for r in res if not r.get('ftr_abs') and r.get('approaching')]

        if abs_results:
            st.subheader(f"🔴 FTR Absorption (войди и проверь!) — {len(abs_results)}")
            for r in abs_results:
                abs_sig = r['ftr_abs'][0]
                with st.expander(
                    f"{abs_sig['emoji']} {r['symbol']} — {abs_sig['type']} | "
                    f"Δ {abs_sig['delta']:+.0f}"
                ):
                    c1,c2,c3 = st.columns(3)
                    c1.metric("1D", r['phase_d1'])
                    c2.metric("1H", r['phase_h1'])
                    c3.metric("15m", r['phase_m15'])
                    z = abs_sig['zone']
                    st.markdown(f"**Зона FTR:** `{z['zl']:.4f} – {z['zh']:.4f}`")
                    st.caption(f"Цена: {r['price']:.5f} | {r['timestamp']}")

        if zone_results:
            st.subheader(f"🟡 Приближение к зонам — {len(zone_results)}")
            for r in zone_results:
                if not r['approaching']: continue
                _render_ctx_card(r)

    elif 'ctx_results2' in st.session_state:
        st.info("Нет результатов — попробуй увеличить порог")
    st.stop()

def _render_ctx_card(r):
    """Рендерит карточку Context сигнала в интерфейсе."""
    z0 = r['approaching'][0]
    dir_arrow = "🟢" if z0['dir']==1 else ("🔴" if z0['dir']==-1 else "🔵")
    sot_warn  = " ⚠️SOT" if r.get('sot_d1') or r.get('sot_h1') else ""
    with st.expander(
        f"{r['symbol']}  {dir_arrow}  "
        f"{z0['tf']} {z0['zone_type']} — {z0['dist']:.2f}%  "
        f"| {r['phase_d1']} / {r['phase_h1']}{sot_warn}"
    ):
        # Фазы рынка
        c1, c2, c3 = st.columns(3)
        c1.metric("1D", r['phase_d1'])
        c2.metric("1H", r['phase_h1'])
        c3.metric("15m", r['phase_m15'])

        # Совпадение трендов
        st.markdown(f"**{r['alignment']}**")

        if r.get('bal_phase_h1') and r['bal_phase_h1'] != 'NEUTRAL':
            bp_col = {"ACCUMULATION":"🟢","DISTRIBUTION":"🔴","COMPRESSED":"🟡"}.get(r['bal_phase_h1'],"⚪")
            st.markdown(f"{bp_col} H1 баланс: **{r['bal_phase_h1']}**")

        if r.get('sot_d1'): st.warning("⚠️ SOT на D1 — тренд D1 слабеет")
        if r.get('sot_h1'): st.warning("⚠️ SOT на H1 — тренд H1 слабеет")

        # Зоны
        st.markdown("**📍 Приближение к зонам:**")
        for z in r['approaching']:
            arrow = "🟢" if z['dir']==1 else ("🔴" if z['dir']==-1 else "🔵")
            st.markdown(
                f"{z['strength']} **{z['tf']}** {arrow} {z['zone_type']} | "
                f"`{z['zl']:.4f} – {z['zh']:.4f}` | до зоны: **{z['dist']:.2f}%**"
            )

        # Подсказка
        if r.get('hint_d1'):
            st.caption(f"💡 {r['hint_d1']}")
        st.caption(f"Цена: {r['price']:.5f} | {r['timestamp']}")

        # Кнопка открыть сделку
        if st.button(f"📌 Следить за {r['symbol']}", key=f"ctx_watch_{r['symbol']}"):
            watches = st.session_state.get('ctx_watchlist', [])
            if r['symbol'] not in watches:
                watches.append(r['symbol'])
                st.session_state['ctx_watchlist'] = watches
                st.success(f"{r['symbol']} добавлен в список наблюдения")


# ══════════════════════════════════════════════════════════════
#  ВКЛАДКА: CONTEXT SCANNER
# ══════════════════════════════════════════════════════════════
if "NEVER_MATCH_THIS_OLD_TAB" in tab:
    st.header("🗺 Context Scanner — MTF анализ приближения к зонам")
    st.markdown("Сканирует **1D → 1H → 15m** и находит где цена приближается к зонам FTR / VAH / VAL / POC")

    ALL_CTX = list(dict.fromkeys(["BTCUSDT"] + LIST_ALTS_MAIN + LIST_ALTS_2 + LIST_ALTS_3))
    ctx_tf_info = st.empty()

    col1, col2 = st.columns(2)
    ctx_thresh = col1.slider("Порог приближения (%)", 0.5, 5.0, 2.0, 0.1, key="ctx_thresh")
    ctx_tf_filter = col2.multiselect("Таймфреймы зон:", ["1D","1H"], default=["1D","1H"], key="ctx_tff")

    if st.button("🗺 Запустить Context Scan", key="run_ctx"):
        ctx_results = []
        prog = st.progress(0)
        for idx_c, sym in enumerate(ALL_CTX):
            prog.progress((idx_c+1)/len(ALL_CTX))
            try:
                r = scan_context(sym)
                if r:
                    # Фильтруем по выбранным ТФ и пользовательскому порогу
                    filtered_zones = [
                        z for z in r['approaching']
                        if z['tf'] in ctx_tf_filter and z['dist'] <= ctx_thresh
                    ]
                    if filtered_zones:
                        r['approaching'] = filtered_zones[:2]  # max 2 зоны
                        ctx_results.append(r)
            except Exception:
                pass
        prog.empty()

        # Сортируем: D1 зоны сначала, потом по близости
        ctx_results.sort(key=lambda x: (
            0 if any(z['tf']=='1D' for z in x['approaching']) else 1,
            x['approaching'][0]['dist']
        ))
        st.session_state['ctx_results'] = ctx_results

    if 'ctx_results' in st.session_state and st.session_state['ctx_results']:
        res = st.session_state['ctx_results']
        st.success(f"Найдено активов: {len(res)}")

        # Группируем по совпадению трендов
        aligned   = [r for r in res if "совпадают" in r.get('alignment','')]
        unaligned = [r for r in res if "совпадают" not in r.get('alignment','')]

        if aligned:
            st.subheader(f"✅ Тренды совпадают на D1+H1 ({len(aligned)})")
            for r in aligned:
                _render_ctx_card(r)

        if unaligned:
            st.subheader(f"⚠️ Разные фазы ({len(unaligned)})")
            for r in unaligned:
                _render_ctx_card(r)

    elif 'ctx_results' in st.session_state:
        st.info("Активов вблизи зон не найдено — увеличь порог")
    st.stop()

# ══════════════════════════════════════════════════════════════
#  ВКЛАДКА 17: FOREX — Живые графики через Yahoo Finance
# ══════════════════════════════════════════════════════════════
if "FOREX" in tab:
    st.header("FOREX — Графики инструментов")

    _FX_PAIRS = {
        'EUR/USD': 'EURUSD=X',
        'GBP/USD': 'GBPUSD=X',
        'USD/JPY': 'USDJPY=X',
        'USD/CHF': 'USDCHF=X',
        'USD/CAD': 'USDCAD=X',
        'AUD/USD': 'AUDUSD=X',
        'NZD/USD': 'NZDUSD=X',
        'EUR/GBP': 'EURGBP=X',
        'EUR/JPY': 'EURJPY=X',
        'GBP/JPY': 'GBPJPY=X',
        'AUD/JPY': 'AUDJPY=X',
        'XAU/USD': 'GC=F',
    }

    # Параметры yfinance для каждого TF
    # 3m: скачиваем 1m за 5 дней → ресемплируем в 3m (yfinance не имеет 3m)
    # 15m: 60 дней (максимум для 15m у yfinance)
    # 1h: 2 года
    # 1D: полная история
    _FX_YF_PARAMS = {
        '3m':  ('1m',  '5d'),
        '15m': ('15m', '60d'),
        '1h':  ('1h',  '2y'),
        '1D':  ('1d',  'max'),
    }

    @st.cache_data(ttl=180, show_spinner=False)
    def _fetch_forex_live(ticker, tf):
        yf_interval, yf_period = _FX_YF_PARAMS[tf]
        try:
            raw = yf.download(ticker, interval=yf_interval, period=yf_period,
                              auto_adjust=True, progress=False)
            if raw.empty:
                return pd.DataFrame()
            raw = raw.reset_index()
            raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
            raw = raw.rename(columns={
                'Datetime': 'timestamp', 'Date': 'timestamp',
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume',
            })
            raw['timestamp'] = pd.to_datetime(raw['timestamp'], utc=True).dt.tz_localize(None)
            raw = raw[['timestamp', 'open', 'high', 'low', 'close', 'volume']].dropna()
            raw = raw.sort_values('timestamp').reset_index(drop=True)
            # Ресемплируем 1m → 3m
            if tf == '3m':
                raw = raw.set_index('timestamp')
                ohlcv = raw['close'].resample('3min').ohlc()
                ohlcv['volume'] = raw['volume'].resample('3min').sum()
                ohlcv = ohlcv.dropna(subset=['open']).reset_index()
                ohlcv.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                return ohlcv
            return raw
        except Exception:
            return pd.DataFrame()

    c1, c2 = st.columns([3, 1])
    with c1:
        fx_pair = st.selectbox("Инструмент:", list(_FX_PAIRS.keys()), key="fx_pair_sel")
    with c2:
        fx_tf = st.radio("Таймфрейм:", ["3m", "15m", "1h", "1D"], key="fx_tf_sel")

    yf_ticker = _FX_PAIRS[fx_pair]

    with st.spinner(f"Загружаю {fx_pair} {fx_tf}..."):
        df_fx = _fetch_forex_live(yf_ticker, fx_tf)

    if df_fx.empty:
        st.warning(f"Нет данных для {fx_pair} {fx_tf}. Yahoo Finance недоступен или пара не найдена.")
        if st.button("Повторить"):
            st.cache_data.clear()
            st.rerun()
        st.stop()

    # Показываем последние N свечей
    _N = {"3m": 480, "15m": 384, "1h": 500, "1D": 500}.get(fx_tf, 500)
    df_plot = df_fx.tail(_N).reset_index(drop=True)

    fig_fx = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.78, 0.22],
    )

    fig_fx.add_trace(go.Candlestick(
        x=df_plot['timestamp'],
        open=df_plot['open'], high=df_plot['high'],
        low=df_plot['low'],   close=df_plot['close'],
        name=fx_pair,
        increasing_line_color='#00E676',
        decreasing_line_color='#FF1744',
    ), 1, 1)

    if 'volume' in df_plot.columns:
        fig_fx.add_trace(go.Bar(
            x=df_plot['timestamp'],
            y=df_plot['volume'],
            name="Volume",
            marker_color='rgba(100,180,255,0.4)',
        ), 2, 1)

    # ZigZag на форекс-графике
    try:
        _zz_fx = calc_zigzag(df_plot)
        for _tr in _zigzag_traces(_zz_fx):
            fig_fx.add_trace(_tr, 1, 1)
    except Exception:
        pass

    fig_fx.update_layout(
        height=680,
        template='plotly_dark',
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        xaxis_rangeslider_visible=False,
        dragmode='pan',
        hovermode='x unified',
        title=dict(text=f"{fx_pair}  {fx_tf}", font=dict(size=16)),
        margin=dict(l=10, r=10, t=40, b=10),
        hoverlabel=dict(bgcolor='rgba(30,30,40,0.95)', font_size=12, font_family="monospace"),
    )
    fig_fx.update_xaxes(
        gridcolor='#1f2937', showgrid=True,
        showspikes=True, spikecolor="white",
        spikemode="across", spikedash="dash", spikethickness=1,
    )
    fig_fx.update_yaxes(gridcolor='#1f2937', showgrid=True)

    st.plotly_chart(fig_fx, use_container_width=True, config={
        'scrollZoom': True,
        'modeBarButtonsToAdd': ['drawline', 'drawrect', 'eraseshape'],
    })

    _ts_min = df_plot['timestamp'].iloc[0]
    _ts_max = df_plot['timestamp'].iloc[-1]
    _src_note = "1m→3m resample" if fx_tf == "3m" else "Yahoo Finance"
    st.caption(
        f"Источник: {_src_note}  |  "
        f"Период: {_ts_min.strftime('%d.%m.%Y')} — {_ts_max.strftime('%d.%m.%Y')}  |  "
        f"Свечей: {len(df_plot)}  |  Обновление: каждые 3 мин"
    )
    st.stop()



# Лабораторные вкладки имеют собственный рендер в конце файла
# Основной рендер выполняется только для вкладок 1-16
if tab not in _LAB_TABS:
    # Загружаем данные только для графических вкладок
    df     = fetch_main_data(asset, tf, is_renko, src)
    levels = fetch_daily_levels(asset, src) if not is_renko else None

    if df.empty:
        st.warning(f"Данные не загружены... (asset={asset}, tf={tf}, src={src})")
        if st.button("🔄 Повторить загрузку"):
            st.cache_data.clear()
            st.rerun()
        st.stop()

# ══════════════════════════════════════════════════════════════
#  MTF CONFLUENCE — функции расчёта и отображения
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=60, show_spinner=False)
def _calc_tf_signals(symbol: str, tf: str, src: str) -> dict:
    """
    Рассчитывает все сигналы для одного ТФ.
    Возвращает dict с сигналами и итоговым score (-10..+10).
    """
    try:
        df = fetch_main_data(symbol, tf, False, src)
        if df.empty or len(df) < 50:
            return {}
        df = apply_order_flow(df)
    except:
        return {}

    signals = {}
    last_close = float(df['close'].iloc[-1])

    # ── 1. Фаза рынка (ZigZag + vol_ratio) ───────────────────────────────────
    try:
        _pv = calc_zigzag(df)
        _mp = calc_market_phase(df, _pv, tf)
        signals['phase']    = _mp['market_phase']
        signals['vol_ratio'] = round(_mp['vol_ratio'], 2)
        signals['pivot_comp'] = _mp['pivot_compression']
    except:
        signals['phase'] = 'unknown'

    # ── 2. dPOC направление ──────────────────────────────────────────────────
    try:
        _dpoc = calc_dynamic_poc(df, tf=tf)
        _valid = [x for x in _dpoc if not np.isnan(x)]
        if len(_valid) >= 4:
            if _valid[-1] > _valid[-4] + 1e-8:
                signals['dpoc_dir'] = 'up'
            elif _valid[-1] < _valid[-4] - 1e-8:
                signals['dpoc_dir'] = 'down'
            else:
                signals['dpoc_dir'] = 'flat'
            signals['dpoc_val']       = round(_valid[-1], 5)
            signals['dpoc_vs_price']  = 'above' if last_close >= _valid[-1] else 'below'
        else:
            signals['dpoc_dir'] = 'unknown'
    except:
        signals['dpoc_dir'] = 'unknown'

    # ── 3. Цена vs Value Area (POC/VAH/VAL) ──────────────────────────────────
    try:
        _vah, _val, _poc, _, _, _ = get_profile_for_tf(df, tf)
        if _vah and _val and _poc:
            if last_close > float(_vah):
                signals['price_va'] = 'above_vah'
            elif last_close < float(_val):
                signals['price_va'] = 'below_val'
            elif last_close > float(_poc):
                signals['price_va'] = 'above_poc'
            else:
                signals['price_va'] = 'below_poc'
        else:
            signals['price_va'] = 'unknown'
    except:
        signals['price_va'] = 'unknown'

    # ── 4. BOS / CHoCH последнее направление ─────────────────────────────────
    try:
        _bos, _obs, _fvgs = find_bos_ob_fvg(df)
        if _bos:
            _lb = _bos[-1]
            signals['bos_dir']   = 'bull' if _lb['dir'] == 1 else 'bear'
            signals['bos_choch'] = bool(_lb.get('choch', False))
        else:
            signals['bos_dir'] = 'unknown'
        signals['_obs']  = _obs
        signals['_fvgs'] = _fvgs
    except:
        signals['bos_dir'] = 'unknown'
        signals['_obs']  = []
        signals['_fvgs'] = []

    # ── 5. Цена в зоне OB? ───────────────────────────────────────────────────
    _in_ob = None
    for _ob in signals.get('_obs', [])[-6:]:
        if _ob['dir'] == 1 and not (_ob['y0'] > last_close or _ob['y1'] < last_close):
            _in_ob = 'demand'; break
        elif _ob['dir'] == -1 and not (_ob['y0'] > last_close or _ob['y1'] < last_close):
            _in_ob = 'supply'; break
    signals['in_ob'] = _in_ob

    # ── 6. Цена в зоне FTR? ──────────────────────────────────────────────────
    _in_ftr = None
    try:
        _zones = calc_ftr_zones(df, **get_ftr_params(symbol))
        for _z in _zones:
            if _z.get('active') and _z['zl'] <= last_close <= _z['zh']:
                _in_ftr = 'demand' if _z['dir'] == 1 else 'supply'
                break
    except:
        pass
    signals['in_ftr'] = _in_ftr

    # ── 7. CVD-дивергенция ────────────────────────────────────────────────────
    try:
        # zz_pivots = _pv из шага 1 — те же hard-пивоты что Key H/L (единая методология)
        try:
            _zz_for_div = _pv  # определён в шаге 1
        except NameError:
            _zz_for_div = None
        _divs = find_divergences(df, lookback=5, min_dist=8, zz_pivots=_zz_for_div)
        signals['cvd_div'] = _divs[-1]['type'] if _divs else None
    except:
        signals['cvd_div'] = None

    # ── 8. HVN — цена в зоне высокого объёма? ────────────────────────────────
    try:
        _hvns = find_hvn_zones(df, tf=tf, n_zones=3)
        _in_hvn = any(
            abs(last_close - h['price']) <= h['width'] * 5
            for h in _hvns
        )
        signals['in_hvn'] = _in_hvn
    except:
        signals['in_hvn'] = False

    # ── Итоговый счёт ─────────────────────────────────────────────────────────
    score = 0

    # Фаза
    if signals.get('phase') == 'breakout_imminent': score += 1
    elif signals.get('phase') == 'active': score += 1

    # dPOC
    if signals.get('dpoc_dir') == 'up':   score += 1
    elif signals.get('dpoc_dir') == 'down': score -= 1
    if signals.get('dpoc_vs_price') == 'above': score += 1
    elif signals.get('dpoc_vs_price') == 'below': score -= 1

    # Value Area
    va = signals.get('price_va', 'unknown')
    if va == 'below_val':   score += 2   # перепроданность — бычий потенциал
    elif va == 'above_vah': score -= 2   # перекупленность — медвежий потенциал
    elif va == 'above_poc': score += 1
    elif va == 'below_poc': score -= 1

    # BOS
    if signals.get('bos_dir') == 'bull':  score += 1
    elif signals.get('bos_dir') == 'bear': score -= 1
    if signals.get('bos_choch'):          score += 1  # CHoCH — сильный сигнал

    # OB / FTR
    if signals.get('in_ob') == 'demand':  score += 1
    elif signals.get('in_ob') == 'supply': score -= 1
    if signals.get('in_ftr') == 'demand': score += 1
    elif signals.get('in_ftr') == 'supply': score -= 1

    # CVD
    if signals.get('cvd_div') == 'bull':  score += 2
    elif signals.get('cvd_div') == 'bear': score -= 2

    # HVN
    if signals.get('in_hvn'): score += 1  # стоим у зоны интереса

    signals['score'] = score
    return signals


def _render_mtf_confluence(asset: str, src: str, current_tf: str):
    """Отображает панель MTF-конфлюэнса под основным графиком."""

    _TFS = ['15m', '1h', '1D']
    _TF_LABELS = {'15m': '15 мин', '1h': '1 час', '1D': '1 день'}

    _PHASE_ICONS = {
        'compression':       '🔵 Сжатие',
        'breakout_imminent': '🟣 Пробой',
        'quiet_trend':       '⚪ Тренд',
        'active':            '🟢 Актив',
        'unknown':           '❓',
    }
    _DIR_ICONS = {'up': '↑', 'down': '↓', 'flat': '→', 'unknown': '—'}
    _VA_LABELS = {
        'above_vah': '▲ выше VAH',
        'below_val': '▼ ниже VAL',
        'above_poc': '↑ выше POC',
        'below_poc': '↓ ниже POC',
        'unknown':   '—',
    }

    def _cell(val, bull_val=None, bear_val=None):
        """Возвращает (текст, цвет_css) для ячейки."""
        if val is None or val == 'unknown': return '—', '#555'
        if val == bull_val: return str(val), '#00C853'
        if val == bear_val: return str(val), '#FF1744'
        return str(val), '#BBBBBB'

    with st.expander("📊 MTF Confluence — мультитаймфреймовый анализ", expanded=True):
        data = {}
        for _tf in _TFS:
            with st.spinner(f'Загружаю {_tf}…') if _tf != current_tf else _dummy_ctx():
                data[_tf] = _calc_tf_signals(asset, _tf, src)

        # Заголовки колонок
        cols = st.columns([2.2, 1, 1, 1])
        cols[0].markdown("**Сигнал**")
        for i, _tf in enumerate(_TFS):
            _score = data[_tf].get('score', 0) if data[_tf] else 0
            _col = '#00C853' if _score > 2 else ('#FF1744' if _score < -2 else '#FFD600')
            cols[i+1].markdown(
                f"**{_TF_LABELS[_tf]}**<br>"
                f"<span style='font-size:18px;color:{_col};font-weight:bold'>"
                f"{'▲' if _score>0 else ('▼' if _score<0 else '●')} {_score:+d}"
                f"</span>",
                unsafe_allow_html=True
            )

        st.divider()

        # Строки таблицы
        _ROWS = [
            ("🌀 Фаза рынка",   lambda s: (_PHASE_ICONS.get(s.get('phase','unknown'),'❓'), '#BBBBBB')),
            ("📈 dPOC миграция",lambda s: (
                f"{_DIR_ICONS.get(s.get('dpoc_dir','unknown'),'—')} {s.get('dpoc_dir','—')}",
                '#00C853' if s.get('dpoc_dir')=='up' else ('#FF1744' if s.get('dpoc_dir')=='down' else '#FFD600')
            )),
            ("💰 Цена vs dPOC", lambda s: (
                s.get('dpoc_vs_price','—'),
                '#00C853' if s.get('dpoc_vs_price')=='above' else '#FF1744'
            )),
            ("📊 Value Area",   lambda s: (_VA_LABELS.get(s.get('price_va','unknown'),'—'), '#BBBBBB')),
            ("🔨 BOS / CHoCH",  lambda s: (
                ('CHoCH ↑' if s.get('bos_choch') else 'BOS ↑') if s.get('bos_dir')=='bull'
                else (('CHoCH ↓' if s.get('bos_choch') else 'BOS ↓') if s.get('bos_dir')=='bear' else '—'),
                '#00C853' if s.get('bos_dir')=='bull' else ('#FF1744' if s.get('bos_dir')=='bear' else '#555')
            )),
            ("🟦 OB зона",      lambda s: (
                s.get('in_ob') or '—',
                '#00C853' if s.get('in_ob')=='demand' else ('#FF1744' if s.get('in_ob')=='supply' else '#555')
            )),
            ("🔶 FTR зона",     lambda s: (
                s.get('in_ftr') or '—',
                '#00C853' if s.get('in_ftr')=='demand' else ('#FF1744' if s.get('in_ftr')=='supply' else '#555')
            )),
            ("📉 CVD дивергенция", lambda s: (
                s.get('cvd_div') or '—',
                '#00C853' if s.get('cvd_div')=='bull' else ('#FF1744' if s.get('cvd_div')=='bear' else '#555')
            )),
            ("🔆 HVN зона",     lambda s: (
                'да' if s.get('in_hvn') else '—',
                '#FFD600' if s.get('in_hvn') else '#555'
            )),
        ]

        for row_label, row_fn in _ROWS:
            row_cols = st.columns([2.2, 1, 1, 1])
            row_cols[0].markdown(f"<small>{row_label}</small>", unsafe_allow_html=True)
            for i, _tf in enumerate(_TFS):
                _s = data[_tf] or {}
                try:
                    _txt, _col = row_fn(_s)
                except:
                    _txt, _col = '—', '#555'
                row_cols[i+1].markdown(
                    f"<span style='color:{_col};font-size:12px'>{_txt}</span>",
                    unsafe_allow_html=True
                )

        st.divider()

        # Итоговый вывод
        _scores = [data[_tf].get('score', 0) for _tf in _TFS if data[_tf]]
        _total  = sum(_scores)
        _max    = len(_ROWS) * len(_TFS)

        if _total >= 6:
            _verdict = "🟢 БЫЧИЙ КОНФЛЮЭНС — высокая вероятность роста"
            _vc = '#00C853'
        elif _total <= -6:
            _verdict = "🔴 МЕДВЕЖИЙ КОНФЛЮЭНС — высокая вероятность падения"
            _vc = '#FF1744'
        elif _total >= 3:
            _verdict = "🟡 Умеренно бычий — неплохие условия для лонга"
            _vc = '#FFD600'
        elif _total <= -3:
            _verdict = "🟠 Умеренно медвежий — неплохие условия для шорта"
            _vc = '#FF6D00'
        else:
            _verdict = "⚪ Нейтрально — рынок в равновесии, ждём сигнала"
            _vc = '#9E9E9E'

        st.markdown(
            f"<div style='text-align:center;padding:8px;border-radius:6px;"
            f"background:rgba(255,255,255,0.05);border:1px solid {_vc}'>"
            f"<span style='color:{_vc};font-size:14px;font-weight:bold'>{_verdict}</span><br>"
            f"<span style='color:#888;font-size:11px'>Суммарный счёт: {_total:+d} | "
            f"15m: {_scores[0]:+d}  1h: {_scores[1] if len(_scores)>1 else 0:+d}  "
            f"1D: {_scores[2] if len(_scores)>2 else 0:+d}</span>"
            f"</div>",
            unsafe_allow_html=True
        )


from contextlib import contextmanager as _ctxmgr
@_ctxmgr
def _dummy_ctx():
    yield


# ══════════════════════════════════════════════════════════════
#  СВЕЧНОЙ РЕЖИМ (только вкладки 1-17)
# ══════════════════════════════════════════════════════════════
if tab not in _LAB_TABS and not is_renko:
    df = apply_order_flow(df)
    # Параметры FTR подобраны под конкретные пары по оригиналу AlgoPoint
    ALTS_MAIN = ["XRPUSDT","XLMUSDT","SOLUSDT","TRXUSDT","BCHUSDT","LTCUSDT",
                 "AAVEUSDT","BNBUSDT","AVAXUSDT","ETCUSDT","LINKUSDT","DOGEUSDT",
                 "NEARUSDT","ATOMUSDT","UNIUSDT","POLUSDT","ADAUSDT","DOTUSDT",
                 "ETHUSDT","OPUSDT","FILUSDT","ARBUSDT"]
    # Единые параметры FTR — та же функция что в сканерах
    zones = calc_ftr_zones(df, **get_ftr_params(asset))

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.015,
        row_heights=[0.55, 0.17, 0.15, 0.13],
        subplot_titles=("Price","CVD","Volume","Delta")
    )

    # — Свечи (дата/время в hover)
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'], high=df['high'],
        low=df['low'],   close=df['close'],
        name="Price",
        increasing_line_color='#00E676', decreasing_line_color='#FF1744',
        hovertext=[str(t)[:16] for t in df['timestamp']],
    ), 1, 1)

    # — PDH / PDL
    if levels:
        fig.add_hline(y=levels['pdh'], line=dict(color="orange",width=1,dash="dot"),
                      annotation_text="PDH", row=1, col=1)
        fig.add_hline(y=levels['pdl'], line=dict(color="orange",width=1,dash="dot"),
                      annotation_text="PDL", row=1, col=1)

    # — FTR зоны: только активные (ghost деактивированные не рисуем — безымянные прямоугольники)
    for s in ftr_shapes(df, zones):
        if not s['active']:
            continue  # пропускаем деактивированные FTR — они создавали безымянные прямоугольники
        is_bull  = s['dir'] == 1
        fill_c   = "rgba(0,230,118,0.18)"  if is_bull else "rgba(255,50,50,0.18)"
        border_c = "rgba(0,230,118,0.80)"  if is_bull else "rgba(255,50,50,0.80)"
        fig.add_shape(type="rect",
                      x0=s['x0'], x1=s['x1'], y0=s['y0'], y1=s['y1'],
                      fillcolor=fill_c,
                      line=dict(color=border_c, width=1, dash='solid'),
                      row=1, col=1)
        label_txt = "FTR D" if is_bull else "FTR S"
        fig.add_annotation(x=s['x0'], y=s['y1'] if is_bull else s['y0'],
                            text=label_txt, showarrow=False,
                            font=dict(color=border_c, size=9),
                            xanchor='left',
                            yanchor='bottom' if is_bull else 'top',
                            row=1, col=1)

    # ── ZigZag + Market Phase — всегда на графике (независимо от MDE) ─────────
    try:
        _zz_pivots = calc_zigzag(df)
        for _tr in _zigzag_traces(_zz_pivots):
            fig.add_trace(_tr, row=1, col=1)

        # Метка фазы рынка (Дядя Миша) в левом верхнем углу
        _mp = calc_market_phase(df, _zz_pivots, tf)
        _phase_labels = {
            'compression':       ('🔵 Сжатие',          'rgba(33,150,243,0.85)'),
            'breakout_imminent': ('🟣 Объём пошёл',      'rgba(156,39,176,0.85)'),
            'quiet_trend':       ('⚪ Тихий тренд',      'rgba(96,125,139,0.85)'),
            'active':            ('🟢 Активная фаза',   'rgba(76,175,80,0.85)'),
        }
        _ph_txt, _ph_col = _phase_labels.get(_mp['market_phase'], ('', 'grey'))
        _n_hard = sum(1 for p in _zz_pivots if p.get('confirmed') and p.get('quality') == 'hard')
        _n_soft = sum(1 for p in _zz_pivots if p.get('confirmed') and p.get('quality') == 'soft')
        fig.add_annotation(
            xref='paper', yref='paper',
            x=0.01, y=0.99,
            text=(f"<b>{_ph_txt}</b>  "
                  f"<span style='color:#9C27B0'>▼{_n_hard} твёрд.</span>  "
                  f"<span style='color:#757575'>●{_n_soft} мягк.</span>  "
                  f"vol×{_mp['vol_ratio']:.2f}"),
            showarrow=False,
            font=dict(size=11, color='white'),
            bgcolor=_ph_col,
            borderpad=4,
            xanchor='left', yanchor='top',
        )
    except Exception as _zze:
        import traceback
        print(f"[ZigZag ERROR] {_zze}\n{traceback.format_exc()}")

    # ── MARKET DECISION ENGINE ──────────────────────────────────────────────
    if show_mde:
        vah, val, cur_poc, mkt_mode, bal_high, bal_low = get_levels(df, tf)
        (in_bal, t_dir, key_high, key_low, _v, _vl, _p,
         _bko_ch, _str_ch, _khs_ch, _kls_ch) = get_struct_levels(df, tf=tf, sw=3)

        # ── Последние 5 балансов (зоны + POC по методологии Беггса/Далтона)
        balances = find_all_balances(df, tf=tf, max_balances=5)

        for i, b in enumerate(balances):
            is_cur = b.get('is_current', b.get('is_active', False))
            # Прямоугольник баланса убран — только VAH/POC/VAL линии

            # POC баланса
            poc_color = 'yellow' if is_cur else 'rgba(255,220,0,0.5)'
            poc_width  = 2 if is_cur else 1
            fig.add_shape(type="line",
                x0=b['x0'], x1=b['x1'],
                y0=b['poc'], y1=b['poc'],
                line=dict(color=poc_color, width=poc_width, dash='solid'),
                row=1, col=1)

            # VAH/VAL баланса
            fig.add_shape(type="line",
                x0=b['x0'], x1=b['x1'],
                y0=b['vah'], y1=b['vah'],
                line=dict(color='rgba(100,140,255,0.6)', width=1, dash='dot'),
                row=1, col=1)
            fig.add_shape(type="line",
                x0=b['x0'], x1=b['x1'],
                y0=b['val'], y1=b['val'],
                line=dict(color='rgba(100,140,255,0.6)', width=1, dash='dot'),
                row=1, col=1)

            # Аннотации: POC/VAH/VAL + CVD тренд баланса
            cvd_t   = b.get('cvd_trend', 'neutral')
            cvd_sym = " 📦" if cvd_t=="accumulation" else (" 📤" if cvd_t=="distribution" else "")
            bars_n  = b.get('bars', 0)

            fig.add_annotation(
                x=b['x1'], y=b['poc'],
                text=f"POC{cvd_sym}", showarrow=False,
                font=dict(color='yellow', size=10),
                xanchor='left', row=1, col=1)
            fig.add_annotation(
                x=b['x1'], y=b['vah'],
                text="VAH", showarrow=False,
                font=dict(color='rgba(100,140,255,0.9)', size=9),
                xanchor='left', row=1, col=1)
            fig.add_annotation(
                x=b['x1'], y=b['val'],
                text="VAL", showarrow=False,
                font=dict(color='rgba(100,140,255,0.9)', size=9),
                xanchor='left', row=1, col=1)

        # ── Если балансов нет — рисуем уровни из get_levels
        if not balances:
            if not np.isnan(vah):
                fig.add_hline(y=vah, line=dict(color='rgba(100,140,255,0.7)',width=1,dash='dot'),
                              annotation_text="VAH", row=1, col=1)
            if not np.isnan(val):
                fig.add_hline(y=val, line=dict(color='rgba(100,140,255,0.7)',width=1,dash='dot'),
                              annotation_text="VAL", row=1, col=1)
            if not np.isnan(cur_poc):
                fig.add_hline(y=cur_poc, line=dict(color='yellow',width=2),
                              annotation_text="POC", row=1, col=1)

        # ── Динамический POC (dPOC) — цветная ступенчатая линия миграции
        try:
            dpoc_arr = calc_dynamic_poc(df, window=50, tf=tf)
            dpoc_valid = [(df['timestamp'].iloc[i], float(dpoc_arr[i]))
                          for i in range(len(df)) if not np.isnan(dpoc_arr[i])]
            if dpoc_valid:
                dpoc_x = [v[0] for v in dpoc_valid]
                dpoc_y = [v[1] for v in dpoc_valid]

                # Цветная линия по направлению миграции (зел/красн/жёлт)
                for _seg_tr in _dpoc_colored_traces(dpoc_x, dpoc_y):
                    fig.add_trace(_seg_tr, row=1, col=1)

                # Аннотация с направлением миграции
                last_dpoc = dpoc_y[-1]
                _cur_close = float(df['close'].iloc[-1])
                _above = _cur_close >= last_dpoc
                # Направление: сравниваем последние 3 значения
                if len(dpoc_y) >= 3 and dpoc_y[-1] > dpoc_y[-3] + 1e-10:
                    _dir_sym, _dir_col = '↑', '#00E676'
                elif len(dpoc_y) >= 3 and dpoc_y[-1] < dpoc_y[-3] - 1e-10:
                    _dir_sym, _dir_col = '↓', '#FF1744'
                else:
                    _dir_sym, _dir_col = '→', '#FFD600'
                _pos_txt = 'выше' if _above else 'ниже'
                fig.add_annotation(
                    x=dpoc_x[-1], y=last_dpoc,
                    text=f"dPOC {_dir_sym} {last_dpoc:.4f}  ({_pos_txt})",
                    showarrow=False,
                    font=dict(color=_dir_col, size=9, family='monospace'),
                    bgcolor='rgba(0,0,0,0.5)',
                    borderpad=2,
                    xanchor='left', row=1, col=1)

                # HVN зоны — уровни где сосредоточен основной объём сессии
                _hvn = find_hvn_zones(df, tf=tf, n_zones=3)
                for _hz in _hvn:
                    _hc  = 'rgba(255,215,0,0.15)'   # золотистая заливка
                    _hbc = 'rgba(255,215,0,0.50)'
                    fig.add_shape(
                        type='rect',
                        x0=df['timestamp'].iloc[max(0, len(df)//4)],
                        x1=df['timestamp'].iloc[-1],
                        y0=_hz['price'] - _hz['width'],
                        y1=_hz['price'] + _hz['width'],
                        fillcolor=_hc,
                        line=dict(color=_hbc, width=0.5, dash='dot'),
                        row=1, col=1)
                    fig.add_annotation(
                        x=df['timestamp'].iloc[-1],
                        y=_hz['price'],
                        text=f"HVN {_hz['vol_pct']*100:.0f}%",
                        showarrow=False,
                        font=dict(color='rgba(255,215,0,0.8)', size=7),
                        xanchor='left', row=1, col=1)

        except Exception as _de:
            import traceback
            print(f"[dPOC] {_de}\n{traceback.format_exc()}")

        # ── BOS + CHoCH + OB + FVG
        bos_list, ob_list, fvg_list = find_bos_ob_fvg(df)

        for b in bos_list[-15:]:
            is_choch = b.get('choch', False)
            label    = b.get('label', 'BOS')

            if b['dir'] == 1:
                c    = 'rgba(255,200,0,0.95)' if is_choch else 'rgba(0,220,80,0.9)'
                dash = 'dot' if is_choch else 'solid'
            else:
                c    = 'rgba(255,200,0,0.95)' if is_choch else 'rgba(220,50,50,0.9)'
                dash = 'dot' if is_choch else 'solid'

            fig.add_shape(type="line", x0=b['x0'], x1=b['x1'],
                          y0=b['y'], y1=b['y'],
                          line=dict(color=c, width=2, dash=dash),
                          row=1, col=1)
            mid = b['x0'] + (b['x1'] - b['x0']) / 2
            fig.add_annotation(x=mid, y=b['y'], text=label,
                                showarrow=False, font=dict(color=c, size=9),
                                yanchor='bottom' if b['dir']==1 else 'top',
                                row=1, col=1)

        # ── ORDER BLOCKS — только активные (цена не пробила зону)
        x1_now = df['timestamp'].iloc[-1]
        _lc_now = float(df['close'].iloc[-1])
        for ob in ob_list[-6:]:
            is_bull_ob = ob['dir'] == 1
            # Пропускаем пробитые OB:
            # Demand (bull): тело закрылось ниже y0 = зона пробита снизу
            # Supply (bear): тело закрылось выше y1 = зона пробита сверху
            if is_bull_ob and _lc_now < ob['y0']:
                continue
            if not is_bull_ob and _lc_now > ob['y1']:
                continue
            ob_fill   = 'rgba(0,180,255,0.14)'  if is_bull_ob else 'rgba(255,120,0,0.14)'
            ob_border = 'rgba(0,180,255,0.70)'  if is_bull_ob else 'rgba(255,120,0,0.70)'
            fig.add_shape(type="rect",
                          x0=ob['x0'], x1=x1_now,
                          y0=ob['y0'],  y1=ob['y1'],
                          fillcolor=ob_fill,
                          line=dict(color=ob_border, width=1, dash='dot'),
                          row=1, col=1)
            # Подпись — противоположный угол (правый):
            # Demand OB: правый нижний угол  |  Supply OB: правый верхний угол
            fig.add_annotation(
                x=x1_now,
                y=ob['y0'] if is_bull_ob else ob['y1'],
                text="OB",
                showarrow=False,
                font=dict(color=ob_border, size=8),
                xanchor='right',
                yanchor='top' if is_bull_ob else 'bottom',
                row=1, col=1)

        # ── FVG — только незаполненные (цена не вернулась через зазор)
        fvg_limit = 8 if tf == "15m" else 12
        for f in fvg_list[-fvg_limit:]:
            # Пропускаем заполненные FVG:
            # Bull FVG (gap вверх): заполнен когда close <= y0 (нижняя граница зазора)
            # Bear FVG (gap вниз): заполнен когда close >= y1 (верхняя граница зазора)
            if f['dir'] == 1 and _lc_now <= f['y0']:
                continue
            if f['dir'] == -1 and _lc_now >= f['y1']:
                continue
            fvg_c = 'rgba(50,100,255,0.20)' if f['dir']==1 else 'rgba(255,140,0,0.20)'
            fig.add_shape(type="rect",
                          x0=f['x0'], x1=f['x1'], y0=f['y0'], y1=f['y1'],
                          fillcolor=fvg_c, line_width=0,
                          row=1, col=1)
            # Метка FVG справа
            fig.add_annotation(
                x=f['x1'], y=(f['y0'] + f['y1']) / 2,
                text="FVG", showarrow=False,
                font=dict(color=fvg_c.replace('0.20', '0.7'), size=7),
                xanchor='left', yanchor='middle',
                row=1, col=1)

        # ── KEY HIGH / KEY LOW — отрезки с историей (машина состояний SMC)
        try:
            sl_list = df.attrs.get('_struct_lines', None)
            if sl_list is None:
                _, _, _, sl_list = get_true_smc_structure(df)

            for line in sl_list:
                is_kh  = line['type'] == 'KEY_HIGH'
                active = line['active']
                color  = ('rgba(220,50,50,1.0)'  if active else 'rgba(220,50,50,0.30)') if is_kh \
                    else ('rgba(0,210,80,1.0)'   if active else 'rgba(0,210,80,0.30)')
                width  = 2 if active else 1
                dash   = 'solid' if active else 'dot'
                label  = (f"KEY HIGH {line['price']:.2f}" if is_kh
                          else f"KEY LOW {line['price']:.2f}") if active else None

                fig.add_shape(type='line',
                              x0=line['start_time'], x1=line['end_time'],
                              y0=line['price'],      y1=line['price'],
                              line=dict(color=color, width=width, dash=dash),
                              row=1, col=1)
                if label:
                    fig.add_annotation(x=line['end_time'], y=line['price'],
                                       text=label, showarrow=False,
                                       font=dict(color=color, size=9),
                                       xanchor='left', yanchor='middle',
                                       row=1, col=1)
        except Exception as _ke:
            if not np.isnan(key_high):
                fig.add_hline(y=key_high,
                              line=dict(color='rgba(220,50,50,0.8)', width=1, dash='dot'),
                              annotation_text="KEY HIGH", row=1, col=1)
            if not np.isnan(key_low):
                fig.add_hline(y=key_low,
                              line=dict(color='rgba(0,210,80,0.8)', width=1, dash='dot'),
                              annotation_text="KEY LOW", row=1, col=1)

        # ── Liq Grab маркеры — тень пробила Key High/Low, тело вернулось
        # Паттерн: high > key_high AND close < key_high AND open < key_high (или наоборот для low)
        if not np.isnan(key_high) and not np.isnan(key_low):
            for _lgi in range(max(0, len(df) - 150), len(df)):
                _row = df.iloc[_lgi]
                _body_top = max(float(_row['open']), float(_row['close']))
                _body_bot = min(float(_row['open']), float(_row['close']))
                # Liq Grab выше Key High: тень выше KH, тело ниже KH
                if float(_row['high']) > key_high and _body_top < key_high:
                    fig.add_annotation(
                        x=_row['timestamp'], y=float(_row['high']),
                        text='LG', showarrow=True,
                        arrowhead=2, arrowcolor='rgba(255,80,80,0.9)',
                        arrowsize=0.8, ax=0, ay=-20,
                        font=dict(color='rgba(255,80,80,0.9)', size=9),
                        row=1, col=1)
                # Liq Grab ниже Key Low: тень ниже KL, тело выше KL
                if float(_row['low']) < key_low and _body_bot > key_low:
                    fig.add_annotation(
                        x=_row['timestamp'], y=float(_row['low']),
                        text='LG', showarrow=True,
                        arrowhead=2, arrowcolor='rgba(0,220,80,0.9)',
                        arrowsize=0.8, ax=0, ay=20,
                        font=dict(color='rgba(0,220,80,0.9)', size=9),
                        row=1, col=1)

        # ── Premium / Discount / OTE зоны
        # Чистый рендеринг: только линии (без заливки прямоугольников)
        # Как рекомендовано: add_hline с opacity вместо rect
        if not np.isnan(key_high) and not np.isnan(key_low) and key_high > key_low:
            rng   = key_high - key_low
            eq    = key_low + rng * 0.5
            ote_h = key_low + rng * (1 - 0.62)   # Fib 0.62 = 38% от хая
            ote_l = key_low + rng * (1 - 0.79)   # Fib 0.79 = 21% от хая
            ote_o = key_low + rng * (1 - 0.705)  # OTE оптимум 0.705

            # EQ — серая тонкая пунктирная (чистая, не заливает)
            fig.add_hline(y=eq,
                          line=dict(color='rgba(200,200,200,0.3)', width=1, dash='dash'),
                          annotation_text="EQ 0.5",
                          annotation_font=dict(color='rgba(200,200,200,0.5)', size=8),
                          row=1, col=1)

            # OTE оптимальная линия 0.705 — золотая жирная
            fig.add_hline(y=ote_o,
                          line=dict(color='gold', width=2, dash='solid'),
                          annotation_text="OTE 0.705 ★",
                          annotation_font=dict(color='gold', size=9),
                          row=1, col=1)

            # OTE диапазон 0.62–0.79 — тонкие пунктиры
            fig.add_hline(y=ote_h,
                          line=dict(color='rgba(255,200,0,0.4)', width=1, dash='dot'),
                          annotation_text="0.62",
                          annotation_font=dict(color='rgba(255,200,0,0.6)', size=8),
                          row=1, col=1)
            fig.add_hline(y=ote_l,
                          line=dict(color='rgba(255,200,0,0.4)', width=1, dash='dot'),
                          annotation_text="0.79",
                          annotation_font=dict(color='rgba(255,200,0,0.6)', size=8),
                          row=1, col=1)

        # ── Swing метки HH / HL / LH / LL (последние 15 каждого)
        # yshift отодвигает метку от свечи чтобы не налипала
        SH, SL = find_swings(df)
        for sh in SH[-15:]:
            lbl = sh.get('label', 'H')
            # HH и LH — разные цвета и размеры
            if lbl == 'HH':
                col, sz = 'rgba(0,220,80,0.95)', 10   # HH зелёный — бычья сила
            elif lbl == 'LH':
                col, sz = 'rgba(220,50,50,0.85)', 9   # LH красный — медвежья слабость
            else:
                col, sz = 'rgba(150,150,150,0.7)', 8
            fig.add_annotation(x=sh['ts'], y=sh['price'],
                                text=lbl, showarrow=False,
                                font=dict(color=col, size=sz),
                                yanchor='bottom', yshift=8,
                                row=1, col=1)

        for sl in SL[-15:]:
            lbl = sl.get('label', 'L')
            if lbl == 'LL':
                col, sz = 'rgba(220,50,50,0.95)', 10  # LL красный — медвежья сила
            elif lbl == 'HL':
                col, sz = 'rgba(0,220,80,0.85)', 9    # HL зелёный — бычья сила
            else:
                col, sz = 'rgba(150,150,150,0.7)', 8
            fig.add_annotation(x=sl['ts'], y=sl['price'],
                                text=lbl, showarrow=False,
                                font=dict(color=col, size=sz),
                                yanchor='top', yshift=-8,
                                row=1, col=1)

        # ── Weak High/Low маркировка (цели для цены — Радар может алертить)
        # Weak = свинг который НЕ привёл к BOS (алгоритм идёт за этой ликвидностью)
        bos_levels_up   = {b['y'] for b in bos_list if b['dir'] == 1}
        bos_levels_down = {b['y'] for b in bos_list if b['dir'] == -1}
        for sh in SH[-10:]:
            if sh['price'] not in bos_levels_up and sh['price'] not in bos_levels_down:
                # Weak High — тонкая пунктирная линия + значок цели
                fig.add_annotation(x=sh['ts'], y=sh['price'],
                                    text='🎯', showarrow=False,
                                    font=dict(size=10), yanchor='bottom',
                                    yshift=18, row=1, col=1)
        for sl in SL[-10:]:
            if sl['price'] not in bos_levels_up and sl['price'] not in bos_levels_down:
                fig.add_annotation(x=sl['ts'], y=sl['price'],
                                    text='🎯', showarrow=False,
                                    font=dict(size=10), yanchor='top',
                                    yshift=-18, row=1, col=1)

        # ── AMT панель в сайдбаре (Далтон / Беггс / Вайсс)
        amt_ctx = analyze_auction_context(df)
        state_txt = "BALANCE ⚖️" if in_bal else ("TREND ↑" if t_dir==1 else "TREND ↓")
        state_col = "#9E9E9E" if in_bal else ("#00C853" if t_dir==1 else "#FF5252")

        st.sidebar.markdown("---")
        st.sidebar.markdown(
            f"**📊 AMT Market State**\n\n"
            f"<span style='color:{state_col};font-size:15px;font-weight:bold'>{state_txt}</span>",
            unsafe_allow_html=True)
        if not np.isnan(cur_poc):
            st.sidebar.markdown(f"**POC:** `{cur_poc:.5f}`")

        # Фаза баланса (Коуллинг)
        if in_bal and amt_ctx.get('balance_phase'):
            bp = amt_ctx['balance_phase']
            bp_col = {"ACCUMULATION":"#00C853","DISTRIBUTION":"#FF5252",
                      "COMPRESSED":"#FFD600","NEUTRAL":"#9E9E9E"}.get(bp,"#9E9E9E")
            bp_emoji = {"ACCUMULATION":"📦 Накопление","DISTRIBUTION":"📤 Распределение",
                        "COMPRESSED":"🗜️ Сжатый баланс","NEUTRAL":"➖ Нейтральный"}.get(bp,bp)
            st.sidebar.markdown(
                f"<span style='color:{bp_col}'>{bp_emoji}</span> "
                f"<small>({amt_ctx.get('bal_duration',0)} баров)</small>",
                unsafe_allow_html=True)

        # SOT — тренд слабеет (Беггс/Вайсс)
        if not in_bal:
            if amt_ctx.get('sot'):
                st.sidebar.markdown("⚠️ <span style='color:#FFD600'>SOT: тренд слабеет</span>",
                                    unsafe_allow_html=True)
            if amt_ctx.get('deep_pullback'):
                st.sidebar.markdown("⚠️ <span style='color:#FF9800'>Глубокие откаты</span>",
                                    unsafe_allow_html=True)

        # Хвосты аукциона (Далтон)
        if amt_ctx.get('bull_tail'):
            st.sidebar.markdown("🐂 <span style='color:#00C853'>Бычий хвост аукциона</span>",
                                unsafe_allow_html=True)
        if amt_ctx.get('bear_tail'):
            st.sidebar.markdown("🐻 <span style='color:#FF5252'>Медвежий хвост аукциона</span>",
                                unsafe_allow_html=True)

        # Подсказка стратегии
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"💡 *{amt_ctx.get('strategy_hint','')}*")

    # — Absorption маркеры убраны с графика (сигналы используются только в скринере)

    # — CVD
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['cvd'],
        line=dict(color='#00E676',width=2), name="CVD"), 2, 1)

    # — Дивергенции CVD (Exhaustion + Absorption)
    if show_divs:
        for d in find_divergences(df, lookback=5, min_dist=8):
            is_bull = d['type'] == 'bull'
            subtype = d.get('subtype', 'exhaustion')
            # Цвета: поглощение ярче, истощение мягче
            if subtype == 'absorption':
                c_price = '#00E5FF' if is_bull else '#FF6D00'
                label   = '🔵 Abs' if is_bull else '🔴 Abs'
            elif subtype == 'hidden':
                c_price = '#B2FF59' if is_bull else '#FF80AB'
                label   = 'H▲' if is_bull else 'H▼'
            else:
                c_price = '#00E676' if is_bull else '#FF5252'
                label   = '▲' if is_bull else '▼'

            # Линия по цене (на панели 1)
            fig.add_shape(type='line',
                x0=d['x0'], x1=d['x1'],
                y0=d['yp0'], y1=d['yp1'],
                line=dict(color=c_price, width=2, dash='dot'),
                row=1, col=1)
            # Линия по CVD (на панели 2)
            fig.add_shape(type='line',
                x0=d['x0'], x1=d['x1'],
                y0=d['yc0'], y1=d['yc1'],
                line=dict(color=c_price, width=2, dash='dot'),
                row=2, col=1)
            # Аннотация на ценовой панели
            fig.add_annotation(
                x=d['x1'], y=d['yp1'],
                text=label, showarrow=False,
                font=dict(color=c_price, size=10),
                yanchor='bottom' if is_bull else 'top',
                xanchor='left',
                row=1, col=1)


    # — Объём (бары + MA линия + заливка hi/lo band)
    fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'],
        marker_color=df['v_col'], name="Volume"), 3, 1)
    # Синяя линия SMA30
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['vol_mean'],
        line=dict(color='#2979FF', width=1), name="Vol MA"), 3, 1)
    # Голубая заливка между hi_band и lo_band
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['vol_hi'],
        line=dict(color='rgba(154,209,255,0.6)', width=1),
        name="Hi Band", showlegend=False), 3, 1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['vol_lo'],
        line=dict(color='rgba(154,209,255,0.6)', width=1),
        fill='tonexty', fillcolor='rgba(154,209,255,0.15)',
        name="Lo Band", showlegend=False), 3, 1)

    # — Delta Candles (реальные 1м данные через Bybit API)
    if src == "crypto":
        df_1m = fetch_1m_data(asset, src)
        dc_df = build_delta_candles(df, df_1m, calc_gimelfarb_delta)
        if not dc_df.empty:
            dc_colors = ['teal' if v >= 0 else '#FF1744' for v in dc_df['c']]
            fig.add_trace(go.Candlestick(
                x=dc_df['timestamp'],
                open=dc_df['o'], high=dc_df['h'],
                low=dc_df['l'],  close=dc_df['c'],
                increasing_line_color='teal', increasing_fillcolor='teal',
                decreasing_line_color='#FF1744', decreasing_fillcolor='#FF1744',
                name="Delta", whiskerwidth=0.3
            ), 4, 1)
        else:
            dc = ['#00E676' if v>=0 else '#FF1744' for v in df['delta']]
            fig.add_trace(go.Bar(x=df['timestamp'], y=df['delta'],
                marker_color=dc, name="Delta"), 4, 1)
    else:
        dc = ['#00E676' if v>=0 else '#FF1744' for v in df['delta']]
        fig.add_trace(go.Bar(x=df['timestamp'], y=df['delta'],
            marker_color=dc, name="Delta"), 4, 1)
    fig.add_hline(y=0, line=dict(color='white',width=1,dash='dot'), row=4, col=1)

# ══════════════════════════════════════════════════════════════
#  RENKO РЕЖИМ (только вкладки 1-17)
# ══════════════════════════════════════════════════════════════
elif tab not in _LAB_TABS:
    rb = build_renko(df, asset)
    if rb.empty:
        st.warning("Недостаточно данных для Renko")
        st.stop()

    # Формируем заголовок с размером кирпича
    _brick_sz = float(rb['brick_size'].iloc[0])
    if _brick_sz < 1:
        _brick_label = f"Renko  |  Кирпич: {_brick_sz:.4f}"
    elif _brick_sz < 100:
        _brick_label = f"Renko  |  Кирпич: {_brick_sz:.2f}"
    else:
        _brick_label = f"Renko  |  Кирпич: {_brick_sz:.1f}"

    # Layout 4 субплота: Цена / CVD / Delta гистограмма / Volume
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.55, 0.17, 0.14, 0.14],
        subplot_titles=(_brick_label, "CVD", "Delta", "Volume")
    )
    idx = list(range(len(rb)))

    # — FTR зоны на Renko (специальная логика по кирпичам)
    renko_zones = calc_ftr_zones_renko(rb, asset, src)
    last_idx = len(rb) - 1
    # Renko — только активные зоны FTR (последние 6)
    active_renko = [z for z in renko_zones if z.get('active', False)]
    show_renko_zones = active_renko[-6:] if len(active_renko) > 6 else active_renko
    for z in show_renko_zones:
        tc = z.get('touch_count', 0)
        if z['dir'] == 1:
            fill   = 'rgba(0,200,80,0.35)' if tc >= 1 else 'rgba(0,200,80,0.20)'
            border = 'rgba(0,220,80,0.7)'
        else:
            fill   = 'rgba(220,50,50,0.35)' if tc >= 1 else 'rgba(220,50,50,0.20)'
            border = 'rgba(220,50,50,0.7)'
        label = "🔥 FTR" if tc >= 1 else ("Demand" if z['dir']==1 else "Supply")
        fig.add_shape(type="rect",
            x0=z['i'], x1=last_idx,
            y0=z['zl'], y1=z['zh'],
            fillcolor=fill,
            line=dict(color=border, width=1),
            row=1, col=1)
        fig.add_annotation(
            x=z['i'], y=(z['zl'] + z['zh']) / 2,
            text=label, showarrow=False,
            font=dict(color=border, size=8),
            xanchor='right', row=1, col=1)
        if z.get('impulse_len', 0) >= 3:
            fig.add_annotation(
                x=last_idx, y=z['zh'],
                text="⚡", showarrow=False,
                font=dict(size=10),
                xanchor='left', row=1, col=1)

    # Время для каждого кирпича (для hover)
    times_str = [str(t)[:16] for t in rb['time'].values]

    # — Renko кирпичи (с временем в hover)
    fig.add_trace(go.Candlestick(
        x=idx,
        open=rb['open'], high=rb['high'],
        low=rb['low'],   close=rb['close'],
        increasing_line_color='#00E676', decreasing_line_color='#FF1744',
        text=times_str,
        hovertext=times_str,
        name="Renko"), 1, 1)

    # — Дельта-дивергенции на ценовом графике
    # Бычья: зелёный кирпич + отрицательная delta → лимитник поглощает продажи
    # Медвежья: красный кирпич + положительная delta → лимитник поглощает покупки
    if 'delta' in rb.columns:
        bull_div_x, bull_div_y = [], []
        bear_div_x, bear_div_y = [], []
        for _i in range(len(rb)):
            _bull  = bool(rb['bull'].iloc[_i])
            _delta = float(rb['delta'].iloc[_i])
            if _bull and _delta < 0:
                bull_div_x.append(_i)
                bull_div_y.append(float(rb['close'].iloc[_i]))
            elif not _bull and _delta > 0:
                bear_div_x.append(_i)
                bear_div_y.append(float(rb['close'].iloc[_i]))

        if bull_div_x:
            fig.add_trace(go.Scatter(
                x=bull_div_x, y=bull_div_y,
                mode='markers+text',
                marker=dict(symbol='triangle-up', size=12,
                            color='#00E5FF', line=dict(color='#00B0FF', width=1)),
                text=['⚡'] * len(bull_div_x),
                textposition='bottom center',
                textfont=dict(size=10, color='#00E5FF'),
                hovertemplate='Погл.Бычье<br>%{x}<extra></extra>',
                name='Погл.Бычье'), 1, 1)

        if bear_div_x:
            fig.add_trace(go.Scatter(
                x=bear_div_x, y=bear_div_y,
                mode='markers+text',
                marker=dict(symbol='triangle-down', size=12,
                            color='#E040FB', line=dict(color='#AA00FF', width=1)),
                text=['⚡'] * len(bear_div_x),
                textposition='top center',
                textfont=dict(size=10, color='#E040FB'),
                hovertemplate='Погл.Медв.<br>%{x}<extra></extra>',
                name='Погл.Медв.'), 1, 1)

    # — CVD (с временем в hover)
    fig.add_trace(go.Scatter(
        x=idx, y=rb['cvd'],
        customdata=times_str,
        hovertemplate="<b>%{customdata}</b><br>CVD: %{y:,.0f}<extra></extra>",
        line=dict(color='#00E676', width=2), name="CVD"), 2, 1)

    # — Структурные дивергенции CVD vs Цена (ZigZag-подход, подтверждение 4 кирпича)
    # Бычья:   цена LL + CVD HL → зелёная пунктирная линия на цене и CVD
    # Медвежья: цена HH + CVD LH → красная пунктирная линия на цене и CVD
    bull_divs, bear_divs = find_structural_divergences(rb, reversal_bricks=4)

    for s_idx, e_idx in bull_divs:
        # Линия на ценовом субплоте (row=1)
        fig.add_trace(go.Scatter(
            x=[s_idx, e_idx],
            y=[float(rb['close'].iloc[s_idx]), float(rb['close'].iloc[e_idx])],
            mode='lines',
            line=dict(color='rgba(0,230,118,0.9)', width=2, dash='dot'),
            hoverinfo='skip', showlegend=False), 1, 1)
        # Линия на CVD субплоте (row=2)
        fig.add_trace(go.Scatter(
            x=[s_idx, e_idx],
            y=[float(rb['cvd'].iloc[s_idx]), float(rb['cvd'].iloc[e_idx])],
            mode='lines',
            line=dict(color='rgba(0,230,118,0.9)', width=2, dash='dot'),
            hoverinfo='skip', showlegend=False), 2, 1)
        # Аннотация на ценовом субплоте
        fig.add_annotation(
            x=e_idx, y=float(rb['close'].iloc[e_idx]),
            text='🟢 Спринг', showarrow=False,
            font=dict(color='#00E676', size=11),
            yanchor='top', xanchor='left', row=1, col=1)

    for s_idx, e_idx in bear_divs:
        # Линия на ценовом субплоте (row=1)
        fig.add_trace(go.Scatter(
            x=[s_idx, e_idx],
            y=[float(rb['close'].iloc[s_idx]), float(rb['close'].iloc[e_idx])],
            mode='lines',
            line=dict(color='rgba(255,82,82,0.9)', width=2, dash='dot'),
            hoverinfo='skip', showlegend=False), 1, 1)
        # Линия на CVD субплоте (row=2)
        fig.add_trace(go.Scatter(
            x=[s_idx, e_idx],
            y=[float(rb['cvd'].iloc[s_idx]), float(rb['cvd'].iloc[e_idx])],
            mode='lines',
            line=dict(color='rgba(255,82,82,0.9)', width=2, dash='dot'),
            hoverinfo='skip', showlegend=False), 2, 1)
        # Аннотация на ценовом субплоте
        fig.add_annotation(
            x=e_idx, y=float(rb['close'].iloc[e_idx]),
            text='🔴 Аптраст', showarrow=False,
            font=dict(color='#FF5252', size=11),
            yanchor='bottom', xanchor='left', row=1, col=1)

    # — Delta гистограмма (под CVD)
    # Зелёный = покупатели доминировали в кирпиче, красный = продавцы
    if 'delta' in rb.columns:
        delta_colors = ['#00E676' if d >= 0 else '#FF1744'
                        for d in rb['delta'].values]
        fig.add_trace(go.Bar(
            x=idx, y=rb['delta'],
            marker_color=delta_colors,
            customdata=times_str,
            hovertemplate="<b>%{customdata}</b><br>Δ: %{y:,.0f}<extra></extra>",
            name="Delta"), 3, 1)

    # — Нулевая линия на Delta субплоте
    fig.add_hline(y=0, line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dot'),
                  row=3, col=1)

    # — Volume (row=4)
    fig.add_trace(go.Bar(
        x=idx, y=rb['vol'],
        marker_color=rb['v_col'],
        customdata=times_str,
        hovertemplate="<b>%{customdata}</b><br>Vol: %{y:,.0f}<extra></extra>",
        name="Vol"), 4, 1)

    # Подписи оси X — реальное время каждые N кирпичей
    step = max(1, len(rb) // 20)
    for _row in [1, 2, 3, 4]:
        fig.update_xaxes(
            tickvals=idx[::step],
            ticktext=[str(rb['time'].iloc[i])[:16] for i in idx[::step]],
            row=_row, col=1)

# ─────────────────────────────────────────────────────────────────────────────
# 10. LAYOUT (только вкладки 1-17)
# ─────────────────────────────────────────────────────────────────────────────
if tab not in _LAB_TABS:
    fig.update_layout(
        height=950, template="plotly_dark",
        margin=dict(l=0,r=50,t=25,b=0),
        xaxis_rangeslider_visible=False,
        dragmode='pan', hovermode='x unified',
        showlegend=False,
        hoverlabel=dict(
            bgcolor='rgba(30,30,40,0.95)',
            font_size=12,
            font_family="monospace",
        ),
    )
    fig.update_xaxes(
        showspikes=True, spikecolor="white",
        spikemode="across", spikedash="dash", spikethickness=1,
        hoverformat="%d %b %Y  %H:%M",
    )
    fig.update_yaxes(side="right", fixedrange=False)

# ── Ручной Volume Profile (сайдбар) — только для вкладок 1-17
if tab not in _LAB_TABS:
    st.sidebar.markdown("---")
    with st.sidebar.expander("📐 Volume Profile вручную", expanded=False):
        vp_rows = st.slider("Рядов профиля", 10, 48, 24, key="vp_rows")

        if tf == "1D":
            dt_fmt  = "YYYY-MM-DD"
            dt_help = "Формат: 2024-01-15"
        else:
            dt_fmt  = "YYYY-MM-DD HH:MM"
            dt_help = "Формат: 2024-01-15 14:30"

        if not is_renko and not df.empty:
            ts_min_str = df["timestamp"].min().strftime("%Y-%m-%d %H:%M")
            ts_max_str = df["timestamp"].max().strftime("%Y-%m-%d %H:%M")
            st.caption(f"Данные: {ts_min_str} → {ts_max_str}")

        vp_t0_str = st.text_input("От (дата/время начала)", value="", placeholder=dt_help, key="vp_t0")
        vp_t1_str = st.text_input("До (дата/время конца)",  value="", placeholder=dt_help, key="vp_t1")
        show_vp   = st.button("▶ Построить профиль", key="vp_btn")

        vp_t0 = None
        vp_t1 = None
        if vp_t0_str:
            try:    vp_t0 = pd.Timestamp(vp_t0_str)
            except: st.error("Неверный формат даты начала")
        if vp_t1_str:
            try:    vp_t1 = pd.Timestamp(vp_t1_str)
            except: st.error("Неверный формат даты конца")
else:
    # Заглушки для переменных VP которые используются ниже
    show_vp = False
    vp_t0 = vp_t1 = None

# ── Рассчитываем и рисуем VP на графике если нажата кнопка
def draw_vp_on_fig(fig_obj, vp_df, rows_n=24):
    """Рисует горизонтальный Volume Profile внутри диапазона vp_df."""
    if len(vp_df) < 2:
        return fig_obj, None

    p_hi = vp_df["high"].max()
    p_lo = vp_df["low"].min()
    if p_hi <= p_lo:
        return fig_obj, None

    row_h = (p_hi - p_lo) / rows_n
    H = vp_df["high"].values
    L = vp_df["low"].values
    V = vp_df["volume"].values

    lvl_lo = p_lo + np.arange(rows_n) * row_h
    lvl_hi = lvl_lo + row_h
    Hm = H[:,None]; Lm = L[:,None]; Vm = V[:,None]
    br = (H - L)[:,None]
    ov = (np.minimum(Hm, lvl_hi) - np.maximum(Lm, lvl_lo)).clip(min=0)
    prof = np.where(br > 0, Vm * ov / br, 0).sum(axis=0)
    mx = prof.max()
    if mx == 0:
        return fig_obj, None
    prof_norm = prof / mx

    # POC и Value Area 70%
    poc_j = int(np.argmax(prof))
    poc   = p_lo + poc_j * row_h + row_h / 2
    total = prof.sum()
    va_vol = total * 0.70
    sorted_j = np.argsort(prof)[::-1]
    cum = 0; va_set = set()
    for jj in sorted_j:
        cum += prof[jj]; va_set.add(jj)
        if cum >= va_vol: break
    vah = p_lo + (max(va_set) + 1) * row_h
    val = p_lo + min(va_set) * row_h

    # Ширина профиля = 30% длины диапазона по X
    x0_ts = vp_df["timestamp"].iloc[0]
    x1_ts = vp_df["timestamp"].iloc[-1]
    try:
        span_s = (pd.Timestamp(x1_ts) - pd.Timestamp(x0_ts)).total_seconds()
        max_w  = span_s * 0.30
    except:
        max_w  = 3600 * 4

    # Рисуем профиль: один scatter-силуэт + POC линия
    xs, ys, cs = [], [], []
    for j in range(rows_n):
        w   = pd.Timedelta(seconds=float(prof_norm[j]) * max_w)
        y0b = p_lo + j * row_h
        y1b = y0b + row_h
        x1b = pd.Timestamp(x0_ts) + w
        is_va  = j in va_set
        is_poc = j == poc_j
        col = ("rgba(255,220,0,0.7)"   if is_poc else
               "rgba(0,200,100,0.45)"  if is_va  else
               "rgba(100,160,255,0.30)")
        # Добавляем прямоугольник как 4 точки замкнутого контура
        xs  += [x0_ts, x1b,   x1b,   x0_ts, x0_ts, None]
        ys  += [y0b,   y0b,   y1b,   y1b,   y0b,   None]

    fig_obj.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="lines",
        fill="toself",
        fillcolor="rgba(100,160,255,0.18)",
        line=dict(color="rgba(100,160,255,0.5)", width=1),
        showlegend=False, hoverinfo="skip",
    ), 1, 1)

    # POC линия — жёлтая горизонталь по всей ширине диапазона
    fig_obj.add_shape(type="line",
        x0=x0_ts, x1=x1_ts, y0=poc, y1=poc,
        line=dict(color="yellow", width=2), row=1, col=1)

    # VAH / VAL линии
    fig_obj.add_shape(type="line",
        x0=x0_ts, x1=x1_ts, y0=vah, y1=vah,
        line=dict(color="rgba(0,200,100,0.8)", width=1, dash="dot"), row=1, col=1)
    fig_obj.add_shape(type="line",
        x0=x0_ts, x1=x1_ts, y0=val, y1=val,
        line=dict(color="rgba(0,200,100,0.8)", width=1, dash="dot"), row=1, col=1)

    stats = {"poc": poc, "vah": vah, "val": val,
             "total_vol": total, "bars": len(vp_df)}
    return fig_obj, stats

if tab not in _LAB_TABS:
    if show_vp and not is_renko:
        if vp_t0 and vp_t1:
            if vp_t0 > vp_t1:
                vp_t0, vp_t1 = vp_t1, vp_t0
            mask = (df["timestamp"] >= vp_t0) & (df["timestamp"] <= vp_t1)
            vp_slice = df[mask]
            if len(vp_slice) < 2:
                st.sidebar.warning("В указанном диапазоне нет данных")
            else:
                fig, vp_stats = draw_vp_on_fig(fig, vp_slice, rows_n=vp_rows)
                if vp_stats:
                    st.session_state["vp_stats"] = vp_stats
        else:
            st.sidebar.warning("Введите дату начала и конца")

    # Показываем результаты VP под графиком
    if "vp_stats" in st.session_state and show_vp:
        s = st.session_state["vp_stats"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("POC",    f"{s['poc']:.5f}")
        c2.metric("VAH",    f"{s['vah']:.5f}")
        c3.metric("VAL",    f"{s['val']:.5f}")
        c4.metric("Объём",  f"{s['total_vol']:,.0f}")
        c5.metric("Баров",  str(s['bars']))

    st.plotly_chart(fig, use_container_width=True, config={
        'scrollZoom': True,
        'modeBarButtonsToAdd': ['drawline','drawrect','eraseshape']
    })

    # ── MTF Confluence панель (только свечные вкладки, не renko) ─────────────
    if not is_renko and src == 'crypto':
        _render_mtf_confluence(asset, src, tf)


# ═══════════════════════════════════════════════════════════════════════════════
# ВКЛАДКА 15: Phase 0 Trades — визуализация сделок стратегии на свечном графике
# Изолирована от других вкладок. Используется для визуальной отладки ядра стратегии.
# ═══════════════════════════════════════════════════════════════════════════════

if tab == "15. 📊 Phase 0 Trades":
    from phase0_chart import (
        list_strategy_parquets, load_trades, load_ohlcv,
        make_chart, make_demo_chart,
    )

    st.header("📊 Phase 0 Trades — визуализация сделок стратегии")
    st.caption(
        "Инструмент визуальной отладки ядра стратегии. Выберите bt_*_raw.parquet, "
        "символ и ТФ — увидите где именно стратегия открывает сделки, какая структура "
        "рынка вокруг, как SL/TP относятся к свечам."
    )

    # ── Селекторы ────────────────────────────────────────────────────────────
    parquets = list_strategy_parquets()
    parquet_names = [p.name for p in parquets]

    col_a, col_b = st.columns([3, 1])
    with col_a:
        if not parquet_names:
            st.warning("Нет bt_*_raw.parquet в backtest_data/ — показываю Demo.")
            sel_parquet = None
        else:
            sel_parquet_name = st.selectbox(
                "Стратегия (bt_*_raw.parquet):", parquet_names,
                key="ph0_strategy_sel",
            )
            sel_parquet = next(p for p in parquets if p.name == sel_parquet_name)
    with col_b:
        demo_mode = st.checkbox("Demo (synthetic)", value=False, key="ph0_demo")

    if demo_mode or sel_parquet is None:
        # ── DEMO режим: synthetic OHLCV + 3 mock trades ─────────────────────
        st.info("🧪 Demo режим — synthetic random walk + 3 mock сделки. "
                "Снимите галочку 'Demo' чтобы работать с реальной стратегией.")
        fig = make_demo_chart()
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
    else:
        # ── РЕАЛЬНЫЕ ДАННЫЕ ─────────────────────────────────────────────────
        try:
            trades_all = load_trades(sel_parquet, split=None)
        except Exception as e:
            st.error(f"Ошибка чтения {sel_parquet.name}: {e}")
            st.stop()

        if trades_all.empty:
            st.warning("В parquet нет сделок.")
            st.stop()

        # Селекторы: split, символ, ТФ
        # Dynamic keys включают имя parquet — при смене стратегии state сбрасывается.
        _pkey = sel_parquet.stem
        col1, col2, col3, col4 = st.columns([1, 2, 1, 1])

        with col1:
            splits = ['All'] + sorted(trades_all['split'].dropna().unique().tolist()) \
                     if 'split' in trades_all.columns else ['All']
            # Default = IS если есть
            split_idx = splits.index('IS') if 'IS' in splits else 0
            sel_split = st.selectbox("Сплит:", splits, index=split_idx,
                                      key=f"ph0_split_{_pkey}")

        td = trades_all if sel_split == 'All' else trades_all[trades_all['split'] == sel_split]

        # Сортировка: приоритет популярным, затем алфавит
        def _sort_symbols(syms):
            priority = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT',
                        'EURUSD', 'GBPUSD', 'XAUUSD']
            head = [s for s in priority if s in syms]
            tail = sorted([s for s in syms if s not in priority])
            return head + tail

        # Естественный порядок ТФ (а не алфавитный 15m,1D,1h,3m)
        _TF_ORDER = ['1m', '3m', '5m', '15m', '30m', '1h', '4h', '1D']
        def _sort_tfs(tfs):
            known = [t for t in _TF_ORDER if t in tfs]
            unknown = [t for t in tfs if t not in _TF_ORDER]
            return known + sorted(unknown)

        with col2:
            symbols = _sort_symbols(td['symbol'].dropna().unique().tolist())
            # ДИАГНОСТИКА: пишем реальное число символов чтобы видеть что попало в dropdown
            st.caption(
                f"📊 Доступно символов: **{len(symbols)}** | "
                f"BTCUSDT в списке: **{'✅ да' if 'BTCUSDT' in symbols else '❌ нет'}** | "
                f"в parquet всего: **{trades_all['symbol'].nunique()}**"
            )
            if not symbols:
                st.warning("Нет сделок в выбранном сплите.")
                st.stop()
            sel_sym = st.selectbox(
                f"Символ ({len(symbols)} опций):",
                symbols, index=0,
                key=f"ph0_sym_{_pkey}_{sel_split}",
                help="Печатай для поиска (например 'BTC'). "
                     "Dropdown скроллится — мотай вниз если не видишь нужный символ.",
            )

        with col3:
            tfs = _sort_tfs(td[td['symbol'] == sel_sym]['tf'].dropna().unique().tolist())
            if not tfs:
                st.warning("Нет ТФ для этого символа.")
                st.stop()
            default_tf = '15m' if '15m' in tfs else tfs[0]
            sel_tf = st.selectbox("ТФ:", tfs,
                                   index=tfs.index(default_tf),
                                   key=f"ph0_tf_{_pkey}_{sel_split}_{sel_sym}")

        with col4:
            sel_dir = st.multiselect("Направление:", ['LONG', 'SHORT'],
                                      default=['LONG', 'SHORT'],
                                      key=f"ph0_dir_{_pkey}")

        # Фильтрованные сделки
        sub_trades = td[
            (td['symbol'] == sel_sym) &
            (td['tf'] == sel_tf) &
            (td['direction'].isin(sel_dir))
        ].copy()

        if sub_trades.empty:
            st.warning(f"Нет сделок для {sel_sym} {sel_tf} (split={sel_split}) с выбранными направлениями.")
            st.stop()

        # ── Окно времени для рендера ──────────────────────────────────────────
        sub_trades['timestamp'] = pd.to_datetime(sub_trades['timestamp'], utc=True)
        t_min = sub_trades['timestamp'].min()
        t_max = sub_trades['timestamp'].max()

        st.markdown(f"**Найдено сделок:** {len(sub_trades):,} | "
                    f"период: `{t_min:%Y-%m-%d}` → `{t_max:%Y-%m-%d}`")

        # Слайдер: окно из N свечей вокруг одной выбранной сделки ИЛИ всё
        col5, col6, col7, col8 = st.columns([2, 1, 1, 1])
        with col5:
            window_mode = st.radio("Окно:", ["По одной сделке", "Все сделки в диапазоне"],
                                    horizontal=True, key="ph0_window_mode")
        with col6:
            window_bars = st.slider("Свечей вокруг:", 50, 1500, 300, 50,
                                     key="ph0_window_bars")
        with col7:
            rr_value = st.slider("TP RR:", 1.0, 5.0, 3.0, 0.5, key="ph0_rr")
        with col8:
            show_fvg = st.checkbox("Показать FVG зоны", value=True,
                                    key="ph0_show_fvg",
                                    help="Полупрозрачные прямоугольники: зелёные = bullish FVG, "
                                         "красные = bearish. Длина = lookback стратегии.")
            # Параметры FVG из самого скрипта стратегии (синхронизация с ядром).
            # v10 переименовал параметры: FVG_MIN_SIZE_ATR (без подчёркивания),
            # FVG_BODY_MULTIPLIER (вместо _MIN_IMPULSE_BODY_RATIO).
            try:
                from bt_fvg_retest_Bar import (
                    FVG_LOOKBACK as _FVG_LB,
                    FVG_MIN_SIZE_ATR as _FVG_MIN,
                    FVG_BODY_MULTIPLIER as _FVG_BODY,
                )
            except ImportError:
                _FVG_LB, _FVG_MIN, _FVG_BODY = 96, 0.5, 3.0

        # Загружаем OHLCV для выбранного символа/ТФ
        df_ohlcv_full = load_ohlcv(sel_sym, sel_tf)
        if df_ohlcv_full.empty:
            st.error(f"Не найден OHLCV для {sel_sym} {sel_tf}.")
            st.stop()
        # ── DEBUG: показать диапазоны OHLCV vs trades (для отладки пустого графика) ──
        _ohlcv_start = df_ohlcv_full['timestamp'].min()
        _ohlcv_end   = df_ohlcv_full['timestamp'].max()
        _trades_start = sub_trades['timestamp'].min()
        _trades_end   = sub_trades['timestamp'].max()
        _overlap_n = ((sub_trades['timestamp'] >= _ohlcv_start) &
                      (sub_trades['timestamp'] <= _ohlcv_end)).sum()
        st.caption(
            f"🔎 OHLCV: **{len(df_ohlcv_full):,}** свечей, "
            f"диапазон `{_ohlcv_start:%Y-%m-%d}` → `{_ohlcv_end:%Y-%m-%d}` | "
            f"Trades range: `{_trades_start:%Y-%m-%d}` → `{_trades_end:%Y-%m-%d}` | "
            f"Сделок в диапазоне OHLCV: **{_overlap_n}/{len(sub_trades)}**"
        )
        if _overlap_n == 0:
            st.error(
                f"❌ Сделки выходят за диапазон OHLCV. "
                f"OHLCV закончил {_ohlcv_end:%Y-%m-%d}, первая сделка {_trades_start:%Y-%m-%d}. "
                f"Локальный OHLCV кеш не покрывает период сделок — нужно его обновить."
            )

        # ── Определяем окно ──────────────────────────────────────────────────
        if window_mode == "По одной сделке":
            # Выбор конкретной сделки из таблицы
            sub_trades = sub_trades.sort_values('timestamp').reset_index(drop=True)
            trade_labels = [
                f"{i+1}. {row['timestamp']:%Y-%m-%d %H:%M} | {row['direction']} | "
                f"entry={row['entry']:.6g} | sl={row['sl_price']:.6g} | "
                f"exit_r={row.get('exit_r', 0):.2f}R"
                for i, row in sub_trades.iterrows()
            ]
            idx = st.selectbox(
                "Выберите сделку:", range(len(trade_labels)),
                format_func=lambda i: trade_labels[i],
                key="ph0_trade_idx",
            )
            target_ts = sub_trades.iloc[idx]['timestamp']
            # Окно ±window_bars/2 от bar сделки
            half = window_bars // 2
            mask_ts = df_ohlcv_full['timestamp']
            pos = mask_ts.searchsorted(target_ts)
            lo = max(0, pos - half)
            hi = min(len(df_ohlcv_full), pos + half)
            df_window = df_ohlcv_full.iloc[lo:hi].copy()
            # Рендерим только эту сделку (одну) для чистоты
            sub_show = sub_trades.iloc[[idx]].copy()
        else:
            # Все сделки + buffer ±window_bars/2
            # Берём last window_bars свечей покрывающих сделки
            t_start = t_min - pd.Timedelta(minutes=15 * window_bars)
            t_end   = t_max + pd.Timedelta(minutes=15 * window_bars)
            mask = (df_ohlcv_full['timestamp'] >= t_start) & (df_ohlcv_full['timestamp'] <= t_end)
            df_window = df_ohlcv_full[mask].copy()
            # Ограничим количество свечей для производительности — берём центр диапазона сделок
            if len(df_window) > 2000:
                # центр массы сделок
                center = sub_trades['timestamp'].quantile(0.5)
                pos = df_window['timestamp'].searchsorted(center)
                half = 1000
                lo = max(0, pos - half)
                hi = min(len(df_window), pos + half)
                df_window = df_window.iloc[lo:hi].copy()
                st.caption(f"⚠️ Большой диапазон — обрезано до 2000 свечей вокруг центра сделок")
            # Сделки в окне
            tw_min, tw_max = df_window['timestamp'].min(), df_window['timestamp'].max()
            sub_show = sub_trades[
                (sub_trades['timestamp'] >= tw_min) &
                (sub_trades['timestamp'] <= tw_max)
            ].copy()

        # ── Параметры canonical ICT FVG (исследование пользователя) ──────────
        # threshold_pct       : минимальный gap (в % от цены) — 0 = вообще без фильтра
        # body_multiplier     : middle bar body ≥ M × avg_body за N баров
        # body_lookback       : окно для average body
        # mitigation_threshold: % заполнения зоны для статуса 'mitigated' (0.5 = 50%)
        ph0_col_a, ph0_col_b, ph0_col_c = st.columns(3)
        with ph0_col_a:
            fvg_threshold_pct = st.slider(
                "Min gap (%)", 0.0, 0.5, 0.0, 0.01,
                key=f"ph0_fvg_thr_{_pkey}",
                help="Минимальный размер FVG в % от цены (0 = все 3-bar gaps)",
            )
        with ph0_col_b:
            fvg_body_mult = st.slider(
                "Impulse × avg body", 1.0, 4.0, 1.5, 0.1,
                key=f"ph0_fvg_body_{_pkey}",
                help="Импульсная свеча: body ≥ N × среднего body за 20 баров",
            )
        with ph0_col_c:
            fvg_mit_thr = st.slider(
                "Mitigation %", 0.1, 1.0, 0.5, 0.1,
                key=f"ph0_fvg_mit_{_pkey}",
                help="% заполнения зоны для статуса 'mitigated' (0.5 = 50%)",
            )

        # ── Базовые слои (FVG/HTF/mitigated — существовали раньше) ────────
        ph0_col_h, ph0_col_m, _spacer = st.columns(3)
        with ph0_col_h:
            show_htf_zones = st.checkbox(
                "HTF zones (старший ТФ)", value=False,
                key=f"ph0_show_htf_{_pkey}",
                help="Order Blocks + FVG со старшего ТФ.",
            )
        with ph0_col_m:
            hide_mitigated_fvg = st.checkbox(
                "Скрыть mitigated FVG", value=True,
                key=f"ph0_hide_mit_{_pkey}",
                help="По умолчанию закрытые FVG скрываются.",
            )

        # ── Phase Framework: 4 ОТДЕЛЬНЫХ галки для каждого слоя ──────────
        st.markdown("**🌀 Phase Framework — слои (включай по одному):**")
        ph0_pf1, ph0_pf2, ph0_pf3, ph0_pf4 = st.columns(4)
        with ph0_pf1:
            show_phase_tint = st.checkbox(
                "1️⃣ Phase tint", value=False,
                key=f"ph0_phf_tint_{_pkey}",
                help="Фон: зелёный=balance, красный=impulse, жёлтый=transition.",
            )
        with ph0_pf2:
            show_lvn_bands = st.checkbox(
                "2️⃣ LVN bands", value=False,
                key=f"ph0_phf_lvn_{_pkey}",
                help="Жёлтые пунктирные полосы = низкообъёмные зоны (impulse режет насквозь).",
            )
        with ph0_pf3:
            show_vltr_markers = st.checkbox(
                "3️⃣ VLTR markers", value=False,
                key=f"ph0_phf_vltr_{_pkey}",
                help="Красные кружки над барами с паттерном Volume Led Trend Reversal.",
            )
        with ph0_pf4:
            show_shape_label = st.checkbox(
                "4️⃣ Profile shape", value=False,
                key=f"ph0_phf_shape_{_pkey}",
                help="Надпись формы профиля в правом углу: P/b/I/double_humped/normal.",
            )

        # Параметры (показываются только если что-то из phase framework включено)
        _any_phf_on = show_phase_tint or show_lvn_bands or show_vltr_markers or show_shape_label
        if _any_phf_on:
            phf_mode = st.radio(
                "Phase mode:",
                ["Compact (визуальный, micro-balance)", "Composite (макро-режим, Hurst)"],
                index=0, horizontal=True,
                key=f"ph0_phf_mode_{_pkey}",
                help="Compact: tight range+flat slope detection в окне (matches визуальное восприятие). "
                     "Composite: Hurst+ADX+ATR+BB composite — для долгосрочного режима.",
            )
            phf_is_compact = phf_mode.startswith('Compact')

            if phf_is_compact:
                # Defaults per TF (после калибровки на BTC 1D скриншотах user'а)
                _w_default = {'1D': 30, '1h': 72, '15m': 96, '3m': 200}.get(sel_tf, 30)
                ph0_col_w, ph0_col_r2b, ph0_col_r2i, ph0_col_pers = st.columns(4)
                with ph0_col_w:
                    phf_window = st.slider(
                        "Окно (баров)", 10, 300, int(_w_default), 1,
                        key=f"ph0_phf_window_{_pkey}_{sel_tf}",
                        help="Окно для R² линейной регрессии. "
                             "1D: 25-40 (1-2 месяца), 1h: 60-100, 15m: 80-120.",
                    )
                with ph0_col_r2b:
                    phf_r2_bal_max = st.slider(
                        "R² balance max", 0.05, 0.50, 0.30, 0.05,
                        key=f"ph0_phf_r2bal_{_pkey}",
                        help="R² < N → цена осциллирует (BALANCE). "
                             "Меньше = строже балансом",
                    )
                with ph0_col_r2i:
                    phf_r2_imp_min = st.slider(
                        "R² impulse min", 0.40, 0.95, 0.60, 0.05,
                        key=f"ph0_phf_r2imp_{_pkey}",
                        help="R² > N → цена следует линии (IMPULSE)",
                    )
                with ph0_col_pers:
                    phf_persistence = st.slider(
                        "Persistence", 1, 15, 5, 1,
                        key=f"ph0_phf_pers_{_pkey}",
                        help="Majority vote в окне N баров — устраняет choppy flip-flop",
                    )
                # Legacy params (не используются в Compact v2, но в API нужны)
                phf_range_bal = 2.0
                phf_slope_bal = 0.5
                phf_balance_max = 40.0
                phf_impulse_min = 60.0
            else:
                ph0_col_b, ph0_col_i, _ = st.columns(3)
                with ph0_col_b:
                    phf_balance_max = st.slider(
                        "Balance threshold", 0.0, 60.0, 40.0, 5.0,
                        key=f"ph0_phf_bmax_{_pkey}",
                        help="composite score < N → balance",
                    )
                with ph0_col_i:
                    phf_impulse_min = st.slider(
                        "Impulse threshold", 40.0, 100.0, 60.0, 5.0,
                        key=f"ph0_phf_imin_{_pkey}",
                        help="composite score > N → impulse",
                    )
                phf_window = 100
                phf_range_bal = 2.0
                phf_slope_bal = 0.5
                phf_r2_bal_max = 0.30
                phf_r2_imp_min = 0.60
                phf_persistence = 5

            phf_lvn_pct = st.slider(
                "LVN threshold (% от POC)", 0.05, 0.50, 0.20, 0.05,
                key=f"ph0_phf_lvnpct_{_pkey}",
                help="LVN = volume < N × max volume",
            )
        else:
            phf_is_compact = True
            phf_balance_max = 40.0
            phf_impulse_min = 60.0
            phf_window = 30
            phf_range_bal = 2.0
            phf_slope_bal = 0.5
            phf_r2_bal_max = 0.30
            phf_r2_imp_min = 0.60
            phf_persistence = 5
            phf_lvn_pct = 0.20
        # Backward compat
        show_market_phase = False
        disp_threshold = 50.0

        # Поиск FVG в РАСШИРЕННОМ окне (видимое + 200 баров слева)
        precomputed_zones = None
        if show_fvg:
            from phase0_chart import find_fvg_zones
            ts_full = df_ohlcv_full['timestamp']
            start_idx = int(ts_full.searchsorted(df_window['timestamp'].iloc[0]))
            end_idx   = int(ts_full.searchsorted(df_window['timestamp'].iloc[-1], side='right'))
            ext_start = max(0, start_idx - 200)
            df_ext = df_ohlcv_full.iloc[ext_start:end_idx].copy()
            precomputed_zones = find_fvg_zones(
                df_ext,
                threshold_pct=float(fvg_threshold_pct) / 100.0,
                body_multiplier=float(fvg_body_mult),
                body_lookback=20,
                mitigation_threshold=float(fvg_mit_thr),
            )

        # ── HTF zones (старший ТФ для MTF context) ─────────────────────────
        htf_zones_computed = None
        if show_htf_zones:
            try:
                from phase0_chart import load_ohlcv as _load_o
                from smc_core import find_htf_zones, get_htf_for_tf
                htf = get_htf_for_tf(sel_tf)
                if htf != sel_tf:
                    df_htf = _load_o(sel_sym, htf)
                    if not df_htf.empty:
                        # Берём HTF данные в той же временной зоне что и df_window
                        win_start = df_window['timestamp'].iloc[0]
                        win_end   = df_window['timestamp'].iloc[-1]
                        mask = ((df_htf['timestamp'] >= win_start - pd.Timedelta(days=30)) &
                                (df_htf['timestamp'] <= win_end))
                        df_htf_slice = df_htf[mask].copy().reset_index(drop=True)
                        if len(df_htf_slice) >= 50:
                            htf_zones_computed = find_htf_zones(df_htf_slice, sw=5, fvg_mult=0.5)
            except Exception as _e:
                st.caption(f"⚠️ HTF zones error: {_e}")

        # ── Market Phase (legacy displacement, отключено по умолчанию) ─────
        disp_score_arr = None
        if show_market_phase:
            try:
                from smc_core import calc_displacement_score
                disp_score_arr = calc_displacement_score(df_window)
            except Exception as _e:
                st.caption(f"⚠️ Displacement score error: {_e}")

        # ── Phase Framework: вычисляем только включённые слои ────────────
        phase_labels_arr = None
        lvn_bands_list   = None
        vltr_flags_arr   = None
        profile_shape_str = None

        _vols_for_phf = (df_window['volume'].values
                          if 'volume' in df_window.columns
                          else np.ones(len(df_window)))

        if show_phase_tint:
            try:
                if phf_is_compact:
                    from smc_core import classify_phase_compact
                    phase_info = classify_phase_compact(
                        df_window,
                        window=int(phf_window),
                        r2_balance_max=float(phf_r2_bal_max),
                        r2_impulse_min=float(phf_r2_imp_min),
                        slope_impulse_min=1.5,
                        persistence=int(phf_persistence),
                    )
                else:
                    from smc_core import classify_market_phase
                    phase_info = classify_market_phase(
                        df_window,
                        balance_max_score=float(phf_balance_max),
                        impulse_min_score=float(phf_impulse_min),
                        hurst_window=100, use_corrected_hurst=True,
                    )
                phase_labels_arr = phase_info['phase']
                n_bal = int((phase_labels_arr == 'balance').sum())
                n_imp = int((phase_labels_arr == 'impulse').sum())
                n_tr  = int((phase_labels_arr == 'transition').sum())
                _mode = 'Compact' if phf_is_compact else 'Composite'
                st.caption(
                    f"🟢 balance={n_bal}  🔴 impulse={n_imp}  🟡 transition={n_tr} "
                    f"| Mode: {_mode}  window={int(phf_window) if phf_is_compact else 100}"
                )
            except Exception as _e:
                st.caption(f"⚠️ Phase tint error: {_e}")

        if show_lvn_bands:
            try:
                from smc_core import find_lvn_zones_window
                lvn_bands_list = find_lvn_zones_window(
                    df_window['high'].values, df_window['low'].values, _vols_for_phf,
                    rows=24, lvn_pct=float(phf_lvn_pct),
                )
                st.caption(f"🟡 LVN bands: {len(lvn_bands_list)}")
            except Exception as _e:
                st.caption(f"⚠️ LVN error: {_e}")

        if show_vltr_markers:
            try:
                from smc_core import detect_vltr
                vltr_flags_arr = detect_vltr(
                    df_window, lookback=96, lvn_pct=float(phf_lvn_pct),
                )
                st.caption(f"🔴 VLTR markers: {int(vltr_flags_arr.sum())}")
            except Exception as _e:
                st.caption(f"⚠️ VLTR error: {_e}")

        if show_shape_label:
            try:
                from smc_core import classify_profile_shape
                profile_shape_str = classify_profile_shape(
                    df_window['high'].values, df_window['low'].values, _vols_for_phf,
                    rows=24,
                )
                st.caption(f"📐 Profile shape: **{profile_shape_str}**")
            except Exception as _e:
                st.caption(f"⚠️ Shape error: {_e}")

        # ── DEBUG: что попало в df_window перед рендером ──────────────────────
        st.caption(
            f"🎬 Render: df_window=**{len(df_window):,}** свечей, "
            f"sub_show=**{len(sub_show)}** сделок"
            + (f", FVG zones=**{len(precomputed_zones)}**" if precomputed_zones else "")
            + (f", HTF zones=**{len(htf_zones_computed)}**" if htf_zones_computed else "")
        )
        if df_window.empty:
            st.error(
                "❌ df_window пустой — нет свечей для рендера. "
                "Скорее всего timestamps сделок выходят за диапазон локального OHLCV. "
                "Попробуй переключить режим 'Окно' на 'По одной сделке'."
            )
            st.stop()

        # ── Рендер графика ───────────────────────────────────────────────────
        title = f"{sel_sym} {sel_tf} — {sel_parquet.stem.replace('_raw', '')} ({sel_split})"
        try:
            fig = make_chart(
                df_window, sub_show, title=title, rr=float(rr_value),
                show_fvg=bool(show_fvg),
                precomputed_fvg_zones=precomputed_zones,
                hide_mitigated_fvg=bool(hide_mitigated_fvg),
                htf_zones=htf_zones_computed,
                displacement_score=disp_score_arr,
                displacement_threshold=float(disp_threshold),
                # Phase Framework overlays
                phase_labels=phase_labels_arr,
                lvn_bands=lvn_bands_list,
                vltr_flags=vltr_flags_arr,
                profile_shape=profile_shape_str,
            )
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
        except Exception as _e:
            import traceback
            st.error(f"❌ Ошибка рендера графика: {_e}")
            st.code(traceback.format_exc(), language='python')

        # ── Таблица сделок в окне ────────────────────────────────────────────
        with st.expander(f"📋 Сделки в окне ({len(sub_show)})", expanded=False):
            cols_show = ['timestamp', 'direction', 'entry', 'sl_price',
                          'mae_r', 'mfe_r', 'exit_r', 'sl_hit',
                          'bars_to_sl', 'bars_to_mfe', 'fill_status', 'pattern']
            cols_present = [c for c in cols_show if c in sub_show.columns]
            st.dataframe(
                sub_show[cols_present].reset_index(drop=True),
                use_container_width=True,
                height=400,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# ЛАБОРАТОРИЯ 2026 — РЕНДЕР ВКЛАДОК 18-21 (Группа 3)
# Полностью изолированы от вкладок 1-17. Данные берутся независимо.
# ═══════════════════════════════════════════════════════════════════════════════

if tab in ("18. 🔵 Крипто Фьючерсы", "19. 🟡 Форекс & Металлы",
           "20. 🟢 Крипто Спот", "21. 📊 RS & Heatmap"):

    st.markdown("---")
    st.markdown("### 🔬 Лаборатория 2026")

    # ─── ВКЛАДКА 18: КРИПТО ФЬЮЧЕРСЫ ────────────────────────────────────────
    if tab == "18. 🔵 Крипто Фьючерсы":
        _all_futures = ["BTCUSDT","ETHUSDT"] + LIST_ALTS_MAIN + LIST_ALTS_2 + LIST_ALTS_3

        col18a, col18b, col18c = st.columns(3)
        with col18a:
            lab_sym = st.selectbox("Актив:", _all_futures, key="lab18_sym")
            lab_tf  = st.selectbox("Таймфрейм:", ["15m","1h","1D"], key="lab18_tf")
        with col18b:
            show_sq18  = st.checkbox("Squeeze Momentum", value=True,  key="lab18_sq")
            show_rsi18 = st.checkbox("RSI дивергенции",  value=True,  key="lab18_rsi")
            show_vsa18 = st.checkbox("VSA паттерны",     value=False, key="lab18_vsa")
        with col18c:
            show_adx18 = st.checkbox("ADX фильтр",       value=True,  key="lab18_adx")
            show_bb18  = st.checkbox("BB Squeeze",        value=True,  key="lab18_bb")

        df18 = fetch_data_bg(lab_sym, lab_tf)
        if df18 is None or df18.empty:
            st.warning("Нет данных")
            st.stop()
        df18 = apply_order_flow(df18)

        # ── Индикаторы
        fvgs18  = lab_calc_fvg(df18)
        obs18   = lab_calc_ob(df18)
        sess18  = lab_get_sessions(df18) if lab_tf in ("15m","1h") else []
        vsa18   = lab_calc_vsa(df18) if show_vsa18 else []
        rsi18, bull_rsi18, bear_rsi18 = (
            lab_calc_rsi_divergence(df18) if show_rsi18 else (None,[],[]))
        adx18, pdi18, ndi18 = (
            lab_calc_adx(df18) if show_adx18 else (None,None,None))
        sq_on18, sq_off18, sq_mom18 = (
            lab_calc_squeeze_momentum(df18) if show_sq18 else (None,None,None))
        bb_up18, bb_mid18, bb_lo18, bb_sq18 = (
            lab_calc_bb_squeeze(df18) if show_bb18 else (None,None,None,None))

        # ── ADX сигнал
        adx_val   = float(adx18[-1]) if adx18 is not None else 0
        trend_ok  = adx_val >= 20  # тренд есть
        trend_str = ("🔥 Сильный тренд" if adx_val >= 40
                     else "📈 Умеренный тренд" if adx_val >= 20
                     else "😴 Флэт — осторожно")

        # ── Субплоты
        rows18   = [0.48, 0.15, 0.15, 0.12, 0.10]
        titles18 = [f"{lab_sym} Фьючерс | {lab_tf}",
                    "CVD", "RSI(14)", "ADX", "Squeeze Momentum"]
        fig18 = make_subplots(rows=5, cols=1, shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=rows18,
            subplot_titles=titles18)

        # ── Сессионные окна
        lab_add_sessions_to_fig(fig18, sess18, row=1)

        # ── Bollinger Bands
        if show_bb18 and bb_up18 is not None:
            fig18.add_trace(go.Scatter(x=df18['timestamp'], y=bb_up18,
                line=dict(color='rgba(150,150,255,0.4)', width=1),
                showlegend=False), 1, 1)
            fig18.add_trace(go.Scatter(x=df18['timestamp'], y=bb_lo18,
                line=dict(color='rgba(150,150,255,0.4)', width=1),
                fill='tonexty', fillcolor='rgba(100,100,255,0.05)',
                showlegend=False), 1, 1)
            if bb_sq18 is not None:
                sq_x18 = [df18['timestamp'].iloc[i] for i in range(len(bb_sq18)) if bb_sq18[i]]
                sq_y18 = [float(df18['close'].iloc[i]) for i in range(len(bb_sq18)) if bb_sq18[i]]
                if sq_x18:
                    fig18.add_trace(go.Scatter(x=sq_x18, y=sq_y18, mode='markers',
                        marker=dict(symbol='diamond', size=7, color='#FFD700', opacity=0.8),
                        showlegend=False), 1, 1)

        # ── Свечи
        fig18.add_trace(go.Candlestick(
            x=df18['timestamp'],
            open=df18['open'], high=df18['high'],
            low=df18['low'],   close=df18['close'],
            increasing_line_color='#00E676',
            decreasing_line_color='#FF1744',
            name=lab_sym), 1, 1)

        # ── FVG + OB + ATR
        lab_add_fvg_to_fig(fig18, fvgs18, row=1)
        lab_add_ob_to_fig(fig18,  obs18,  row=1)
        lab_add_atr_stops_to_fig(fig18, df18, row=1)

        # ── VSA паттерны
        if show_vsa18:
            for sig in vsa18:
                fig18.add_annotation(
                    x=sig['ts'], y=sig['price'],
                    text=sig['label'], showarrow=False,
                    font=dict(color=sig['color'], size=10),
                    yanchor='bottom' if sig['type'] != 'UT' else 'top',
                    row=1, col=1)

        # ── Squeeze Momentum OFF маркеры на ценовом графике
        if show_sq18 and sq_off18 is not None:
            off_x = [df18['timestamp'].iloc[i] for i in range(len(sq_off18)) if sq_off18[i]]
            off_y = [float(df18['close'].iloc[i]) for i in range(len(sq_off18)) if sq_off18[i]]
            if off_x:
                fig18.add_trace(go.Scatter(x=off_x, y=off_y, mode='markers',
                    marker=dict(symbol='star', size=14, color='#FF9800', opacity=0.9),
                    name='Squeeze OFF', showlegend=False), 1, 1)

        # ── CVD (row=2)
        fig18.add_trace(go.Scatter(
            x=df18['timestamp'], y=df18['cvd'],
            line=dict(color='#00E676', width=1.5),
            name="CVD"), 2, 1)

        # ── RSI (row=3)
        if rsi18 is not None:
            fig18.add_trace(go.Scatter(
                x=df18['timestamp'], y=rsi18,
                line=dict(color='#AA88FF', width=1.5),
                name='RSI'), 3, 1)
            for lvl, clr in [(70,'rgba(255,80,80,0.35)'),(30,'rgba(0,200,80,0.35)'),(50,'rgba(255,255,255,0.15)')]:
                fig18.add_hline(y=lvl, line=dict(color=clr, width=1, dash='dot'), row=3, col=1)
            for t1,t2,p1,p2,r1,r2 in bull_rsi18:
                fig18.add_trace(go.Scatter(x=[t1,t2], y=[p1,p2], mode='lines',
                    line=dict(color='rgba(0,230,118,0.8)',width=2,dash='dot'),
                    showlegend=False), 1, 1)
                fig18.add_trace(go.Scatter(x=[t1,t2], y=[r1,r2], mode='lines',
                    line=dict(color='rgba(0,230,118,0.8)',width=2,dash='dot'),
                    showlegend=False), 3, 1)
            for t1,t2,p1,p2,r1,r2 in bear_rsi18:
                fig18.add_trace(go.Scatter(x=[t1,t2], y=[p1,p2], mode='lines',
                    line=dict(color='rgba(255,82,82,0.8)',width=2,dash='dot'),
                    showlegend=False), 1, 1)
                fig18.add_trace(go.Scatter(x=[t1,t2], y=[r1,r2], mode='lines',
                    line=dict(color='rgba(255,82,82,0.8)',width=2,dash='dot'),
                    showlegend=False), 3, 1)

        # ── ADX (row=4)
        if adx18 is not None:
            fig18.add_trace(go.Scatter(
                x=df18['timestamp'], y=adx18,
                line=dict(color='#FF9800', width=2),
                name='ADX'), 4, 1)
            fig18.add_trace(go.Scatter(
                x=df18['timestamp'], y=pdi18,
                line=dict(color='#00E676', width=1, dash='dot'),
                name='+DI'), 4, 1)
            fig18.add_trace(go.Scatter(
                x=df18['timestamp'], y=ndi18,
                line=dict(color='#FF1744', width=1, dash='dot'),
                name='-DI'), 4, 1)
            fig18.add_hline(y=20, line=dict(color='rgba(255,152,0,0.5)',width=1,dash='dot'), row=4, col=1)
            fig18.add_hline(y=40, line=dict(color='rgba(255,152,0,0.8)',width=1,dash='dot'), row=4, col=1)

        # ── Squeeze Momentum гистограмма (row=5)
        if show_sq18 and sq_mom18 is not None:
            mom_colors = []
            for i in range(len(sq_mom18)):
                if sq_on18 is not None and sq_on18[i]:
                    mom_colors.append('#888888')  # серый = сжатие
                elif sq_mom18[i] >= 0:
                    mom_colors.append('#00E676')   # зелёный = бычий
                else:
                    mom_colors.append('#FF1744')   # красный = медвежий
            fig18.add_trace(go.Bar(
                x=df18['timestamp'], y=sq_mom18,
                marker_color=mom_colors,
                name='Squeeze Mom'), 5, 1)
            fig18.add_hline(y=0, line=dict(color='rgba(255,255,255,0.3)',width=1), row=5, col=1)

        fig18.update_layout(height=950, template="plotly_dark",
            margin=dict(l=0,r=50,t=25,b=0),
            xaxis_rangeslider_visible=False,
            showlegend=False)
        fig18.update_yaxes(side="right")
        st.plotly_chart(fig18, use_container_width=True,
            config={'scrollZoom': True})

        # ── Инфопанель
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            atr_v, sl_l, sl_s = lab_calc_atr_stops(df18)
            if atr_v:
                st.metric("ATR(14)", f"{atr_v:.4f}")
        with c2:
            st.metric("ADX", f"{adx_val:.1f}", delta=trend_str, delta_color="off")
        with c3:
            if adx18 is not None and pdi18 is not None:
                bias = "🟢 Бычий" if pdi18[-1] > ndi18[-1] else "🔴 Медвежий"
                st.metric("DI Bias", bias)
        with c4:
            if show_sq18 and sq_on18 is not None:
                sq_status = "⚡ Выход из сжатия!" if sq_off18[-1] else ("🔘 Сжатие" if sq_on18[-1] else "📊 Открыто")
                st.metric("Squeeze", sq_status)

        if not trend_ok and show_adx18:
            st.warning("⚠️ ADX < 20 — рынок во флэте. Высокая вероятность ложных сигналов. Лучше подождать.")

    # ─── ВКЛАДКА 19: ФОРЕКС & МЕТАЛЛЫ ────────────────────────────────────────
    elif tab == "19. 🟡 Форекс & Металлы":
        import yfinance as yf19

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            lab_sym19 = st.selectbox("Актив:",
                ["EURUSD=X","GBPUSD=X","GC=F"], key="lab19_sym")
        with col_b:
            lab_tf19 = st.selectbox("Таймфрейм:", ["15m","1h","1D"], key="lab19_tf")
        with col_c:
            show_vsa19 = st.checkbox("VSA паттерны",     value=True,  key="lab19_vsa")
            show_rsi19 = st.checkbox("RSI дивергенции", value=True,  key="lab19_rsi")
            show_bb19  = st.checkbox("BB Squeeze",       value=True,  key="lab19_bb")
            show_adx19 = st.checkbox("ADX фильтр",       value=True,  key="lab19_adx")
            show_sb19  = st.checkbox("ICT Silver Bullet", value=True, key="lab19_sb")

        def _yf_load(symbol, tf):
            """Загрузка данных Yahoo Finance с нормализацией колонок."""
            period_map = {"15m": "5d", "1h": "60d", "1D": "2y"}
            iv_map     = {"15m": "15m", "1h": "1h", "1D": "1d"}
            raw = yf19.download(symbol, period=period_map[tf],
                                interval=iv_map[tf],
                                progress=False, auto_adjust=True)
            if raw.empty:
                return None
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0].lower() for c in raw.columns]
            else:
                raw.columns = [c.lower() for c in raw.columns]
            raw = raw.reset_index()
            date_col = next((c for c in ['datetime','date','Datetime','Date']
                             if c in raw.columns), raw.columns[0])
            raw['timestamp'] = pd.to_datetime(raw[date_col])
            return raw[['timestamp','open','high','low','close','volume']].dropna().tail(300)

        try:
            df19 = _yf_load(lab_sym19, lab_tf19)
            if df19 is None or df19.empty:
                st.warning("Нет данных Yahoo Finance")
                st.stop()
        except Exception as e:
            st.error(f"Ошибка загрузки: {e}")
            st.stop()

        # Для Золота грузим DXY корреляцию
        dxy_df = None
        if lab_sym19 == "GC=F":
            try:
                dxy_df = _yf_load("DX-Y.NYB", lab_tf19)
            except Exception:
                pass

        # Считаем все индикаторы
        fvgs19    = lab_calc_fvg(df19)
        obs19     = lab_calc_ob(df19)
        sess19    = lab_get_sessions(df19) if lab_tf19 in ("15m","1h") else []
        asian19   = lab_calc_asian_session(df19) if lab_tf19 in ("15m","1h") else []
        vsa19     = lab_calc_vsa(df19) if show_vsa19 else []
        rsi19, bull_rsi19, bear_rsi19 = (
            lab_calc_rsi_divergence(df19) if show_rsi19 else (None, [], []))
        bb_up, bb_mid, bb_lo, bb_sq = (
            lab_calc_bb_squeeze(df19) if show_bb19 else (None, None, None, None))
        adx19, pdi19, ndi19 = (
            lab_calc_adx(df19) if show_adx19 else (None, None, None))
        sb_windows19 = (
            lab_get_silver_bullet_windows(df19) if show_sb19 and lab_tf19 in ("15m","1h") else [])
        # London Breakout уровни (из азиатского диапазона)
        asian_lb19 = lab_calc_asian_session(df19) if lab_tf19 in ("15m","1h") else []

        # Определяем количество субплотов
        n_rows19      = 3  # цена + RSI + ADX
        row_heights19 = [0.58, 0.17, 0.15]
        titles19      = [f"{lab_sym19} | {lab_tf19}", "RSI(14)", "ADX"]
        if dxy_df is not None:
            n_rows19 += 1
            row_heights19.append(0.10)
            titles19.append("DXY (корреляция)")

        fig19 = make_subplots(
            rows=n_rows19, cols=1, shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights19,
            subplot_titles=titles19)

        # ── Сессионные окна (Азия/Лондон/NY)
        lab_add_sessions_to_fig(fig19, sess19, row=1)

        # ── ICT Silver Bullet окна (золотые/синие/фиолетовые полосы)
        for sb in sb_windows19:
            fig19.add_vrect(
                x0=sb['x0'], x1=sb['x1'],
                fillcolor=sb['color'], line_width=0,
                row=1, col=1)
            fig19.add_annotation(
                x=sb['x0'], y=1, yref='paper',
                text=sb['label'], showarrow=False,
                font=dict(color='rgba(255,215,0,0.7)', size=8),
                xanchor='left', yanchor='top')

        # ── ICT Power of 3: диапазон Азиатской сессии
        for asn in asian19:
            ts_end_str = asn['ts_end']
            fig19.add_shape(type='rect',
                x0=asn['ts_start'], x1=ts_end_str,
                y0=asn['low'], y1=asn['high'],
                fillcolor='rgba(100,150,255,0.08)',
                line=dict(color='rgba(100,150,255,0.4)', width=1, dash='dot'),
                row=1, col=1)
            fig19.add_annotation(
                x=asn['ts_start'], y=asn['high'],
                text='🌙 Asia', showarrow=False,
                font=dict(color='rgba(150,180,255,0.8)', size=9),
                xanchor='left', yanchor='bottom', row=1, col=1)

        # ── London Breakout — уровни пробоя азиатского диапазона
        for asn_lb in asian_lb19:
            # Линия хая азиатской сессии — уровень для Long
            fig19.add_shape(type='line',
                x0=asn_lb['ts_end'], x1=asn_lb['ts_end'] + pd.Timedelta(hours=8),
                y0=asn_lb['high'], y1=asn_lb['high'],
                line=dict(color='rgba(0,230,118,0.6)', width=1.5, dash='dash'),
                row=1, col=1)
            # Линия лоя азиатской сессии — уровень для Short
            fig19.add_shape(type='line',
                x0=asn_lb['ts_end'], x1=asn_lb['ts_end'] + pd.Timedelta(hours=8),
                y0=asn_lb['low'], y1=asn_lb['low'],
                line=dict(color='rgba(255,82,82,0.6)', width=1.5, dash='dash'),
                row=1, col=1)

        # ── Свечной график
        fig19.add_trace(go.Candlestick(
            x=df19['timestamp'],
            open=df19['open'], high=df19['high'],
            low=df19['low'],   close=df19['close'],
            increasing_line_color='#00E676',
            decreasing_line_color='#FF1744',
            name=lab_sym19), 1, 1)

        # ── Bollinger Bands
        if show_bb19 and bb_up is not None:
            fig19.add_trace(go.Scatter(
                x=df19['timestamp'], y=bb_up,
                line=dict(color='rgba(150,150,255,0.4)', width=1),
                name='BB Upper', showlegend=False), 1, 1)
            fig19.add_trace(go.Scatter(
                x=df19['timestamp'], y=bb_lo,
                line=dict(color='rgba(150,150,255,0.4)', width=1),
                fill='tonexty', fillcolor='rgba(100,100,255,0.05)',
                name='BB Lower', showlegend=False), 1, 1)
            fig19.add_trace(go.Scatter(
                x=df19['timestamp'], y=bb_mid,
                line=dict(color='rgba(150,150,255,0.25)', width=1, dash='dot'),
                name='BB Mid', showlegend=False), 1, 1)
            # Маркеры Squeeze
            if bb_sq is not None:
                sq_ts = [df19['timestamp'].iloc[i]
                         for i in range(len(bb_sq)) if bb_sq[i]]
                sq_pr = [float(df19['close'].iloc[i])
                         for i in range(len(bb_sq)) if bb_sq[i]]
                if sq_ts:
                    fig19.add_trace(go.Scatter(
                        x=sq_ts, y=sq_pr, mode='markers',
                        marker=dict(symbol='diamond', size=6,
                                    color='#FFD700', opacity=0.7),
                        name='Squeeze', showlegend=False), 1, 1)

        # ── FVG и Order Blocks
        lab_add_fvg_to_fig(fig19, fvgs19, row=1)
        lab_add_ob_to_fig(fig19,  obs19,  row=1)

        # ── ATR динамический стоп
        lab_add_atr_stops_to_fig(fig19, df19, row=1)

        # ── VSA паттерны
        if show_vsa19:
            vsa_colors = {'SC': '#FFD700', 'UT': '#FF4444', 'ND': '#AAAAAA'}
            for sig in vsa19:
                fig19.add_annotation(
                    x=sig['ts'], y=sig['price'],
                    text=sig['label'], showarrow=False,
                    font=dict(color=sig['color'], size=10),
                    yanchor='bottom' if sig['type'] != 'UT' else 'top',
                    row=1, col=1)

        # ── RSI субплот
        if rsi19 is not None:
            fig19.add_trace(go.Scatter(
                x=df19['timestamp'], y=rsi19,
                line=dict(color='#AA88FF', width=1.5),
                name='RSI'), 2, 1)
            # Зоны 70/30
            for level, color in [(70, 'rgba(255,80,80,0.3)'), (30, 'rgba(0,200,80,0.3)')]:
                fig19.add_hline(y=level,
                    line=dict(color=color, width=1, dash='dot'), row=2, col=1)
            # RSI дивергенции
            for t1, t2, p1, p2, r1, r2 in bull_rsi19:
                fig19.add_trace(go.Scatter(
                    x=[t1,t2], y=[p1,p2], mode='lines',
                    line=dict(color='rgba(0,230,118,0.8)', width=2, dash='dot'),
                    showlegend=False), 1, 1)
                fig19.add_trace(go.Scatter(
                    x=[t1,t2], y=[r1,r2], mode='lines',
                    line=dict(color='rgba(0,230,118,0.8)', width=2, dash='dot'),
                    showlegend=False), 2, 1)
            for t1, t2, p1, p2, r1, r2 in bear_rsi19:
                fig19.add_trace(go.Scatter(
                    x=[t1,t2], y=[p1,p2], mode='lines',
                    line=dict(color='rgba(255,82,82,0.8)', width=2, dash='dot'),
                    showlegend=False), 1, 1)
                fig19.add_trace(go.Scatter(
                    x=[t1,t2], y=[r1,r2], mode='lines',
                    line=dict(color='rgba(255,82,82,0.8)', width=2, dash='dot'),
                    showlegend=False), 2, 1)

        # ── ADX (row=3)
        if adx19 is not None:
            fig19.add_trace(go.Scatter(
                x=df19['timestamp'], y=adx19,
                line=dict(color='#FF9800', width=2), name='ADX'), 3, 1)
            fig19.add_trace(go.Scatter(
                x=df19['timestamp'], y=pdi19,
                line=dict(color='#00E676', width=1, dash='dot'), name='+DI'), 3, 1)
            fig19.add_trace(go.Scatter(
                x=df19['timestamp'], y=ndi19,
                line=dict(color='#FF1744', width=1, dash='dot'), name='-DI'), 3, 1)
            fig19.add_hline(y=20, line=dict(color='rgba(255,152,0,0.5)',width=1,dash='dot'), row=3, col=1)
            fig19.add_hline(y=40, line=dict(color='rgba(255,152,0,0.8)',width=1,dash='dot'), row=3, col=1)

        # ── DXY корреляция для Золота
        if dxy_df is not None:
            dxy_row = n_rows19
            fig19.add_trace(go.Scatter(
                x=dxy_df['timestamp'], y=dxy_df['close'],
                line=dict(color='#FF9800', width=1.5),
                name='DXY'), dxy_row, 1)
            fig19.add_annotation(
                text='📌 Золото ↑ = DXY ↓ (обратная корреляция)',
                xref='paper', yref='paper',
                x=0.01, y=0.01,
                showarrow=False,
                font=dict(color='rgba(255,152,0,0.7)', size=10))

        fig19.update_layout(
            height=800 if dxy_df is None else 950,
            template="plotly_dark",
            margin=dict(l=0,r=50,t=25,b=0),
            xaxis_rangeslider_visible=False,
            showlegend=False)
        fig19.update_yaxes(side="right")
        st.plotly_chart(fig19, use_container_width=True,
            config={'scrollZoom': True})

        # ── Информационная панель внизу
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        atr_v, sl_long, sl_short = lab_calc_atr_stops(df19)
        if atr_v:
            with col1:
                st.metric("ATR(14)", f"{atr_v:.5f}")
                st.caption("Мера волатильности")
            with col2:
                st.metric("SL Long", f"{sl_long:.5f}",
                    delta=f"-{atr_v*1.5:.5f}", delta_color="inverse")
            with col3:
                st.metric("SL Short", f"{sl_short:.5f}",
                    delta=f"+{atr_v*1.5:.5f}")

        # VSA расшифровка
        if show_vsa19 and vsa19:
            with st.expander("📖 VSA паттерны на графике"):
                st.markdown("""
| Метка | Паттерн | Значение |
|---|---|---|
| 🟡 SC | Selling Climax | Кульминация продаж — покупатели поглощают панику |
| 🔴 UT | Upthrust | Ложный пробой хая — продавцы встретили рост |
| ⚪ ND | No Demand | Слабый рост без объёма — покупатели не поддерживают |
                """)

    # ─── ВКЛАДКА 20: КРИПТО СПОТ ─────────────────────────────────────────────
    elif tab == "20. 🟢 Крипто Спот":
        _spot_syms = ["BTCUSDT","ETHUSDT"] + LIST_ALTS_MAIN + LIST_ALTS_2

        col20a, col20b, col20c = st.columns(3)
        with col20a:
            lab_sym20 = st.selectbox("Актив:", _spot_syms, key="lab20_sym")
        with col20b:
            show_rsi20  = st.checkbox("RSI дивергенции", value=True, key="lab20_rsi")
            show_bb20   = st.checkbox("BB Squeeze",       value=True, key="lab20_bb")
        with col20c:
            show_perp20 = st.checkbox("Спот vs Фьюч CVD", value=True, key="lab20_perp")
            show_zsc20  = st.checkbox("Z-Score спреда",   value=False, key="lab20_zsc")

        # Загрузка спот данных
        df20 = lab_load_spot_bybit(lab_sym20, limit=300)
        if df20 is None or df20.empty:
            st.warning("Нет данных Bybit Spot")
            st.stop()
        df20 = apply_order_flow(df20)

        # Загрузка фьючерс данных для сравнения CVD
        perp20 = None
        if show_perp20:
            perp20 = lab_load_perp_bybit(lab_sym20, limit=300)
            if perp20 is not None:
                perp20 = apply_order_flow(perp20)

        # Z-Score спреда BTC/ETH (коинтеграция)
        zsc_ts, zsc_vals = None, None
        if show_zsc20 and lab_sym20 in ("BTCUSDT","ETHUSDT"):
            pair_sym = "ETHUSDT" if lab_sym20 == "BTCUSDT" else "BTCUSDT"
            pair_df  = lab_load_spot_bybit(pair_sym, limit=300)
            if pair_df is not None:
                zsc_ts, zsc_vals = lab_calc_zscore_spread(df20, pair_df, window=30)

        # Индикаторы
        fvgs20 = lab_calc_fvg(df20)
        obs20  = lab_calc_ob(df20)
        bb_up20, bb_mid20, bb_lo20, bb_sq20 = (
            lab_calc_bb_squeeze(df20) if show_bb20 else (None,None,None,None))
        rsi20, bull_rsi20, bear_rsi20 = (
            lab_calc_rsi_divergence(df20) if show_rsi20 else (None,[],[]))

        # Структурные дивергенции CVD
        bull_div20, bear_div20 = find_structural_divergences(
            df20.assign(brick_size=float(df20['close'].iloc[-1]) * 0.01),
            reversal_bricks=4)

        # Сигнал Спот/Фьюч дивергенция
        squeeze_signal = lab_calc_spot_perp_divergence(df20, perp20) if perp20 is not None else None

        # ── Определяем субплоты
        rows20   = [0.52, 0.16, 0.16]
        titles20 = [f"{lab_sym20} Спот | 1D", "CVD Спот", "RSI(14)"]
        n20 = 3
        if perp20 is not None:
            rows20.append(0.16)
            titles20.append("CVD Фьюч")
            n20 += 1
        if show_zsc20 and zsc_vals is not None:
            rows20.append(0.12)
            titles20.append(f"Z-Score BTC/ETH")
            n20 += 1

        fig20 = make_subplots(rows=n20, cols=1, shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=rows20,
            subplot_titles=titles20)

        # ── Свечи
        fig20.add_trace(go.Candlestick(
            x=df20['timestamp'],
            open=df20['open'], high=df20['high'],
            low=df20['low'],   close=df20['close'],
            increasing_line_color='#00E676',
            decreasing_line_color='#FF1744',
            name=lab_sym20), 1, 1)

        # ── Bollinger Bands
        if show_bb20 and bb_up20 is not None:
            fig20.add_trace(go.Scatter(x=df20['timestamp'], y=bb_up20,
                line=dict(color='rgba(150,150,255,0.4)', width=1),
                showlegend=False), 1, 1)
            fig20.add_trace(go.Scatter(x=df20['timestamp'], y=bb_lo20,
                line=dict(color='rgba(150,150,255,0.4)', width=1),
                fill='tonexty', fillcolor='rgba(100,100,255,0.05)',
                showlegend=False), 1, 1)
            if bb_sq20 is not None:
                sq_x = [df20['timestamp'].iloc[i] for i in range(len(bb_sq20)) if bb_sq20[i]]
                sq_y = [float(df20['close'].iloc[i]) for i in range(len(bb_sq20)) if bb_sq20[i]]
                if sq_x:
                    fig20.add_trace(go.Scatter(x=sq_x, y=sq_y, mode='markers',
                        marker=dict(symbol='diamond', size=7, color='#FFD700', opacity=0.8),
                        showlegend=False, name='Squeeze'), 1, 1)

        # ── FVG + OB + ATR стоп
        lab_add_fvg_to_fig(fig20, fvgs20, row=1)
        lab_add_ob_to_fig(fig20,  obs20,  row=1)
        lab_add_atr_stops_to_fig(fig20, df20, row=1)

        # ── Структурные CVD дивергенции
        for s_i, e_i in bull_div20:
            fig20.add_trace(go.Scatter(
                x=[df20['timestamp'].iloc[s_i], df20['timestamp'].iloc[e_i]],
                y=[float(df20['close'].iloc[s_i]), float(df20['close'].iloc[e_i])],
                mode='lines', line=dict(color='rgba(0,230,118,0.9)',width=2,dash='dot'),
                hoverinfo='skip', showlegend=False), 1, 1)
            fig20.add_trace(go.Scatter(
                x=[df20['timestamp'].iloc[s_i], df20['timestamp'].iloc[e_i]],
                y=[float(df20['cvd'].iloc[s_i]), float(df20['cvd'].iloc[e_i])],
                mode='lines', line=dict(color='rgba(0,230,118,0.9)',width=2,dash='dot'),
                hoverinfo='skip', showlegend=False), 2, 1)
            fig20.add_annotation(
                x=df20['timestamp'].iloc[e_i], y=float(df20['close'].iloc[e_i]),
                text='🟢 Спринг', showarrow=False,
                font=dict(color='#00E676', size=11),
                yanchor='top', xanchor='left', row=1, col=1)
        for s_i, e_i in bear_div20:
            fig20.add_trace(go.Scatter(
                x=[df20['timestamp'].iloc[s_i], df20['timestamp'].iloc[e_i]],
                y=[float(df20['close'].iloc[s_i]), float(df20['close'].iloc[e_i])],
                mode='lines', line=dict(color='rgba(255,82,82,0.9)',width=2,dash='dot'),
                hoverinfo='skip', showlegend=False), 1, 1)
            fig20.add_trace(go.Scatter(
                x=[df20['timestamp'].iloc[s_i], df20['timestamp'].iloc[e_i]],
                y=[float(df20['cvd'].iloc[s_i]), float(df20['cvd'].iloc[e_i])],
                mode='lines', line=dict(color='rgba(255,82,82,0.9)',width=2,dash='dot'),
                hoverinfo='skip', showlegend=False), 2, 1)
            fig20.add_annotation(
                x=df20['timestamp'].iloc[e_i], y=float(df20['close'].iloc[e_i]),
                text='🔴 Аптраст', showarrow=False,
                font=dict(color='#FF5252', size=11),
                yanchor='bottom', xanchor='left', row=1, col=1)

        # ── CVD Спот (row=2)
        fig20.add_trace(go.Scatter(
            x=df20['timestamp'], y=df20['cvd'],
            line=dict(color='#00E676', width=2),
            name="CVD Спот"), 2, 1)

        # ── RSI (row=3)
        if rsi20 is not None:
            fig20.add_trace(go.Scatter(
                x=df20['timestamp'], y=rsi20,
                line=dict(color='#AA88FF', width=1.5),
                name='RSI'), 3, 1)
            for lvl, clr in [(70,'rgba(255,80,80,0.3)'),(30,'rgba(0,200,80,0.3)')]:
                fig20.add_hline(y=lvl,
                    line=dict(color=clr,width=1,dash='dot'), row=3, col=1)
            for t1,t2,p1,p2,r1,r2 in bull_rsi20:
                fig20.add_trace(go.Scatter(x=[t1,t2], y=[p1,p2], mode='lines',
                    line=dict(color='rgba(0,230,118,0.8)',width=2,dash='dot'),
                    showlegend=False), 1, 1)
                fig20.add_trace(go.Scatter(x=[t1,t2], y=[r1,r2], mode='lines',
                    line=dict(color='rgba(0,230,118,0.8)',width=2,dash='dot'),
                    showlegend=False), 3, 1)
            for t1,t2,p1,p2,r1,r2 in bear_rsi20:
                fig20.add_trace(go.Scatter(x=[t1,t2], y=[p1,p2], mode='lines',
                    line=dict(color='rgba(255,82,82,0.8)',width=2,dash='dot'),
                    showlegend=False), 1, 1)
                fig20.add_trace(go.Scatter(x=[t1,t2], y=[r1,r2], mode='lines',
                    line=dict(color='rgba(255,82,82,0.8)',width=2,dash='dot'),
                    showlegend=False), 3, 1)

        # ── CVD Фьюч (row=4 если есть)
        if perp20 is not None and n20 >= 4:
            fig20.add_trace(go.Scatter(
                x=perp20['timestamp'], y=perp20['cvd'],
                line=dict(color='#FF9800', width=1.5),
                name="CVD Фьюч"), 4, 1)

        # ── Z-Score спреда (последний row)
        if show_zsc20 and zsc_vals is not None and n20 >= 5:
            zr = n20
            fig20.add_trace(go.Scatter(
                x=zsc_ts, y=zsc_vals,
                line=dict(color='#00BCD4', width=1.5),
                name="Z-Score"), zr, 1)
            for lvl, clr in [(2,'rgba(255,80,80,0.4)'),(-2,'rgba(0,200,80,0.4)'),(0,'rgba(255,255,255,0.2)')]:
                fig20.add_hline(y=lvl,
                    line=dict(color=clr,width=1,dash='dot'), row=zr, col=1)

        fig20.update_layout(
            height=900 if n20 <= 4 else 1050,
            template="plotly_dark",
            margin=dict(l=0,r=50,t=25,b=0),
            xaxis_rangeslider_visible=False,
            showlegend=False)
        fig20.update_yaxes(side="right")
        st.plotly_chart(fig20, use_container_width=True,
            config={'scrollZoom': True})

        # ── Информационная панель
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)

        # Спот/Фьючерс базис
        try:
            import requests as _rb
            r_t = _rb.get("https://api.bybit.com/v5/market/tickers",
                params={"category":"linear","symbol":lab_sym20}, timeout=5)
            perp_px = float(r_t.json().get('result',{}).get('list',[{}])[0].get('lastPrice',0))
            spot_px = float(df20['close'].iloc[-1])
            if perp_px > 0:
                basis = (perp_px - spot_px) / spot_px * 100
                with c1:
                    st.metric("Базис Спот/Фьюч", f"{basis:+.3f}%",
                        help="+ Контанго (бычье) | − Бэквордация (медвежье)")
        except Exception:
            pass

        # ATR стоп
        atr_v, sl_l, sl_s = lab_calc_atr_stops(df20)
        if atr_v:
            with c2:
                st.metric("ATR(14)", f"{atr_v:.4f}")
            with c3:
                st.metric("SL Long",  f"{sl_l:.4f}", delta=f"-{atr_v*1.5:.4f}", delta_color="inverse")
            with c4:
                st.metric("SL Short", f"{sl_s:.4f}", delta=f"+{atr_v*1.5:.4f}")

        # Сигнал Спот/Фьюч дивергенции
        if squeeze_signal == 'LONG_SQUEEZE':
            st.success("⚡ **Long Squeeze** — Фьючерс CVD падает, Спот держится → возможный вход в лонг")
        elif squeeze_signal == 'SHORT_SQUEEZE':
            st.warning("⚡ **Short Squeeze** — Спот CVD падает, Фьючерс растёт → давление продавцов на споте")

        # Z-Score интерпретация
        if show_zsc20 and zsc_vals is not None:
            last_z = float(zsc_vals[-1]) if not np.isnan(zsc_vals[-1]) else 0
            with st.expander("📊 Z-Score BTC/ETH интерпретация"):
                st.markdown(f"""
| Z-Score | Сигнал |
|---|---|
| > +2.0 | BTC переоценён vs ETH → Продать BTC / Купить ETH |
| < -2.0 | ETH переоценён vs BTC → Продать ETH / Купить BTC |
| −2..+2 | Нейтрально, спред в норме |

**Текущий Z-Score: `{last_z:.2f}`**
                """)

    # ─── ВКЛАДКА 21: RS & HEATMAP ─────────────────────────────────────────────
    elif tab == "21. 📊 RS & Heatmap":
        st.markdown("#### 📈 Relative Strength — Альты vs BTC")
        st.caption("Показывает какие монеты сильнее/слабее BTC за выбранный период")

        rs_period = st.slider("Период RS (дней):", 7, 90, 30, key="rs_period")

        btc_df = fetch_data_bg("BTCUSDT", "1D")
        if btc_df is None or btc_df.empty:
            st.warning("Нет данных BTC")
            st.stop()

        btc_ret = float(btc_df['close'].iloc[-1]) / float(
            btc_df['close'].iloc[max(-rs_period-1,-len(btc_df))]) - 1

        rs_data = []
        symbols_rs = LIST_ALTS_MAIN[:20]  # топ-20 для скорости
        for sym in symbols_rs:
            try:
                d = fetch_data_bg(sym, "1D")
                if d is None or len(d) < 5:
                    continue
                ret = float(d['close'].iloc[-1]) / float(
                    d['close'].iloc[max(-rs_period-1,-len(d))]) - 1
                rs  = ret - btc_ret  # relative strength vs BTC
                rs_data.append({
                    'Символ': sym.replace('USDT',''),
                    'RS vs BTC': round(rs * 100, 2),
                    'Доходность %': round(ret * 100, 2),
                    'BTC %': round(btc_ret * 100, 2),
                    'Цена': round(float(d['close'].iloc[-1]), 4),
                })
            except Exception:
                continue

        if rs_data:
            rs_df = pd.DataFrame(rs_data).sort_values('RS vs BTC', ascending=False)
            # Цветовая таблица
            def color_rs(val):
                if val > 5:  return 'background-color: rgba(0,200,80,0.3)'
                if val > 0:  return 'background-color: rgba(0,200,80,0.1)'
                if val < -5: return 'background-color: rgba(220,50,50,0.3)'
                if val < 0:  return 'background-color: rgba(220,50,50,0.1)'
                return ''
            st.dataframe(
                rs_df.style.applymap(color_rs, subset=['RS vs BTC']),
                use_container_width=True, height=400)

            # График выбранной монеты
            st.markdown("#### 📊 График выбранной монеты")
            sel_sym = st.selectbox("Монета:",
                [r['Символ'] + 'USDT' for r in rs_data], key="rs_sel_sym")
            sel_df = fetch_data_bg(sel_sym, "1D")
            if sel_df is not None and not sel_df.empty:
                sel_df = apply_order_flow(sel_df)
                fvgs_sel = lab_calc_fvg(sel_df)
                obs_sel  = lab_calc_ob(sel_df)

                fig_sel = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.75, 0.25],
                    subplot_titles=(f"{sel_sym} | 1D", "CVD"))

                fig_sel.add_trace(go.Candlestick(
                    x=sel_df['timestamp'],
                    open=sel_df['open'], high=sel_df['high'],
                    low=sel_df['low'], close=sel_df['close'],
                    increasing_line_color='#00E676',
                    decreasing_line_color='#FF1744',
                    name=sel_sym), 1, 1)

                lab_add_fvg_to_fig(fig_sel, fvgs_sel, row=1)
                lab_add_ob_to_fig(fig_sel, obs_sel, row=1)

                fig_sel.add_trace(go.Scatter(
                    x=sel_df['timestamp'], y=sel_df['cvd'],
                    line=dict(color='#00E676', width=1.5),
                    name="CVD"), 2, 1)

                fig_sel.update_layout(height=600, template="plotly_dark",
                    margin=dict(l=0,r=50,t=25,b=0),
                    xaxis_rangeslider_visible=False,
                    showlegend=False)
                fig_sel.update_yaxes(side="right")
                st.plotly_chart(fig_sel, use_container_width=True,
                    config={'scrollZoom': True})
