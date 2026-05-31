> 💡 **Developer's Note:** Below is a real output from my validation engine testing a baseline "Balance Zone Retest" strategy. 
> Notice how the engine correctly identified that this specific logic lacks a statistical edge (edge_gap ≤ 0) and rigorously rejected it for live trading (🔴 FAIL). This brutal, math-based filtering (including Monte Carlo and Walk-Forward tests) is exactly what prevents curve-fitting and saves traders from losing money on raw ideas.
---

# Phase 0 Report — bt_balance_zone_retest_Bar.py

**Дата:** 2026-05-31 16:33  
**Время выполнения:** 18.8 мин  
**Вердикт:** ⚠️ **UNKNOWN**

## Параметры запуска

| Параметр | Значение |
|---|---|
| Скрипт | `D:\MyScreener\bt_balance_zone_retest_Bar.py` |
| Parquet | `D:\MyScreener\backtest_data\bt_balance_zone_retest_raw.parquet` |
| RR | 3.0 |
| Workers | 15 |
| MC итераций | 200 |
| skip-backtest | False |
| include-ar | True |
| tail-focus | True |

## Ключевые метрики

| Метрика | Значение |
|---|---|
| N сигналов IS | 11,002 |
| avg_R (IS) | -0.1616% |
| edge_gap (vs random) | -0.0670% |

---

========================================================================
  Phase 0 Orchestrator — bt_balance_zone_retest_Bar.py
========================================================================
  Скрипт:    D:\MyScreener\bt_balance_zone_retest_Bar.py
  Parquet:   D:\MyScreener\backtest_data\bt_balance_zone_retest_raw.parquet
  RR:        3.0  |  Workers: 15  |  Iter: 200

## Шаг 1 — Бэктест

**Команда:** `python bt_balance_zone_retest_Bar.py --reset --workers 15`

```
[OK] Known-answer tests passed
[OK] Smoke test: 1 signals на synthetic
[OK] Leakage test
[OK] ML integration: MTF continuation — входы VAH/VAL + HVN, SL=VAL/VAH, TP 1:3
[OK] No-breakout test: цена внутри зоны → 0 retest сигналов
[OK] Train exclusion test
[OK] Fail-fast test
16:33:39 [INFO] [Balance Zone Retest] Задач: 94 | Workers: 15
16:35:37 [INFO] ✅ XRPUSDT_15m: 430 | Всего: 430
16:35:38 [INFO] ✅ AAVEUSDT_15m: 479 | Всего: 909
16:35:39 [INFO] ✅ BCHUSDT_15m: 426 | Всего: 1335
16:35:43 [INFO] ✅ TRXUSDT_15m: 490 | Всего: 1825
16:36:12 [INFO] ✅ BNBUSDT_15m: 451 | Всего: 2276
16:36:12 [INFO] Flush → 2276 записей (bt_balance_zone_retest_raw.parquet)
16:36:27 [INFO] ✅ NEARUSDT_15m: 501 | Всего: 2777
16:36:34 [INFO] ✅ XLMUSDT_15m: 485 | Всего: 3262
16:36:41 [INFO] ✅ ATOMUSDT_15m: 514 | Всего: 3776
16:36:48 [INFO] ✅ POLUSDT_15m: 190 | Всего: 3966
16:37:09 [INFO] ✅ AVAXUSDT_15m: 460 | Всего: 4426
16:37:10 [INFO] Flush → 4426 записей (bt_balance_zone_retest_raw.parquet)
16:37:12 [INFO] ✅ BTCUSDT_15m: 398 | Всего: 4824
16:37:22 [INFO] ✅ ETCUSDT_15m: 472 | Всего: 5296
16:37:26 [INFO] ✅ SOLUSDT_15m: 423 | Всего: 5719
16:37:50 [INFO] ✅ ADAUSDT_15m: 469 | Всего: 6188
16:37:50 [INFO] ✅ DOGEUSDT_15m: 427 | Всего: 6615
16:37:51 [INFO] Flush → 6615 записей (bt_balance_zone_retest_raw.parquet)
16:37:51 [INFO] ✅ LINKUSDT_15m: 487 | Всего: 7102
16:37:54 [INFO] ✅ LTCUSDT_15m: 477 | Всего: 7579
16:38:12 [INFO] ✅ ETHUSDT_15m: 419 | Всего: 7998
16:38:20 [INFO] ✅ ARBUSDT_15m: 352 | Всего: 8350
16:38:24 [INFO] ✅ ZILUSDT_15m: 254 | Всего: 8604
16:38:24 [INFO] Flush → 8604 записей (bt_balance_zone_retest_raw.parquet)
16:38:25 [INFO] ✅ CAKEUSDT_15m: 244 | Всего: 8848
16:38:28 [INFO] ✅ UNIUSDT_15m: 483 | Всего: 9331
16:38:45 [INFO] ✅ FILUSDT_15m: 395 | Всего: 9726
16:38:50 [INFO] ✅ DOTUSDT_15m: 437 | Всего: 10163
16:38:52 [INFO] ✅ OPUSDT_15m: 416 | Всего: 10579
16:38:53 [INFO] Flush → 10579 записей (bt_balance_zone_retest_raw.parquet)
16:39:24 [INFO] ✅ WOOUSDT_15m: 474 | Всего: 11053
16:39:32 [INFO] ✅ STXUSDT_15m: 436 | Всего: 11489
16:39:57 [INFO] ✅ XTZUSDT_15m: 490 | Всего: 11979
16:40:08 [INFO] ✅ PEOPLEUSDT_15m: 405 | Всего: 12384
16:40:24 [INFO] ✅ GALAUSDT_15m: 408 | Всего: 12792
16:40:25 [INFO] Flush → 12792 записей (bt_balance_zone_retest_raw.parquet)
16:40:56 [INFO] ✅ CRVUSDT_15m: 494 | Всего: 13286
16:41:04 [INFO] ✅ ICPUSDT_15m: 485 | Всего: 13771
16:41:48 [INFO] ✅ TONUSDT_15m: 267 | Всего: 14038
16:42:07 [INFO] ✅ ORDIUSDT_15m: 288 | Всего: 14326
16:42:14 [INFO] ✅ SEIUSDT_15m: 292 | Всего: 14618
16:42:15 [INFO] Flush → 14618 записей (bt_balance_zone_retest_raw.parquet)
16:42:17 [INFO] ✅ VETUSDT_15m: 469 | Всего: 15087
16:43:17 [INFO] ✅ INJUSDT_15m: 376 | Всего: 15463
16:43:24 [INFO] ✅ THETAUSDT_15m: 477 | Всего: 15940
16:43:31 [INFO] ✅ ZRXUSDT_15m: 500 | Всего: 16440
16:43:40 [INFO] ✅ JASMYUSDT_15m: 424 | Всего: 16864
16:43:40 [INFO] Flush → 16864 записей (bt_balance_zone_retest_raw.parquet)
16:43:48 [INFO] ✅ WAVESUSDT_15m: 379 | Всего: 17243
16:44:51 [INFO] ✅ YGGUSDT_15m: 426 | Всего: 17669
16:44:56 [INFO] ✅ SUIUSDT_15m: 317 | Всего: 17986
16:45:14 [INFO] ✅ KSMUSDT_15m: 426 | Всего: 18412
16:45:25 [INFO] ✅ RUNEUSDT_15m: 507 | Всего: 18919
16:45:26 [INFO] Flush → 18919 записей (bt_balance_zone_retest_raw.parquet)
16:45:26 [INFO] ✅ ANKRUSDT_15m: 456 | Всего: 19375
16:45:56 [INFO] ✅ GASUSDT_15m: 234 | Всего: 19609
16:46:07 [INFO] ✅ APTUSDT_15m: 397 | Всего: 20006
16:46:18 [INFO] ✅ ASTRUSDT_15m: 435 | Всего: 20441
16:46:33 [INFO] ✅ MASKUSDT_15m: 508 | Всего: 20949
16:46:35 [INFO] Flush → 20949 записей (bt_balance_zone_retest_raw.parquet)
16:46:35 [INFO] ✅ RVNUSDT_15m: 468 | Всего: 21417
16:46:53 [INFO] ✅ APEUSDT_15m: 428 | Всего: 21845
16:47:10 [INFO] ✅ ONTUSDT_15m: 445 | Всего: 22290
16:47:16 [INFO] ✅ BOMEUSDT_15m: 250 | Всего: 22540
16:47:53 [INFO] ✅ GLMUSDT_15m: 285 | Всего: 22825
16:47:55 [INFO] Flush → 22825 записей (bt_balance_zone_retest_raw.parquet)
16:47:56 [INFO] ✅ BANDUSDT_15m: 461 | Всего: 23286
16:47:58 [INFO] ✅ 1000PEPEUSDT_15m: 314 | Всего: 23600
16:48:16 [INFO] ✅ HBARUSDT_15m: 404 | Всего: 24004
16:48:16 [INFO] ✅ MANAUSDT_15m: 428 | Всего: 24432
16:48:30 [INFO] ✅ WIFUSDT_15m: 274 | Всего: 24706
16:48:31 [INFO] Flush → 24706 записей (bt_balance_zone_retest_raw.parquet)
16:48:44 [INFO] ✅ ARUSDT_15m: 488 | Всего: 25194
16:48:52 [INFO] ✅ CKBUSDT_15m: 423 | Всего: 25617
16:48:53 [INFO] ✅ BLURUSDT_15m: 352 | Всего: 25969
16:49:00 [INFO] ✅ GMTUSDT_15m: 345 | Всего: 26314
16:49:17 [INFO] ✅ POPCATUSDT_15m: 235 | Всего: 26549
16:49:18 [INFO] Flush → 26549 записей (bt_balance_zone_retest_raw.parquet)
16:49:18 [INFO] ✅ MUBARAKUSDT_15m: 129 | Всего: 26678
16:49:30 [INFO] ✅ 1000BONKUSDT_15m: 329 | Всего: 27007
16:49:36 [INFO] ✅ SANDUSDT_15m: 466 | Всего: 27473
16:49:43 [INFO] ✅ AGIUSDT_15m: 164 | Всего: 27637
16:50:06 [INFO] ✅ KAVAUSDT_15m: 443 | Всего: 28080
16:50:07 [INFO] Flush → 28080 записей (bt_balance_zone_retest_raw.parquet)
16:50:24 [INFO] ✅ SKLUSDT_15m: 452 | Всего: 28532
16:50:28 [INFO] ✅ PENGUUSDT_15m: 162 | Всего: 28694
16:50:28 [INFO] ✅ XVSUSDT_15m: 292 | Всего: 28986
16:50:31 [INFO] ✅ IMXUSDT_15m: 41 | Всего: 29027
16:50:46 [INFO] ✅ MEMEUSDT_15m: 266 | Всего: 29293
16:50:47 [INFO] Flush → 29293 записей (bt_balance_zone_retest_raw.parquet)
16:50:49 [INFO] ✅ GBPUSD_15m: 238 | Всего: 29531
16:51:05 [INFO] ✅ EURUSD_15m: 231 | Всего: 29762
16:51:14 [INFO] ✅ SUSHIUSDT_15m: 471 | Всего: 30233
16:51:17 [INFO] ✅ USDJPY_15m: 271 | Всего: 30504
16:51:20 [INFO] ✅ ZETAUSDT_15m: 272 | Всего: 30776
16:51:22 [INFO] Flush → 30776 записей (bt_balance_zone_retest_raw.parquet)
16:51:23 [INFO] ✅ AUDUSD_15m: 245 | Всего: 31021
16:51:25 [INFO] ✅ LRCUSDT_15m: 451 | Всего: 31472
16:51:30 [INFO] ✅ COTIUSDT_15m: 508 | Всего: 31980
16:51:37 [INFO] ✅ MAGICUSDT_15m: 374 | Всего: 32354
16:51:42 [INFO] ✅ DYDXUSDT_15m: 496 | Всего: 32850
16:51:43 [INFO] Flush → 32850 записей (bt_balance_zone_retest_raw.parquet)
16:51:45 [INFO] ✅ EURGBP_15m: 228 | Всего: 33078
16:51:50 [INFO] ✅ USDCAD_15m: 239 | Всего: 33317
16:51:50 [INFO] ✅ EURJPY_15m: 240 | Всего: 33557
16:51:52 [INFO] ✅ NZDUSD_15m: 251 | Всего: 33808
16:51:54 [INFO] ✅ USDCHF_15m: 218 | Всего: 34026
16:51:54 [INFO] Flush → 34026 записей (bt_balance_zone_retest_raw.parquet)
16:52:00 [INFO] ✅ GBPJPY_15m: 245 | Всего: 34271
16:52:03 [INFO] ✅ ENSUSDT_15m: 509 | Всего: 34780
16:52:03 [INFO] ✅ XAUUSD_15m: 207 | Всего: 34987
16:52:08 [INFO] ✅ AUDJPY_15m: 281 | Всего: 35268
16:52:09 [INFO] Flush → 35268 записей (bt_balance_zone_retest_raw.parquet)
16:52:09 [INFO] IS=11002, VAL=4519, OOS=19622
16:52:10 [INFO] 
========================================================================
16:52:10 [INFO] РЕЗУЛЬТАТЫ — Balance Zone Retest
16:52:10 [INFO] ========================================================================
16:52:10 [INFO] 
  ── IS (11002 сигналов) ──
16:52:10 [INFO]     AR_P50=0.566  AR_P90=1.504  edge=✅
16:52:10 [INFO]     RR1:1.5  WR=13.1%  total=-2233.1%
16:52:10 [INFO]     RR1:2.0  WR=8.8%  total=-1989.5%
16:52:10 [INFO]     RR1:3.0  WR=4.2%  total=-1777.9%
16:52:10 [INFO]     RR1:4.0  WR=2.0%  total=-1731.7%
16:52:10 [INFO]     RR1:5.0  WR=1.2%  total=-1695.2%
16:52:10 [INFO] 
  ── VAL (4519 сигналов) ──
16:52:10 [INFO]     AR_P50=0.756  AR_P90=1.713  edge=✅
16:52:10 [INFO]     RR1:1.5  WR=14.9%  total=-656.7%
16:52:10 [INFO]     RR1:2.0  WR=10.3%  total=-544.5%
16:52:10 [INFO]     RR1:3.0  WR=5.1%  total=-443.3%
16:52:10 [INFO]     RR1:4.0  WR=2.6%  total=-423.6%
16:52:10 [INFO]     RR1:5.0  WR=1.5%  total=-400.3%
16:52:10 [INFO] 
  ── OOS (19622 сигналов) ──
16:52:10 [INFO]     AR_P50=0.673  AR_P90=1.603  edge=✅
16:52:10 [INFO]     RR1:1.5  WR=13.1%  total=-3482.1%
16:52:10 [INFO]     RR1:2.0  WR=8.8%  total=-3063.1%
16:52:10 [INFO]     RR1:3.0  WR=3.8%  total=-2821.7%
16:52:10 [INFO]     RR1:4.0  WR=1.8%  total=-2715.2%
16:52:10 [INFO]     RR1:5.0  WR=1.0%  total=-2682.3%
16:52:10 [INFO] 
  Walk-Forward  : score=0.8575  passed=True
16:52:10 [INFO]   Monte Carlo   : real_AR=0.563  p95=1.0496  passed=False
16:52:10 [INFO] ========================================================================

16:52:10 [INFO] 
16:52:10 [INFO] ════════════════════════════════════════════════════════════════════════
16:52:10 [INFO] LIVE TRADING READINESS — Balance Zone Retest
16:52:10 [INFO] ════════════════════════════════════════════════════════════════════════
16:52:10 [INFO] 
16:52:10 [INFO]   ── СТАТИСТИКА ──
16:52:10 [INFO]   ✅ WF Score ≥ 0.80           : 0.8575  PASS
16:52:10 [INFO]   ✅ OOS WR ≥ IS WR × 0.85     : OOS 8.8% ≥ IS 8.8%×0.85=7.5%  PASS
16:52:10 [INFO]   ❌ MC p95 < real_AR           : real=0.5630  p95=1.0496  FAIL
16:52:10 [INFO]   ❌ IS/VAL/OOS прибыльны       : IS=+-1989.5%  VAL=+-544.5%  OOS=+-3063.1%
16:52:10 [INFO]   ❌ MaxDD OOS < 30%            : 3088.9%  FAIL
16:52:10 [INFO]   ❌ PF OOS > 1.3               : 0.625  FAIL
16:52:10 [INFO]   ❌ 250+ сигналов/год/символ   : 9/год  макс 0.3/день
16:52:10 [INFO] 
16:52:10 [INFO]   ── ЦЕЛОСТНОСТЬ ДАННЫХ ──
16:52:10 [INFO]   ⚪ Look-ahead (AST + runtime)   : проверяется в Phase 3 (strategy_validator.py)
16:52:10 [INFO]   ✅ C34 дедупликация сигналов  : dedup_signals() активна в run_forward_walk
16:52:10 [INFO]   ❌ Listing bias = 0             : WARN — ['AAVEUSDT', 'ADAUSDT', 'ANKRUSDT', 'ARUSDT', 'ATOMUSDT', 'AUDJPY', 'AUDUSD', 'AVAXUSDT', 'BANDUSDT', 'BCHUSDT', 'BNBUSDT', 'BTCUSDT', 'CKBUSDT', 'COTIUSDT', 'CRVUSDT', 'DOGEUSDT', 'DOTUSDT', 'DYDXUSDT', 'ENSUSDT', 'ETCUSDT', 'ETHUSDT', 'EURGBP', 'EURJPY', 'EURUSD', 'FILUSDT', 'GALAUSDT', 'GBPJPY', 'GBPUSD', 'ICPUSDT', 'JASMYUSDT', 'KAVAUSDT', 'KSMUSDT', 'LINKUSDT', 'LRCUSDT', 'LTCUSDT', 'MANAUSDT', 'MASKUSDT', 'NEARUSDT', 'NZDUSD', 'PEOPLEUSDT', 'RUNEUSDT', 'RVNUSDT', 'SANDUSDT', 'SOLUSDT', 'STXUSDT', 'SUSHIUSDT', 'THETAUSDT', 'TRXUSDT', 'UNIUSDT', 'USDCAD', 'USDCHF', 'USDJPY', 'VETUSDT', 'WAVESUSDT', 'WOOUSDT', 'XAUUSD', 'XLMUSDT', 'XRPUSDT', 'XTZUSDT', 'YGGUSDT']
16:52:10 [INFO]   ⚪ Known-answer формулы       : не передан → define _test_known_answer()
16:52:10 [INFO] 
16:52:10 [INFO]   ── РИСКИ РЕАЛЬНОЙ ТОРГОВЛИ ──
16:52:10 [INFO]   ✅ Cost model (comm+slip+spr)   : calc_fixed_rr_outcomes применяет costs
16:52:10 [INFO]   ✅ Equity bankruptcy check       : equity_total() останавливает при ≤0%
16:52:10 [INFO]   ✅ SL ≥ min_sl_dist             : get_min_sl_dist() доступна в bt_base
16:52:10 [INFO]   ✅ Сигналов ≤ 50/день/символ : 0.3/день  PASS
16:52:10 [INFO] 
16:52:10 [INFO]   ── ВОСПРОИЗВОДИМОСТЬ ──
16:52:10 [INFO]   ✅ Фиксированный seed           : seed=42 в run_monte_carlo
16:52:10 [INFO]   ⚪ Noise injection CV < 0.3    : проверяется в Phase 3 (strategy_validator.py, T8.x)
16:52:10 [INFO] 
16:52:10 [INFO]   ── PHASE 0 DIAGNOSTICS (raw edge) ──
16:52:10 [INFO]   N total: 35,268  ·  IS: 11,002  ·  VAL: 4,519  ·  OOS: 19,622
16:52:10 [INFO]   LONG: 16,695  ·  SHORT: 18,573
16:52:10 [INFO] 
16:52:10 [INFO]   split         N     avg exit_r     median        std
16:52:10 [INFO]   IS       11,002        -0.0627    +0.0000     1.0782
16:52:10 [INFO]   VAL       4,519        -0.0024    +0.0000     1.1045
16:52:10 [INFO]   OOS      19,622        -0.0354    +0.0000     1.0852
16:52:10 [INFO] 
16:52:10 [INFO]   split      AR_P50     AR_P75     AR_P90
16:52:10 [INFO]   IS         0.7105     2.9435     9.2996
16:52:10 [INFO]   OOS        0.8182     3.1487     9.9430
16:52:10 [INFO] 
16:52:10 [INFO]   SL hit% IS: 34.38%
16:52:10 [INFO]   SL hit% OOS: 31.88%
16:52:10 [INFO] 
16:52:10 [INFO]   WR @ RR (IS / OOS — WR%, edge vs BE pp, avg PnL%):
16:52:10 [INFO]   RR        BE    IS WR   IS edge     IS pnl   OOS WR  OOS edge    OOS pnl
16:52:10 [INFO]   1.5   40.00%   27.60%    -12.40    -0.2030   29.16%    -10.84    -0.1775
16:52:10 [INFO]   2.0   33.33%   20.43%    -12.90    -0.1808   21.55%    -11.78    -0.1561
16:52:10 [INFO]   3.0   25.00%   10.97%    -14.03    -0.1616   10.63%    -14.37    -0.1438
16:52:10 [INFO]   4.0   20.00%    5.59%    -14.41    -0.1574    5.49%    -14.51    -0.1384
16:52:10 [INFO]   5.0   16.67%    3.37%    -13.30    -0.1541    2.95%    -13.72    -0.1367
16:52:10 [INFO] 
16:52:10 [INFO]   ── TF BREAKDOWN (IS) ──
16:52:10 [INFO]   TF         N IS     avg_R IS    N OOS    avg_R OOS
16:52:10 [INFO]   15m      11,002      -0.0627   19,622      -0.0354
16:52:10 [INFO] 
16:52:10 [INFO]   ── DIRECTION BREAKDOWN ──
16:52:10 [INFO]   side        N IS     avg_R IS    N OOS    avg_R OOS
16:52:10 [INFO]   LONG       5,247      -0.0610    8,975      -0.0759
16:52:10 [INFO]   SHORT      5,755      -0.0642   10,647      -0.0012
16:52:10 [INFO] 
16:52:10 [INFO]   ── TOP-10 SYMBOLS by N (IS) ──
16:52:10 [INFO]   symbol              N      avg_R
16:52:10 [INFO]   TRXUSDT           238    -0.0910
16:52:10 [INFO]   ICPUSDT           211    +0.0315
16:52:10 [INFO]   ARUSDT            202    -0.1421
16:52:10 [INFO]   COTIUSDT          200    -0.0587
16:52:10 [INFO]   DYDXUSDT          198    -0.0213
16:52:10 [INFO]   ENSUSDT           196    -0.0388
16:52:10 [INFO]   RUNEUSDT          195    -0.2546
16:52:10 [INFO]   NEARUSDT          193    -0.1648
16:52:10 [INFO]   ATOMUSDT          185    -0.1136
16:52:10 [INFO]   WOOUSDT           185    -0.0183
16:52:10 [INFO] 
16:52:10 [INFO] ────────────────────────────────────────────────────────────────────────
16:52:10 [INFO]   🔴 НЕ ГОТОВ К ЛАЙВ ТОРГОВЛЕ  (1/5 критических — PASS)
16:52:10 [INFO] ════════════════════════════════════════════════════════════════════════
16:52:10 [INFO] 
16:52:10 [INFO] [✓] Readiness report → B0_readiness_balance_zone_retest_20260531_165210.txt
```

## Шаг 2 — B1 Baseline (нулевая точка стратегии)

**Команда:** `python b1_baseline_report.py backtest_data/bt_balance_zone_retest_raw.parquet --split IS`

```
==========================================================
  B1 BASELINE REPORT — нулевая точка стратегии
==========================================================
  Стратегия : Balance Zone Retest
  Файл      : bt_balance_zone_retest_raw.parquet
  Дата      : 31.05.2026 16:52
  Сплит     : IS  |  RR: 3.0
==========================================================

  ОБЩАЯ СТАТИСТИКА
──────────────────────────────────────────────────────────
  N сделок              : 11002
  N символов            : 81
  Период                : 2022–2023
  WR (decisive)         : 11.0%
  avg_R                 : -0.1616%
  median_R              : -0.0258%
  Profit Factor         : 0.32
  Wins / Losses         : 466 / 3782

  ПО ГОДАМ
──────────────────────────────────────────────────────────
  Год         N       WR      avg_R   median_R     PF
  ────── ────── ──────── ────────── ────────── ──────
  2022     5847    11.9%   -0.1386%   -0.0010%   0.35
  2023     5155     9.9%   -0.1877%   -0.0486%   0.28

  ПО СИМВОЛАМ — ТОП 15 (лучшие avg_R, N≥5)
──────────────────────────────────────────────────────────
  Символ               N       WR      avg_R     PF
  ──────────────── ───── ──────── ────────── ──────
  TONUSDT              5   100.0%   +1.7003%    N/A
  XVSUSDT              6     0.0%   +0.3791%    N/A
  EURJPY              86    20.0%   +0.1340%   0.73
  ASTRUSDT           134    20.5%   +0.0933%   0.66
  GMTUSDT             63    26.9%   +0.0593%   0.94
  NZDUSD              94    16.7%   +0.0466%   0.59
  PEOPLEUSDT         163    19.0%   -0.0133%   0.60
  GALAUSDT           154    19.2%   -0.0213%   0.61
  ADAUSDT            181    20.9%   -0.0221%   0.67
  JASMYUSDT          177    18.6%   -0.0239%   0.59
  SOLUSDT            161    18.8%   -0.0303%   0.59
  FILUSDT            169    19.6%   -0.0322%   0.63
  AUDJPY             117    13.7%   -0.0341%   0.47
  YGGUSDT            152    18.6%   -0.0364%   0.59
  BLURUSDT            58    14.3%   -0.0466%   0.43

  ПО СИМВОЛАМ — АУТСАЙДЕРЫ 15 (худшие avg_R, N≥5)
──────────────────────────────────────────────────────────
  Символ               N       WR      avg_R     PF
  ──────────────── ───── ──────── ────────── ──────
  APEUSDT            141     7.0%   -0.2505%   0.20
  ONTUSDT            148     8.5%   -0.2568%   0.24
  ETCUSDT            160     8.5%   -0.2622%   0.24
  XLMUSDT            180     8.4%   -0.2626%   0.24
  XRPUSDT            178     8.9%   -0.2716%   0.25
  AVAXUSDT           166    11.0%   -0.2749%   0.32
  ARBUSDT             48     5.0%   -0.2986%   0.13
  USDJPY              78     2.9%   -0.3046%   0.09
  BTCUSDT            177     5.6%   -0.3048%   0.15
  CRVUSDT            169     5.6%   -0.3062%   0.15
  HBARUSDT            73     3.6%   -0.3067%   0.09
  GLMUSDT             14     0.0%   -0.3153%   0.00
  RUNEUSDT           195     5.9%   -0.3434%   0.16
  SUIUSDT             37     0.0%   -0.3445%   0.00
  LTCUSDT            159     7.0%   -0.3587%   0.19

  РАСПРЕДЕЛЕНИЕ P&L  [rr30_pnl_pct]
──────────────────────────────────────────────────────────
     -1.18 …  -0.90%  ███████████████████████████████████  3815
     -0.90 …  -0.63%  ██  244
     -0.63 …  -0.35%  ████  535
     -0.35 …  -0.07%  ███████  778
     -0.07 …  +0.21%  ██████████████████████████████  3305
     +0.21 …  +0.49%  █████  562
     +0.49 …  +0.76%  ███  416
     +0.76 …  +1.04%  ██  279
     +1.04 …  +1.32%  ██  226
     +1.32 …  +1.60%  █  145
     +1.60 …  +1.87%  █  110
     +1.87 …  +2.15%    75
     +2.15 …  +2.43%    31
     +2.43 …  +2.71%    12
     +2.71 …  +2.98%  ████  469

  ОБЪЁМ СИГНАЛОВ — СРАВНЕНИЕ С ЭТАЛОНОМ ДЯДИ МИШИ
──────────────────────────────────────────────────────────
  Всего (IS+VAL+OOS)          : 35,268
    ├── IS                    : 11,002
    ├── VAL                   : 4,519
    └── OOS                   : 19,622
  Плотность OOS (сд/год/символ): 104.4
  Плотность IS  (сд/год/символ): 52.0

  Эталон Uncle Mike (15m × 100 sym × 2 года):
    Raw (Phase 0, без фильтров): 105,694
    Filtered (edge ≥ +0.013)  :  42,786  (40.5% от raw)
    Production                :  27,583  (26.1% от raw)
    Production avg_R          : +0.1756% | PF=1.51 | MDD=31.4%

  Наш OOS avg_R               : -0.1616%  (мин. планка: +0.1756%)
  Наш OOS PF                  : 0.32  (мин. планка: 1.51)

==========================================================
  ЭТО НУЛЕВАЯ ТОЧКА СТРАТЕГИИ
  Всё что добавляется в Фазе C ОБЯЗАНО улучшать avg_R.
  Если фича ухудшает avg_R — она идёт в мусор.
==========================================================

  Отчёт сохранён : D:\MyScreener\Отчёты\B1_Baselines\B1_bt_balance_zone_retest_raw_IS_20260531_165210.txt
  JSON (Фаза C)  : D:\MyScreener\Отчёты\B1_Baselines\B1_bt_balance_zone_retest_raw_IS_20260531_165210.json

```

## Шаг 3 — A3 Random Baseline (Monte Carlo)

**Команда:** `python a3_random_baseline.py backtest_data/bt_balance_zone_retest_raw.parquet --split IS --rr 3.0 --iter 200`

```

Стратегия : Balance Zone Retest
Файл      : bt_balance_zone_retest_raw.parquet
Сплит     : IS  |  Сделок: 11002  |  Символов: 81  |  TF: 15m
RR        : 3.0  |  Итераций: 200
avg_R стратегии [IS]: -0.1616%

Загружаю OHLC (15m) для 81 символов...
  [WARN] OHLC не найдено для 12 символов: ['GBPUSD', 'EURUSD', 'USDJPY', 'AUDUSD', 'EURGBP']...
  Загружено: 69 символов
  Символов с валидными барами: 49

Запуск Monte Carlo (200 итераций)...
  [ 50/200]  avg_R_random (p50 so far): -0.0969%
  [100/200]  avg_R_random (p50 so far): -0.0943%
  [150/200]  avg_R_random (p50 so far): -0.0930%
  [200/200]  avg_R_random (p50 so far): -0.0946%

================================================================
  РЕЗУЛЬТАТ A3 — Random Entry Baseline
================================================================
  Стратегия      : Balance Zone Retest
  Файл           : bt_balance_zone_retest_raw.parquet
  Дата           : 31.05.2026 16:52
  Сплит          : IS  |  RR: 3.0  |  Итераций: 200
  Сделок (сплит) : 11002  |  Символов: 81  |  Символов в MC: 49

  ── ОБЩИЙ РЕЗУЛЬТАТ ──
────────────────────────────────────────────────────────────────
  avg_R стратегии [IS]:        -0.1616%
  avg_R random   [mean ± std]:    -0.0944% ± 0.0157%
  avg_R random   [95% CI]:        [-0.1259%, -0.0671%]
  avg_R random   [p05/p50/p95]:   -0.1214% / -0.0946% / -0.0689%
────────────────────────────────────────────────────────────────
  edge_gap (стратегия − random):  -0.0670%
  % итераций где стратегия > random: 0.0%  (устойчивость edge)

  ── ПО ГОДАМ (стратегия vs общий random p50) ──
────────────────────────────────────────────────────────────────
  Год              N         avg_R     edge_gap*
  ────── ──────── ────────────── ──────────────
  2022          5847      -0.1386%      -0.0441%
  2023          5155      -0.1877%      -0.0931%

  ── ПО НАПРАВЛЕНИЮ ──
────────────────────────────────────────────────────────────────
  Direction            N         avg_R     edge_gap*
  ────────── ──────── ────────────── ──────────────
  LONG              5247      -0.1612%      -0.0667%
  SHORT             5755      -0.1619%      -0.0674%

  * edge_gap_approx = avg_R(subset) − random_p50_overall
    (точный per-subset random требует отдельной MC симуляции)

================================================================
  ❌  edge_gap ≤ 0 — гипотеза неверна → берёшь другую
================================================================

  Отчёт сохранён : D:\MyScreener\Отчёты\A3_RandomBaseline\A3_bt_balance_zone_retest_raw_IS_20260531_165222.txt
  JSON           : D:\MyScreener\Отчёты\A3_RandomBaseline\A3_bt_balance_zone_retest_raw_IS_20260531_165222.json

```

## Шаг 4 — A4 Asymmetry Ratio (AR + RR sweep)

**Команда:** `python a4_asymmetry_ratio.py backtest_data/bt_balance_zone_retest_raw.parquet --split IS --tail-focus`

```

========================================================================
  A4 — ASYMMETRY RATIO + ANALYTICAL RR SWEEP
========================================================================
  Стратегия      : Balance Zone Retest
  Файл           : bt_balance_zone_retest_raw.parquet
  Дата           : 31.05.2026 16:52
  Сплит          : IS  |  Сделок: 11002

  ── ASYMMETRY RATIO ──
────────────────────────────────────────────────────────────────────────
  AR_P50 = MFE_P50/|MAE_P50|: 0.5630   (random p50: 0.5630 ± 0.0000, z=0.00)
  AR_P75 = MFE_P75/|MAE_P75|: 0.8702
  AR_P90 = MFE_P90/|MAE_P90|: 1.5383   (random p90: 1.5383 ± 0.0000, z=1.00)

  Главная метрика (AR_P90 (tail focus)): 1.5383  vs порог 1.3  |  z=1.00

  ── РАСПРЕДЕЛЕНИЕ MAE / MFE (в R) ──
────────────────────────────────────────────────────────────────────────
  |MAE| P50:  0.5446    |  MFE P50:  0.3066
  |MAE| P90:  1.2739    |  MFE P75:  0.9199
                              |  MFE P90:  1.9597

  ── АНАЛИТИЧЕСКИЙ SWEEP RR ──
────────────────────────────────────────────────────────────────────────
   RR     N  wins  losses  timeouts  WR_pct  avg_R_pct    PF
  1.0 11002  2539    3368      5095   42.98    -7.4846 0.754
  1.5 11002  1623    3602      5777   31.06    -6.9980 0.676
  2.0 11002  1066    3687      6249   22.43    -6.2867 0.578
  2.5 11002   714    3736      6552   16.04    -6.1730 0.478
  3.0 11002   489    3759      6754   11.51    -6.1181 0.390
  4.0 11002   232    3775      6995    5.79    -6.1706 0.246
  5.0 11002   136    3779      7087    3.47    -5.9837 0.180

  Лучший RR (по avg_R): 5.0  → avg_R = -5.9837%

========================================================================
  ❌  Нет статистически значимого edge
========================================================================

  Отчёт: D:\MyScreener\Отчёты\A4_AsymmetryRatio\A4_bt_balance_zone_retest_raw_IS_20260531_165222.txt
  JSON:  D:\MyScreener\Отчёты\A4_AsymmetryRatio\A4_bt_balance_zone_retest_raw_IS_20260531_165222.json
```

========================================================================
  PHASE 0 ИТОГ — bt_balance_zone_retest_Bar.py
========================================================================
  Время:          18.8 мин
  N сигналов IS:  11,002
  avg_R:          -0.1616%
  edge_gap:       -0.0670%

  🔴 FAIL — edge_gap = -0.0670% ≤ 0
