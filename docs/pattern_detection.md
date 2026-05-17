# Pattern Detection — Implementation Reference

This document describes how each chart pattern is detected in the screener.  
All detection runs on daily OHLCV data using the cached price history.

---

## 1. VCP — Volatility Contraction Pattern

Inspired by Mark Minervini's setup criteria. A stock that ran up strongly then spent several weeks quietly coiling tighter — in both price range and volume — is considered a VCP candidate. The implementation enforces temporal ordering in the run-up and uses a tolerance band for contraction, making it more robust to the real-world price behaviour of Indian mid- and small-cap stocks.

**Default parameters**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `VCP_RUNUP_LOOKBACK` | 40 days | Window scanned for the prior run-up |
| `VCP_RUNUP_MIN_PCT` | 15 % | Minimum directional gain in the run-up window |
| `VCP_CONSOL_WEEKS` | 4 | Number of successive weeks that must contract |
| `VCP_CONTRACTION_TOLERANCE` | 15 % | Minimum tightening required week-over-week |
| `VCP_MAX_BASE_DEPTH_PCT` | 12 % | Maximum range allowed in the final (tightest) week |
| `VCP_VOL_DRY_UP_RATIO` | 0.65 | Consolidation avg vol as a fraction of run-up avg vol |

**Detection logic**

The lookback window is split into two sections:

```
[── 40-bar run-up slice ──][── 20-bar consolidation (4 × 5-day weeks) ──] ← today
```

**Step 1 — Temporally ordered run-up check**

The low is found first; only the portion of the run-up window *after* that low is used to find the high. This prevents a crash-and-recovery range from masquerading as a genuine advance.

```
low_idx  =  argmin(Low in run-up slice)
run_up   =  (max(High[low_idx:]) − Low[low_idx]) / Low[low_idx]  ≥  15 %
```

**Step 2 — Contraction across each week**

For each of the four 5-day weeks (w₁ → w₂ → w₃ → w₄):

```
price_range_pct  =  (week_High_max − week_Low_min) / week_Close_avg
```

The pattern passes all three contraction tests:

1. **Tolerance band contraction** — each week must tighten by at least 15%:  
   `week[i+1].range  ≤  week[i].range × (1 − 0.15)`

2. **Base depth** — the final week must be genuinely compressed:  
   `week[-1].range  ≤  VCP_MAX_BASE_DEPTH_PCT  (12 %)`

3. **Volume dry-up** — average consolidation volume vs average run-up volume:  
   `mean(consol_Volume)  /  mean(runup_Volume)  <  0.65`

---

## 2. Flag (Bull Flag)

A sharp, high-volume advance (the pole) followed by a brief tight consolidation on declining volume. The stock is "catching its breath" before a potential continuation. The pole is located dynamically within a scan window rather than assumed to occupy a fixed position — this captures real poles that vary from 3 to 15 bars.

**Default parameters**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `_FLAG_DAYS` | 15 | Maximum length of the flag consolidation |
| `_POLE_SCAN_WINDOW` | 30 | Bars before the flag scanned for the pole |
| `_POLE_MIN_DAYS` | 3 | Minimum pole length in bars |
| `_POLE_MAX_DAYS` | 15 | Maximum pole length in bars |
| `_BASELINE_DAYS` | 20 | Baseline period for volume comparison |
| `_MIN_POLE_GAIN` | 15 % | Minimum gain (temporally ordered trough → peak) |
| `_MIN_VOL_RATIO` | 2.0× | Pole average volume vs. baseline average volume |
| `_MAX_FLAG_RANGE` | 12 % | Maximum (High − Low) / Close over the flag |
| `_MAX_RETRACE` | 50 % | Maximum retracement of the pole move |
| `_FLAG_MAX_SLOPE` | 0.2 % / bar | Maximum upward slope of flag close prices |
| `_FLAG_BREAKOUT_ZONE` | 2/3 | Current close must be above this fraction of the flag range from its low |

**Detection logic**

The window is laid out as:

```
[── 20-bar baseline ──][── 30-bar pole-scan ──][── 15-bar flag ──] ← today
```

**Step 1 — Locate the pole** *(within the 30-bar scan window)*

```
peak_pos   =  argmax(High in scan window)
trough_pos =  argmin(Low in scan window before peak_pos)
pole_span  =  peak_pos − trough_pos   → must be in [3, 15] bars
```

**Step 2 — Validate pole quality**

```
pole_gain  =  (peak_High − trough_Low) / trough_Low  ≥  15 %   (temporal ordering enforced)
pole_avg_vol  /  baseline_avg_vol  ≥  2.0
```

**Step 3 — Validate flag**

```
(flag_High − flag_Low) / current_price  ≤  12 %
flag_Low  ≥  peak_High − 50 % × pole_move               (limited retracement)
flag_avg_vol  <  pole_avg_vol                            (volume declining)
flag_close_slope  ≤  0.2 % × price / bar                (not a rising wedge)
current_Close  ≥  flag_Low + ⅔ × flag_range             (in upper third — breakout readiness)
```

---

## 3. Volume Contraction

When recent trading volume dries up significantly below the longer-term average, the stock is quietly consolidating without active selling pressure. The detection includes a three-tier volume check and two qualitative filters that separate genuine accumulation from quiet distribution.

**Default parameters**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `VOL_SHORT_PERIOD` | 5 days | Recent volume window |
| `VOL_LONG_PERIOD` | 20 days | Tier 1 baseline |
| `VOL_EXTENDED_PERIOD` | 50 days | Tier 2 baseline (stronger confirmation) |
| `VOL_CONTRACTION_RATIO` | 0.80 | Threshold applied to both tiers |
| `VOL_PRICE_STABILITY_PCT` | 3 % | Max price decline allowed over the short window |
| `VOL_DISTRIB_DAY_PCT` | 1 % | Intraday decline threshold for a distribution day |

**Detection logic**

All conditions must hold:

**Tier 1 — Short baseline**
```
mean(Volume[-5:])  /  mean(Volume[-20:])  <  0.80
```

**Tier 2 — Extended baseline** *(applied when ≥ 50 bars available)*
```
mean(Volume[-5:])  /  mean(Volume[-50:])  <  0.80
```

**Price stability filter** — low volume + falling price = distribution, not accumulation
```
(Close[-1] − Close[-5]) / Close[-5]  ≥  −3 %
```

**Distribution day filter** — any bar in the recent window that is down >1% intraday on above-average volume signals active selling and disqualifies the signal
```
for each bar in Volume[-5:]:
    if (Close − Open) / Open  <  −1 %  AND  Volume  >  mean(Volume[-20:]):
        reject
```

---

## 4. Near 10 EMA

Flags stocks that are trading close to their 10-day EMA — a sign that the stock has pulled back to a key level without breaking down. The band is intentionally asymmetric, allowing a slightly deeper dip below the EMA than above it.

**Default parameters**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `EMA10_UPPER_PCT` | +1.5 % | Maximum price extension above EMA10 |
| `EMA10_LOWER_PCT` | −2.5 % | Maximum price distance below EMA10 |

**Detection formula**

```
deviation  =  (Close − EMA10) / EMA10

True  if  −2.5 %  ≤  deviation  ≤  +1.5 %
```

---

## 5. Inside Day (Inside Bar)

The simplest pattern in the set. Today's entire price range is contained within yesterday's range — a pause in volatility that often precedes a directional move.

**No configurable parameters.**

**Detection formula**

```
True  if  today_High  <  yesterday_High
      AND  today_Low   >  yesterday_Low
```

---

## 6. Symmetrical Triangle

Price coils inside converging trendlines — a descending upper boundary (lower highs) and an ascending lower boundary (higher lows) — forming a triangle that points toward a future breakout. The pattern is validated only when the triangle is still actively open and genuinely symmetric.

**Default parameters**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `SYMTRI_LOOKBACK` | 90 bars | History window scanned (real institutional triangles span 80–120 bars) |
| `SYMTRI_MIN_TOUCHES` | 3 | Minimum swing pivots required per trendline |
| `SYMTRI_SWING_WINDOW` | 3 bars | Bars each side a pivot must strictly dominate to qualify |
| `SYMTRI_MIN_BARS_TO_APEX` | 5 bars | Triangle must not be at the apex yet |
| `SYMTRI_MIN_SPAN_BARS` | 15 bars | Minimum span between first and last pivot |
| `SYMTRI_MIN_START_WIDTH_PCT` | 3 % | Minimum initial trendline separation as % of price |
| `SYMTRI_R2_THRESHOLD` | 0.85 | Minimum R² for each trendline fit — filters noisy pivot sets |
| `SYMTRI_SLOPE_SYMMETRY_MIN` | 0.5 | Minimum `\|m_upper\| / \|m_lower\|` ratio |
| `SYMTRI_SLOPE_SYMMETRY_MAX` | 2.0 | Maximum `\|m_upper\| / \|m_lower\|` ratio |
| `SYMTRI_REQUIRE_VOL_CONTRACTION` | True | Volume must dry up into the apex |

**Detection logic**

**Step 1 — Identify swing pivots**

A bar at index `i` is a swing high if it strictly dominates *both* the left and right windows (strict on both sides prevents double-counting pivots on flat-top formations):

```
High[i]  >  max(High[i−3 : i])       (strictly above all 3 bars to the left)
High[i]  >  max(High[i+1 : i+4])     (strictly above all 3 bars to the right)
```

Swing lows use the mirror condition on `Low`. Note: the most recent `swing_window` bars can never produce a pivot (the look-ahead window has not yet closed), so triangle signals lag by 3 bars.

**Step 2 — Fit trendlines and validate R²**

Linear regression (`numpy.polyfit`) is applied to the swing high and swing low prices separately:

```
Upper trendline:  ŷ  =  m_u · x  +  b_u     (m_u must be < 0)
Lower trendline:  ŷ  =  m_l · x  +  b_l     (m_l must be > 0)
```

R² is computed for each fit and must exceed 0.85 — this guards against noisy pivot sets that merely trend in the right direction by chance.

**Step 3 — Slope symmetry**

To distinguish a genuine symmetric triangle from a wedge or ascending/descending triangle:

```
0.5  ≤  |m_u| / |m_l|  ≤  2.0
```

**Step 4 — Validate convergence**

The apex must lie in the future and be at least `SYMTRI_MIN_BARS_TO_APEX` bars away:

```
x_apex  =  (b_l − b_u) / (m_u − m_l)   >   last_bar_index + 5
```

**Step 5 — Quality gates**

| Check | Condition |
|-------|-----------|
| Current price inside triangle | `lower_tl_value  ≤  Close  ≤  upper_tl_value` |
| Pattern genuinely converging | `width_at_current_bar  <  width_at_first_pivot` |
| Starting width meaningful | `width_at_first_pivot  ≥  3 % × Close` |
| Triangle still open | `width_at_current_bar  ≥  0.5 % × Close` |
| Pivot span sufficient | `last_pivot_bar − first_pivot_bar  ≥  15` |
| Volume contraction | `mean(Volume[second half])  <  mean(Volume[first half])` |

**Chart overlay**

When a Symmetrical Triangle is detected, the stock chart renders both trendlines as purple dashed lines with triangle-shaped pivot markers at each turning point.

---

## 7. Ascending Triangle

A flat-to-slightly-rising resistance (horizontal ceiling tested multiple times) combined with an ascending support line (progressively higher lows) — a classic bullish continuation pattern. Price coils tighter as buyers push the floor higher while sellers defend the same overhead level, until supply is exhausted and the stock breaks out.

**Default parameters**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `ASCTRI_LOOKBACK` | 90 bars | History window scanned (ideal formation: 40–90 bars) |
| `ASCTRI_MIN_TOUCHES_RESIST` | 2 | Minimum flat-top swing highs |
| `ASCTRI_MIN_TOUCHES_SUPPORT` | 2 | Minimum rising swing lows |
| `ASCTRI_SWING_WINDOW` | 3 bars | Bars each side a pivot must strictly dominate |
| `ASCTRI_MIN_SPAN_BARS` | 25 | Minimum pivot span — shorter formations are noise |
| `ASCTRI_MAX_SPAN_BARS` | 150 | Maximum pivot span |
| `ASCTRI_RESIST_MAX_SLOPE_PCT` | 0.1 % / bar | Resistance may not slope more than 0.1% of price per bar |
| `ASCTRI_SUPPORT_MIN_SLOPE_PCT` | 0.03 % / bar | Rising support must slope at least 0.03% of price per bar |
| `ASCTRI_SUPPORT_R2` | 0.65 | Minimum R² for the support trendline fit |
| `ASCTRI_VOL_END_RATIO` | 0.70 | Volume in the final bars < 70% of the opening bars |
| `ASCTRI_RSI_MIN` | 45 | RSI lower bound (not deeply oversold) |
| `ASCTRI_RSI_MAX` | 70 | RSI upper bound (not already overbought) |
| `ASCTRI_REQUIRE_ABOVE_200DMA` | True | Stock must be in an established uptrend |

**Detection logic**

**Step 1 — Identify swing pivots** *(strict both sides, swing_window = 3)*

Swing highs form the resistance line; swing lows form the support line.

**Step 2 — Fit and validate trendlines**

```
Resistance:  slope near-flat  — |r_slope| / price  <  0.001 per bar  AND  r_slope ≥ 0
Support:     slope positive   — s_slope / price     >  0.0003 per bar
             R² of support fit  ≥  0.65
```

**Step 3 — Convergence**

```
width_first  =  resist(first_pivot_bar) − support(first_pivot_bar)  >  0
width_last   =  resist(current_bar)     − support(current_bar)
width_last   <  width_first                (lines converging)
width_last   ≥  0.5 % × Close             (pattern still open)
```

**Step 4 — Context filters**

| Filter | Condition |
|--------|-----------|
| Price inside triangle | `support_now  ≤  Close  ≤  resist_now` |
| Trend context | Price above 200-day EMA *(when column available)* |
| RSI | 45 ≤ RSI ≤ 70 *(when column available)* |

**Step 5 — Volume and OBV filters**

| Check | Condition |
|-------|-----------|
| Volume slope | Linear regression on volume over the formation → slope < 0 |
| Volume end/start | `mean(Vol[-10:])  /  mean(Vol[:10])  <  0.70` |
| OBV trend | OBV slope / avg\_daily\_vol  ≥  −0.30 (not in heavy distribution) |

**Chart overlay**

Resistance line renders in **orange** (dashed), support line in **green** (dashed), with matching pivot markers.

---

*Patterns are re-computed on every screener run. Configurable parameters can be adjusted in the Admin → Pattern Detection tab and saved to `config.py`.*
