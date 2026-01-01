import pandas as pd
import numpy as np
from datetime import time, timedelta, datetime

class EagleV25_Sovereign:
    """
    EAGLE V25.0 — THE SOVEREIGN EDITION (FINAL PROTOCOL)
    
    NEW ENGINES:
      1. Day-Type Physics: Classifies day as TREND, TRAP, or RANGE based on Asia/ATR & CVD.
      2. Tier-2 Liquidity: Detects 1H EQH/EQL (Equal Highs/Lows) for precise targeting.
      3. Execution Calculator: Generates exact Entry, SL (with Buffer), and TP1/TP2/TP3.
      4. Dynamic Rules: Adjusts Logic based on Day-Type (e.g., No Runners on Range Days).
    """

    # -----------------------------
    # 1. CONFIGURATION
    # -----------------------------
    CFG = {
        # --- ROLLING WINDOWS ---
        "roll_vol_n": 20,
        "roll_delta_n": 20,
        "h4_lookback": 10,
        "atr_period": 14,          # NEW: For SL Buffers & Range Calc

        # --- ADAPTIVE THRESHOLDS ---
        "abs_delta_mult": 1.5,
        "disp_body_ratio": 0.55,
        "eq_tolerance": 0.0005,    # NEW: 0.05% tolerance for EQH/EQL

        # --- TIME MAP (TEHRAN UTC+3:30) ---
        "T_RESET": time(0, 0),
        "T_ASIA_START": time(3, 30),
        "T_ASIA_END": time(8, 30),
        
        "T_PRE_LON_START": time(8, 30),  # Frankfurt CORE
        "T_LON_START": time(10, 30),     # London Expansion
        "T_LON_END": time(13, 30),
        
        "T_PRE_NY_START": time(13, 30),  # Pre-NY
        "T_OVERLAP_START": time(16, 30), # NY OVERLAP (CORE 2)
        "T_OVERLAP_END": time(20, 30),
        
        "T_NY_CLOSE_START": time(22, 30),

        # --- QUALITY GATES ---
        "K_CORE_MIN": 0.80,
        "K_FLOW_MIN": 0.65
    }

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        
        # --- GLOBAL FSM STATE ---
        self.state = {
            "Status": "SCANNING",   # SCANNING, PENDING, ACTIVE, MANAGE
            "Pending_Order": None,  # {Dir, Entry, SL, TP1, TP2, TP3}
        }
        
        # --- CONTEXT MEMORY ---
        self.context = {
            "Day_Type": "UNKNOWN",      # TREND, TRAP, RANGE
            "HTF_Bias": "NEUTRAL",
            "Flow_Bias": "NEUTRAL",
            "Asia_Range_ATR_Ratio": 0,
            "CVD_Slope": 0
        }
        
        # --- SESSION MEMORY ---
        self.sessions = {
            "ASIA":     {"High": 0, "Low": 0},
            "PRE_LON":  {"High": 0, "Low": 0, "Open_0830": 0},
            "LONDON":   {"High": 0, "Low": 0},
            "PRE_NY":   {"High": 0, "Low": 0},
            "OVERLAP":  {"High": 0, "Low": 0}
        }
        
        # --- TIERED LIQUIDITY MAP ---
        self.liquidity = {
            "Tier1_Major": [], # Session H/L
            "Tier2_Swing": [], # 1H EQH/EQL (The Trader's Feast)
            "Tier3_Minor": []  # 4H Structure
        }
        
        self.logs = []

    def log(self, msg: str):
        self.logs.append(msg)

    # -----------------------------
    # 2. DATA INFRASTRUCTURE
    # -----------------------------
    def _safe_float(self, s):
        return pd.to_numeric(s, errors='coerce').fillna(0.0)

    def load_data(self) -> bool:
        try:
            df = pd.read_csv(self.file_path)
            df.columns = [str(c).strip().lower() for c in df.columns]
            
            def get_col(candidates):
                for cand in candidates:
                    for c in df.columns:
                        if cand in c: return c
                return None

            c_time = get_col(['time', 'date'])
            c_close = get_col(['close'])
            c_high = get_col(['high'])
            c_low = get_col(['low'])
            c_open = get_col(['open'])
            c_delta = get_col(['volume delta', 'delta'])
            c_vol = get_col(['total volume', 'volume'])
            c_cvd = get_col(['cvd', 'cvd line'])

            if not all([c_time, c_close]): return False

            self.df = pd.DataFrame()
            self.df['Time'] = pd.to_datetime(df[c_time])
            self.df['Open'] = self._safe_float(df[c_open])
            self.df['High'] = self._safe_float(df[c_high])
            self.df['Low'] = self._safe_float(df[c_low])
            self.df['Close'] = self._safe_float(df[c_close])
            self.df['Vol'] = self._safe_float(df[c_vol]) if c_vol else 0.0
            self.df['Delta'] = self._safe_float(df[c_delta]) if c_delta else 0.0
            
            if c_cvd: self.df['CVD'] = self._safe_float(df[c_cvd])
            else: self.df['CVD'] = self.df['Delta'].cumsum()

            # Calculate ATR for Physics Engine
            self.df['TR'] = np.maximum(
                self.df['High'] - self.df['Low'],
                np.maximum(
                    abs(self.df['High'] - self.df['Close'].shift(1)),
                    abs(self.df['Low'] - self.df['Close'].shift(1))
                )
            )
            self.df['ATR'] = self.df['TR'].rolling(self.CFG['atr_period']).mean()

            self.df.sort_values('Time', inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            return True
        except Exception as e:
            self.log(f"Load Error: {e}")
            return False

    # -----------------------------
    # 3. TIER-2 LIQUIDITY ENGINE (EQH/EQL)
    # -----------------------------
    def _scan_tier2_liquidity(self):
        """
        Scans 1H timeframe for Equal Highs/Lows (EQH/EQL).
        This is Tier-2: The Smart Money Food.
        """
        agg = {'High':'max', 'Low':'min'}
        df_h1 = self.df.set_index('Time').resample('1H').agg(agg).dropna()
        
        self.liquidity['Tier2_Swing'] = []
        
        # Look at last 48 hours (approx 48 candles)
        recent_h1 = df_h1.iloc[-48:]
        
        # Detect EQH
        highs = recent_h1['High'].values
        for i in range(len(highs)):
            for j in range(i+1, len(highs)):
                h1, h2 = highs[i], highs[j]
                # If prices are very close (within tolerance)
                if abs(h1 - h2) / h1 < self.CFG['eq_tolerance']:
                    self.liquidity['Tier2_Swing'].append({'Price': max(h1, h2), 'Type': '1H EQH'})
                    
        # Detect EQL
        lows = recent_h1['Low'].values
        for i in range(len(lows)):
            for j in range(i+1, len(lows)):
                l1, l2 = lows[i], lows[j]
                if abs(l1 - l2) / l1 < self.CFG['eq_tolerance']:
                    self.liquidity['Tier2_Swing'].append({'Price': min(l1, l2), 'Type': '1H EQL'})

    # -----------------------------
    # 4. PHYSICS ENGINE: DAY-TYPE CLASSIFICATION
    # -----------------------------
    def _classify_day_type(self):
        """
        Determines if today is TREND, TRAP, or RANGE.
        Inputs: Asia Range, ATR, CVD Slope.
        """
        # 1. Get Asia Range
        asia_h = self.sessions['ASIA']['High']
        asia_l = self.sessions['ASIA']['Low']
        
        if asia_h == 0: 
            self.context['Day_Type'] = "FORMING"
            return
            
        asia_rng = asia_h - asia_l
        
        # 2. Get Yesterday's ATR (Daily Context)
        last_atr = self.df['ATR'].iloc[-1]
        if last_atr == 0: last_atr = 10 # Fallback
        
        ratio = asia_rng / last_atr
        self.context['Asia_Range_ATR_Ratio'] = ratio
        
        # 3. Get CVD Slope (Last 4 hours)
        recent_cvd = self.df['CVD'].iloc[-16:]
        try:
            slope = np.polyfit(range(len(recent_cvd)), recent_cvd, 1)[0]
            self.context['CVD_Slope'] = slope
        except: slope = 0
        
        # --- CLASSIFICATION LOGIC ---
        
        # A) TREND DAY: Small Asia Range (<0.5 ATR) + High CVD Momentum
        if ratio < 0.5 and abs(slope) > 500:
            self.context['Day_Type'] = "TREND"
            
        # B) TRAP DAY: Large Asia Range (>0.8 ATR) + Choppy/Divergent CVD
        elif ratio > 0.8:
            self.context['Day_Type'] = "TRAP"
            
        # C) RANGE DAY: Large/Medium Range + Flat CVD
        else:
            self.context['Day_Type'] = "RANGE"

    # -----------------------------
    # 5. SESSION BUILDER (UNCHANGED CORE)
    # -----------------------------
    def _build_session_levels(self, current_time):
        today = current_time.date()
        today_mask = (self.df['Time'].dt.date == today) & (self.df['Time'] <= current_time)
        today_df = self.df[today_mask]
        
        if today_df.empty: return

        def get_hl(start, end):
            mask = (today_df['Time'].dt.time >= start) & (today_df['Time'].dt.time < end)
            sub = today_df[mask]
            if sub.empty: return 0, 0
            return sub['High'].max(), sub['Low'].min()

        # Update Sessions
        self.sessions['ASIA'] = dict(zip(['High', 'Low'], get_hl(self.CFG['T_ASIA_START'], self.CFG['T_ASIA_END'])))
        
        # Pre-Lon & Anchor
        plh, pll = get_hl(self.CFG['T_PRE_LON_START'], self.CFG['T_LON_START'])
        anc = today_df[today_df['Time'].dt.time == self.CFG['T_PRE_LON_START']]
        zp = anc.iloc[0]['Open'] if not anc.empty else self.sessions['PRE_LON'].get('Open_0830', 0)
        self.sessions['PRE_LON'] = {'High': plh, 'Low': pll, 'Open_0830': zp}
        
        self.sessions['LONDON'] = dict(zip(['High', 'Low'], get_hl(self.CFG['T_LON_START'], self.CFG['T_LON_END'])))
        self.sessions['PRE_NY'] = dict(zip(['High', 'Low'], get_hl(self.CFG['T_PRE_NY_START'], self.CFG['T_OVERLAP_START'])))
        self.sessions['OVERLAP'] = dict(zip(['High', 'Low'], get_hl(self.CFG['T_OVERLAP_START'], self.CFG['T_OVERLAP_END'])))

        # Update Tier-1 Liquidity
        self.liquidity['Tier1_Major'] = []
        def add_liq(price, tag):
            if price > 0: self.liquidity['Tier1_Major'].append({'Price': price, 'Type': tag})
            
        add_liq(self.sessions['ASIA']['High'], 'ASIA HIGH'); add_liq(self.sessions['ASIA']['Low'], 'ASIA LOW')
        add_liq(self.sessions['LONDON']['High'], 'LON HIGH'); add_liq(self.sessions['LONDON']['Low'], 'LON LOW')
        add_liq(self.sessions['PRE_NY']['High'], 'PRE-NY HIGH'); add_liq(self.sessions['PRE_NY']['Low'], 'PRE-NY LOW')
        add_liq(self.sessions['OVERLAP']['High'], 'OVERLAP HIGH'); add_liq(self.sessions['OVERLAP']['Low'], 'OVERLAP LOW')

    # -----------------------------
    # 6. TRAP SCANNER (TIER 1 & 2)
    # -----------------------------
    def _scan_traps(self, idx):
        lookback = 3
        curr = self.df.iloc[idx]
        recent = self.df.iloc[max(0, idx-lookback):idx+1]
        traps = []
        
        avg_delta = self.df['Delta'].abs().rolling(self.CFG['roll_delta_n']).mean().iloc[idx]
        abs_thresh = avg_delta * self.CFG['abs_delta_mult']
        
        # Combined Levels (Tier 1 + Tier 2)
        target_levels = self.liquidity['Tier1_Major'] + self.liquidity['Tier2_Swing']
        
        for level in target_levels:
            price = level['Price']
            tag = level['Type']
            
            # BULLISH TRAP
            if "LOW" in tag or "EQL" in tag:
                if recent['Low'].min() < price and curr['Close'] > price:
                    if curr['Delta'] < -abs_thresh: traps.append(f"ABSORPTION BUY on {tag}")
                    elif curr['CVD'] > self.df['CVD'].iloc[idx-5:idx].min(): traps.append(f"CVD DIV BUY on {tag}")
                    else: traps.append(f"SIMPLE SWEEP BUY on {tag}")

            # BEARISH TRAP
            elif "HIGH" in tag or "EQH" in tag:
                if recent['High'].max() > price and curr['Close'] < price:
                    if curr['Delta'] > abs_thresh: traps.append(f"ABSORPTION SELL on {tag}")
                    elif curr['CVD'] < self.df['CVD'].iloc[idx-5:idx].max(): traps.append(f"CVD DIV SELL on {tag}")
                    else: traps.append(f"SIMPLE SWEEP SELL on {tag}")
        return traps

    # -----------------------------
    # 7. EXECUTION CALCULATOR (ENTRY/SL/TP)
    # -----------------------------
    def _calculate_trade_params(self, signal, curr_price, last_idx):
        if signal == "WAIT": return None
        
        atr = self.df['ATR'].iloc[last_idx]
        if atr == 0: atr = 5.0 # Safety fallback
        
        # 1. STOP LOSS (SL)
        # Logic: Behind the swing formed by the trap + Buffer
        buffer = atr * 0.2
        sl = 0
        if signal == "BUY":
            swing_low = self.df.iloc[last_idx-3:last_idx+1]['Low'].min()
            sl = swing_low - buffer
        elif signal == "SELL":
            swing_high = self.df.iloc[last_idx-3:last_idx+1]['High'].max()
            sl = swing_high + buffer
            
        # 2. TAKE PROFITS (TP)
        # Logic: Find nearest Tier 1/2 levels in direction of trade
        all_targets = [x['Price'] for x in self.liquidity['Tier1_Major'] + self.liquidity['Tier2_Swing']]
        tps = []
        
        if signal == "BUY":
            potential_tps = sorted([p for p in all_targets if p > curr_price])
            if len(potential_tps) > 0: tps.append(potential_tps[0]) # TP1
            if len(potential_tps) > 1: tps.append(potential_tps[1]) # TP2
            # TP3 Logic based on Day Type
            if self.context['Day_Type'] == "TREND":
                tps.append(potential_tps[-1] if potential_tps else curr_price + (atr * 5))
            elif self.context['Day_Type'] == "RANGE" and len(tps) > 1:
                pass # No TP3 for Range days, stick to TP2
                
        elif signal == "SELL":
            potential_tps = sorted([p for p in all_targets if p < curr_price], reverse=True)
            if len(potential_tps) > 0: tps.append(potential_tps[0]) # TP1
            if len(potential_tps) > 1: tps.append(potential_tps[1]) # TP2
            if self.context['Day_Type'] == "TREND":
                tps.append(potential_tps[-1] if potential_tps else curr_price - (atr * 5))

        return {
            "Dir": signal,
            "Entry": curr_price,
            "SL": sl,
            "TPs": tps,
            "RR_TP1": abs(tps[0]-curr_price)/abs(curr_price-sl) if tps else 0
        }

    # -----------------------------
    # 8. MAIN ENGINE (K-FACTOR INTEGRATED)
    # -----------------------------
    def run_engine(self):
        if not self.load_data(): return "DATA_ERROR"
        
        # Pre-calc
        self._build_session_levels(self.df.iloc[-1]['Time'])
        self._classify_day_type()
        self._scan_tier2_liquidity()
        
        # Snapshot
        last_idx = self.df.index[-1]
        curr = self.df.iloc[-1]
        t = curr['Time'].time()
        
        # Context
        traps = self._scan_traps(last_idx)
        zp = self.sessions['PRE_LON']['Open_0830']
        zp_stat = "PREMIUM" if curr['Close'] > zp else "DISCOUNT"
        if zp==0: zp_stat = "WAITING"

        # --- SIGNAL GENERATION ---
        phase = "UNKNOWN"; mode = "FLOW"; signal = "WAIT"; setup = "SCANNING"
        
        # (Same Phase Logic as V24 - Preserved for integrity)
        if self.CFG['T_PRE_LON_START'] <= t < self.CFG['T_LON_START']:
            phase = "FRANKFURT (CORE)"
            if "ABSORPTION" in str(traps):
                if zp_stat == "DISCOUNT": signal = "BUY"
                elif zp_status == "PREMIUM": signal = "SELL"
                setup = "FRANKFURT ABSORPTION"

        elif self.CFG['T_LON_START'] <= t < self.CFG['T_LON_END']:
            phase = "LONDON (EXPANSION)"
            if "SWEEP" in str(traps): 
                if zp_stat == "DISCOUNT": signal = "BUY"
                elif zp_status == "PREMIUM": signal = "SELL"
                setup = "LONDON JUDAS"

        elif self.CFG['T_PRE_NY_START'] <= t < self.CFG['T_OVERLAP_START']:
            phase = "PRE-NY"; setup = "Inducement"

        elif self.CFG['T_OVERLAP_START'] <= t < self.CFG['T_OVERLAP_END']:
            phase = "OVERLAP (CORE)"
            if "ABSORPTION" in str(traps) or "DIV" in str(traps):
                if "BUY" in str(traps): signal = "BUY"
                elif "SELL" in str(traps): signal = "SELL"
                setup = "OVERLAP REVERSAL"

        # --- VALIDATION ---
        # 1. Check if Entry Candle is valid
        if signal != "WAIT":
             # Inline Body Check
             rng = curr['High'] - curr['Low']
             body = abs(curr['Close'] - curr['Open'])
             if rng == 0 or (body/rng) < self.CFG['disp_body_ratio']:
                 signal = "WAIT"; setup += " (Weak Candle)"
        
        # 2. Check Day Type Compatibility
        if self.context['Day_Type'] == "TRAP" and mode == "FLOW":
            signal = "WAIT"; setup += " (Flow Forbidden in Trap Day)"

        # --- TRADE PARAMS ---
        trade_plan = self._calculate_trade_params(signal, curr['Close'], last_idx)

        # --- K-FACTOR ---
        k = 0.5
        if trade_plan:
            if "ABSORPTION" in str(traps): k += 0.25
            if (signal=="BUY" and zp_stat=="DISCOUNT") or (signal=="SELL" and zp_stat=="PREMIUM"): k += 0.1
            if self.context['Day_Type'] == "TREND": k += 0.1 # Trend days help all setups
            if phase == "OVERLAP (CORE)": k += 0.1
        
        decision = "NO_TRADE"
        min_k = self.CFG['K_CORE_MIN'] if mode == "CORE" else self.CFG['K_FLOW_MIN']
        
        if k >= min_k and signal != "WAIT":
            decision = "ENTRY_FULL" if k > 0.8 else "ENTRY_HALF"
            self.state['Status'] = "PENDING"
            self.state['Pending_Order'] = trade_plan
        else:
            if signal != "WAIT": setup += f" (GATED: K {k:.2f})"
            signal = "WAIT"
            self.state['Status'] = "SCANNING"

        return {
            "Time": str(t)[:5],
            "Phase": phase,
            "State": self.state['Status'],
            "Signal": signal,
            "Action": decision,
            "K_Factor": f"{k:.2f}",
            "Setup": setup,
            "Context": f"{self.context['Day_Type']} Day | HTF:{self.context['HTF_Bias']}",
            "Plan": trade_plan,
            "Traps": traps
        }

def generate_sovereign_report(data):
    plan_str = "---"
    if data['Plan']:
        p = data['Plan']
        tps = ", ".join([f"{x:.2f}" for x in p['TPs']])
        plan_str = f"\n   🚀 ENTRY: {p['Entry']}\n   🛑 SL: {p['SL']:.2f}\n   🎯 TPs: {tps}\n   ⚖️ RR (TP1): {p['RR_TP1']:.2f}"
    
    return f"""
🦅 **EAGLE V25.0: SOVEREIGN EDITION**
📍 **Phase:** {data['Phase']} | **Type:** {data['Context']}
⏰ **Time:** {data['Time']}

📝 **EXECUTION ORDER**
   • Signal: **{data['Signal']}** ({data['Action']})
   • K-Factor: {data['K_Factor']}
   • Setup: {data['Setup']}
   
🛠 **TRADE PLAN (Standardized)**{plan_str}

🚨 **ACTIVE TRAPS**
   {data['Traps'] if data['Traps'] else "(Scanning...)"}
   
#DayTypePhysics #Tier2Liquidity #Precision
"""

file_path = 'BYBIT_BTCUSDT, 15 (11).csv'
engine = EagleV25_Sovereign(file_path)
print(generate_sovereign_report(engine.run_engine()))
