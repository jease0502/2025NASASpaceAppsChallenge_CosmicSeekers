#!/usr/bin/env python3
"""
Survey Bias Analysis Tool
=========================

目標：針對 Kepler (KOI) 與 TESS (TOI) 兩個來源的行星資料，進行勘測偏誤 (survey bias) 比較：
1. 描述性統計比較 (Descriptive Statistics)
2. 分佈視覺化比較 (Histogram / KDE) 含離群值裁剪 (core vs full) 與自動 log 選項
3. 內部相關性結構比較 (Correlation Matrices + 差異圖)

輸出：
- descriptive_stats.csv (每欄位 KOI/TOI 的 count, mean, std, min, 25%,50%,75%, max)
- quantiles_<feature>.csv (各特徵量化資訊 + 裁剪界線) 於 output/quantiles/
- corr_koi.csv, corr_toi.csv, corr_diff.csv
- 圖表：
  * dist_<feature>_full.png / dist_<feature>_core.png (重疊 KDE + Histogram)
  * dist_<feature>_full_log.png / core_log.png (若觸發 log 門檻)
  * corr_heatmap_koi.png / corr_heatmap_toi.png / corr_heatmap_diff.png

使用方式：
    python scripts/survey_bias_analysis.py --features planet_radius planet_period star_radius \
        --out Correlation/analysis_out

可選參數：
    --all                 使用預設內建全部標準化欄位集合
    --trim-quantiles 0.01 0.99  兩端裁剪分位數 (預設 0.01 0.99)
    --min-n 30            低於此樣本數的欄位跳過 KDE (只畫 histogram)
    --log-threshold 25    若 (max / p95) > 門檻 則產生 log 版圖 (預設 30)

標準化欄位與來源映射定義於 STANDARD_TO_SOURCES。
"""
from __future__ import annotations
import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
from typing import List, Tuple, Dict, Optional

# ------------------ 字型設定 (避免中文缺字) ------------------
FONT_CANDIDATES = [
    # macOS 系統字型優先
    'PingFang TC', 'PingFang SC',
    # 常見 CJK 開源字型
    'Noto Sans CJK TC', 'Noto Sans CJK SC', 'Noto Sans CJK JP',
    # 其它備援
    'Heiti TC', 'STHeiti', 'Arial Unicode MS', 'Songti SC', 'Microsoft YaHei'
]

def setup_chinese_font(specified: Optional[str] = None) -> str:
    """嘗試設定支援中文的字型，回傳實際採用的字型名稱。"""
    import matplotlib
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = None
    if specified:
        if specified in available:
            chosen = specified
        else:
            print(f'[font] 指定字型 {specified} 未找到，將自動搜尋候選。')
    if not chosen:
        for candidate in FONT_CANDIDATES:
            if candidate in available:
                chosen = candidate
                break
    if not chosen:
        # 若完全找不到，仍然回傳目前設定 (可能造成缺字警告)
        print('[font] 未找到適合的中文字型，仍使用預設字型 (可能出現缺字警告)。')
        return matplotlib.rcParams.get('font.sans-serif', ['sans-serif'])[0]
    # 設定 rcParams
    matplotlib.rcParams['font.sans-serif'] = [chosen] + list(matplotlib.rcParams.get('font.sans-serif', []))
    matplotlib.rcParams['axes.unicode_minus'] = False
    print(f'[font] 已採用字型: {chosen}')
    return chosen

# ------------------ 映射設定 ------------------
# 新增第三個布林值：是否進行數值統計/分佈/相關性比較 (True=比較, False=跳過)
STANDARD_TO_SOURCES: Dict[str, Tuple[str | None, str | None, bool]] = {
    # A. Identification & Disposition (ID / label 類多數不做數值比較)
    'candidate_id': ('kepoi_name', 'toi', False),
    'confirmed_name': ('kepler_name', None, False),
    'host_star_id_kepler': ('kepid', None, False),
    'host_star_id_tess': (None, 'tid', False),
    'disposition_archive': ('koi_disposition', 'tfopwg_disp', False),
    'disposition_pipeline': ('koi_pdisposition', None, False),
    'disposition_score': ('koi_score', None, True),  # 分數仍為可比較數值
    'fp_flag_nt': ('koi_fpflag_nt', None, True),
    'fp_flag_ss': ('koi_fpflag_ss', None, True),
    'fp_flag_co': ('koi_fpflag_co', None, True),
    'fp_flag_ec': ('koi_fpflag_ec', None, True),

    # B. Planetary Orbital & Physical Parameters
    'planet_period': ('koi_period', 'pl_orbper', True),
    'planet_period_err_pos': ('koi_period_err1', 'pl_orbpererr1', True),
    'planet_period_err_neg': ('koi_period_err2', 'pl_orbpererr2', True),
    'planet_period_lim_flag': (None, 'pl_orbperlim', True),
    'planet_transit_epoch': ('koi_time0bk', 'pl_tranmid', True),
    'planet_transit_epoch_err_pos': ('koi_time0bk_err1', 'pl_tranmiderr1', True),
    'planet_transit_epoch_err_neg': ('koi_time0bk_err2', 'pl_tranmiderr2', True),
    'planet_transit_epoch_lim_flag': (None, 'pl_tranmidlim', True),
    'planet_transit_duration': ('koi_duration', 'pl_trandurh', True),
    'planet_transit_duration_err_pos': ('koi_duration_err1', 'pl_trandurherr1', True),
    'planet_transit_duration_err_neg': ('koi_duration_err2', 'pl_trandurherr2', True),
    'planet_transit_duration_lim_flag': (None, 'pl_trandurhlim', True),
    'planet_transit_depth': ('koi_depth', 'pl_trandep', True),
    'planet_transit_depth_err_pos': ('koi_depth_err1', 'pl_trandeperr1', True),
    'planet_transit_depth_err_neg': ('koi_depth_err2', 'pl_trandeperr2', True),
    'planet_transit_depth_lim_flag': (None, 'pl_trandeplim', True),
    'planet_radius': ('koi_prad', 'pl_rade', True),
    'planet_radius_err_pos': ('koi_prad_err1', 'pl_radeerr1', True),
    'planet_radius_err_neg': ('koi_prad_err2', 'pl_radeerr2', True),
    'planet_radius_lim_flag': (None, 'pl_radelim', True),
    'planet_equilibrium_temp': ('koi_teq', 'pl_eqt', True),
    'planet_equilibrium_temp_err_pos': ('koi_teq_err1', 'pl_eqterr1', True),
    'planet_equilibrium_temp_err_neg': ('koi_teq_err2', 'pl_eqterr2', True),
    'planet_equilibrium_temp_lim_flag': (None, 'pl_eqtlim', True),
    'planet_insolation': ('koi_insol', 'pl_insol', True),
    'planet_insolation_err_pos': ('koi_insol_err1', 'pl_insolerr1', True),
    'planet_insolation_err_neg': ('koi_insol_err2', 'pl_insolerr2', True),
    'planet_insolation_lim_flag': (None, 'pl_insollim', True),
    'planet_impact_param': ('koi_impact', None, True),
    'planet_impact_param_err_pos': ('koi_impact_err1', None, True),
    'planet_impact_param_err_neg': ('koi_impact_err2', None, True),

    # C. Stellar Physical Parameters
    'star_eff_temp': ('koi_steff', 'st_teff', True),
    'star_eff_temp_err_pos': ('koi_steff_err1', 'st_tefferr1', True),
    'star_eff_temp_err_neg': ('koi_steff_err2', 'st_tefferr2', True),
    'star_logg': ('koi_slogg', 'st_logg', True),
    'star_logg_err_pos': ('koi_slogg_err1', 'st_loggerr1', True),
    'star_logg_err_neg': ('koi_slogg_err2', 'st_loggerr2', True),
    'star_radius': ('koi_srad', 'st_rad', True),
    'star_radius_err_pos': ('koi_srad_err1', 'st_raderr1', True),
    'star_radius_err_neg': ('koi_srad_err2', 'st_raderr2', True),
    'star_distance': (None, 'st_dist', True),
    'star_distance_err_pos': (None, 'st_disterr1', True),
    'star_distance_err_neg': (None, 'st_disterr2', True),

    # D. Astrometry & Observational Parameters
    'ra_deg': ('ra', 'ra', True),
    'dec_deg': ('dec', 'dec', True),
    'star_pm_ra': (None, 'st_pmra', True),
    'star_pm_ra_err_pos': (None, 'st_pmraerr1', True),
    'star_pm_ra_err_neg': (None, 'st_pmraerr2', True),
    'star_pm_dec': (None, 'st_pmdec', True),
    'star_pm_dec_err_pos': (None, 'st_pmdecerr1', True),
    'star_pm_dec_err_neg': (None, 'st_pmdecerr2', True),
    'kepmag': ('koi_kepmag', None, True),
    'tessmag': (None, 'st_tmag', True),
    'signal_to_noise': ('koi_model_snr', None, True),
    'tce_planet_num': ('koi_tce_plnt_num', 'pl_pnum', False),  # 視為流水號
    'tce_delivery': ('koi_tce_delivname', None, False),
}

DEFAULT_FEATURES = [
    'planet_period', 'planet_radius', 'planet_transit_duration', 'planet_transit_depth',
    'planet_equilibrium_temp', 'planet_insolation', 'star_eff_temp', 'star_logg',
    'star_radius', 'tessmag'
]

# ------------------ 中文特徵名稱對照 ------------------
FEATURE_ZH: Dict[str, str] = {
    # Planetary
    'planet_period': '行星公轉週期 (天)',
    'planet_period_err_pos': '行星週期正誤差',
    'planet_period_err_neg': '行星週期負誤差',
    'planet_transit_epoch': '凌星中點時間 (BJD)',
    'planet_transit_epoch_err_pos': '凌星中點正誤差',
    'planet_transit_epoch_err_neg': '凌星中點負誤差',
    'planet_transit_duration': '凌星持續時間 (小時)',
    'planet_transit_duration_err_pos': '凌星持續時間正誤差',
    'planet_transit_duration_err_neg': '凌星持續時間負誤差',
    'planet_transit_depth': '凌星深度 (ppm)',
    'planet_transit_depth_err_pos': '凌星深度正誤差',
    'planet_transit_depth_err_neg': '凌星深度負誤差',
    'planet_radius': '行星半徑 (地球半徑)',
    'planet_radius_err_pos': '行星半徑正誤差',
    'planet_radius_err_neg': '行星半徑負誤差',
    'planet_equilibrium_temp': '行星平衡溫度 (K)',
    'planet_equilibrium_temp_err_pos': '平衡溫度正誤差',
    'planet_equilibrium_temp_err_neg': '平衡溫度負誤差',
    'planet_insolation': '入射輻照 (地球=1)',
    'planet_insolation_err_pos': '入射輻照正誤差',
    'planet_insolation_err_neg': '入射輻照負誤差',
    'planet_impact_param': '凌星衝擊參數',
    'planet_impact_param_err_pos': '衝擊參數正誤差',
    'planet_impact_param_err_neg': '衝擊參數負誤差',
    # Stellar
    'star_eff_temp': '恆星有效溫度 (K)',
    'star_eff_temp_err_pos': '恆星有效溫度正誤差',
    'star_eff_temp_err_neg': '恆星有效溫度負誤差',
    'star_logg': '恆星表面重力 logg',
    'star_logg_err_pos': 'logg 正誤差',
    'star_logg_err_neg': 'logg 負誤差',
    'star_radius': '恆星半徑 (太陽=1)',
    'star_radius_err_pos': '恆星半徑正誤差',
    'star_radius_err_neg': '恆星半徑負誤差',
    'star_distance': '恆星距離 (pc)',
    'star_distance_err_pos': '恆星距離正誤差',
    'star_distance_err_neg': '恆星距離負誤差',
    # Photometry & mags
    'kepmag': 'Kepler 視星等',
    'tessmag': 'TESS 視星等',
    # Astrometry
    'ra_deg': '赤經 (度)',
    'dec_deg': '赤緯 (度)',
    'star_pm_ra': '自行 (RA, mas/yr)',
    'star_pm_ra_err_pos': '自行 RA 正誤差',
    'star_pm_ra_err_neg': '自行 RA 負誤差',
    'star_pm_dec': '自行 (Dec, mas/yr)',
    'star_pm_dec_err_pos': '自行 Dec 正誤差',
    'star_pm_dec_err_neg': '自行 Dec 負誤差',
    # Misc
    'disposition_score': '處理評分',
    'fp_flag_nt': '假訊號標記-非凌星',
    'fp_flag_ss': '假訊號標記-恆星系統',
    'fp_flag_co': '假訊號標記-複合',
    'fp_flag_ec': '假訊號標記-蝕變星',
    'signal_to_noise': '模型訊噪比'
}

def feature_label(std_name: str, lang: str = 'zh') -> str:
    """Return display label for standardized feature.

    lang = 'zh': 英文標準化名稱 + 中文 (若有) 換行
    lang = 'en': 僅英文標準名
    """
    if lang == 'en':
        return std_name
    zh = FEATURE_ZH.get(std_name)
    return f'{std_name}' if not zh else f'{std_name}\n{zh}'

def format_number(v: float) -> str:
    """統一數值顯示：
    - 絕對值 < 1e-3 -> 科學記號 2 位有效數
    - 1e-3 <= |v| < 1 -> 保留 3 位小數 (去除尾端 0)
    - 1 <= |v| < 1e3 -> 最多 3 位有效數 (整數加千分位)
    - 1e3 <= |v| < 1e6 -> 以千分位整數 / 一位小數
    - 1e6 以上 -> 使用 K/M/B 縮寫 (一位小數)
    """
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ''
    av = abs(v)
    sign = '-' if v < 0 else ''
    if av < 1e-3 and av > 0:
        return f'{v:.2e}'
    if av < 1:
        s = f'{v:.3f}'.rstrip('0').rstrip('.')
        return s
    if av < 1e3:
        # 3 位有效數
        s = f'{v:.3g}'
        # 若是純數字且包含小數點，移除不必要尾 0
        if '.' in s and 'e' not in s:
            s = s.rstrip('0').rstrip('.')
        # 加千分位（僅整數部分）
        try:
            fpart = float(s)
            if fpart.is_integer():
                return f'{int(fpart):,}'
        except:
            pass
        return s
    if av < 1e6:
        return f'{v:,.1f}'.rstrip('0').rstrip('.')
    # K/M/B 縮寫
    units = [(1e9,'B'), (1e6,'M'), (1e3,'K')]
    for thresh, label in units:
        if av >= thresh:
            return f'{sign}{av/thresh:.1f}{label}'
    return f'{v:.3g}'

# ------------------ 工具函式 ------------------

def resolve_column(df: pd.DataFrame, std_name: str) -> str | None:
    mapping = STANDARD_TO_SOURCES.get(std_name)
    if std_name in df.columns:
        return std_name
    if mapping:
        k_col, t_col = mapping[0], mapping[1]
        for c in [k_col, t_col]:
            if c and c in df.columns:
                return c
    return None

def is_comparable(std_name: str) -> bool:
    m = STANDARD_TO_SOURCES.get(std_name)
    if not m:
        return True
    return bool(m[2])

def numeric_series(df: pd.DataFrame, col: str | None) -> pd.Series:
    if not col or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors='coerce').dropna()

# ------------------ 描述統計 ------------------

def descriptive_stats(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows = []
    for f in features:
        if not is_comparable(f):
            rows.append({'feature': f, 'count': 0, 'note': 'skipped(id/label)'})
            continue
        s = numeric_series(df, resolve_column(df, f))
        if s.empty:
            rows.append({'feature': f, 'count': 0})
            continue
        desc = s.describe(percentiles=[0.25,0.5,0.75])
        rows.append({
            'feature': f,
            'count': int(desc['count']),
            'mean': desc['mean'],
            'std': s.std(ddof=1),
            'min': desc['min'],
            '25%': desc['25%'],
            '50%': desc['50%'],
            '75%': desc['75%'],
            'max': desc['max'],
        })
    return pd.DataFrame(rows)

# ------------------ 分位數與裁剪邏輯 ------------------

def compute_quantiles(s: pd.Series) -> dict:
    qs = s.quantile([0,0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99,1.0])
    Q1, Q3 = qs[0.25], qs[0.75]
    IQR = Q3 - Q1
    upper_fence = Q3 + 1.5 * IQR
    lower_fence = Q1 - 1.5 * IQR
    return {
        'q00': qs[0.0], 'q01': qs[0.01], 'q05': qs[0.05], 'q10': qs[0.10], 'q25': Q1,
        'q50': qs[0.5], 'q75': Q3, 'q90': qs[0.9], 'q95': qs[0.95], 'q99': qs[0.99], 'q100': qs[1.0],
        'IQR': IQR, 'lower_fence': lower_fence, 'upper_fence': upper_fence
    }

def determine_core_range(qs: dict, trim_low: float, trim_high: float) -> Tuple[float,float]:
    # core 上界：min( upper_fence, quantile(trim_high) )
    upper_candidates = [qs['upper_fence'], qs['q99'] if trim_high >= 0.99 else qs['q95']]
    # 核心策略：用提供 quantile 作為首選；保留合理性
    target_high = qs['q95'] if trim_high <= 0.95 else (qs['q99'] if trim_high <= 0.99 else qs['q100'])
    core_high = min(qs['upper_fence'], target_high)
    # 下界：max( lower_fence, quantile(trim_low) )
    target_low = qs['q05'] if trim_low >= 0.05 else (qs['q01'] if trim_low >= 0.01 else qs['q00'])
    core_low = max(qs['lower_fence'], target_low)
    if core_low >= core_high:  # fallback
        core_low, core_high = qs['q05'], qs['q95']
    return float(core_low), float(core_high)

# ------------------ 繪圖 ------------------

def plot_distribution(feature: str, s1: pd.Series, s2: pd.Series, out_dir: Path, trim_low: float, trim_high: float,
                       min_n_kde: int, log_threshold: float, lang: str = 'zh'):
    if not is_comparable(feature):
        return
    if s1.empty and s2.empty:
        return
    q1 = compute_quantiles(s1) if not s1.empty else None
    q2 = compute_quantiles(s2) if not s2.empty else None

    # 決定核心範圍 (以合併後 quantiles 為主)
    combined = pd.concat([s1, s2]) if not s1.empty and not s2.empty else (s1 if not s1.empty else s2)
    combined_q = compute_quantiles(combined)
    core_low, core_high = determine_core_range(combined_q, trim_low, trim_high)

    # 準備輸出 quantiles csv
    q_rows = []
    for label, qs in [('KOI', q1), ('TOI', q2), ('COMBINED', combined_q)]:
        if qs is None: continue
        row = {'dataset': label, **qs, 'core_low': core_low, 'core_high': core_high}
        q_rows.append(row)
    q_df = pd.DataFrame(q_rows)
    (out_dir / 'quantiles').mkdir(exist_ok=True)
    q_df.to_csv(out_dir / 'quantiles' / f'quantiles_{feature}.csv', index=False)

    # 產生 full & core plots
    def _plot(kind: str, xlim: Tuple[float,float] | None, use_log: bool, suffix: str):
        plt.figure(figsize=(9,6))
        # hist
        bins = 40
        # 若為 core 視圖，先只取核心範圍內資料，避免 y 軸受全域極端值影響，並更精準反映核心內頻數。
        if kind == 'core' and xlim is not None:
            s1_plot = s1[(s1 >= core_low) & (s1 <= core_high)] if not s1.empty else s1
            s2_plot = s2[(s2 >= core_low) & (s2 <= core_high)] if not s2.empty else s2
            binrange = (core_low, core_high)
        else:
            s1_plot, s2_plot = s1, s2
            binrange = None

        if not s1_plot.empty:
            sns.histplot(s1_plot, stat='count', bins=bins, binrange=binrange, color='tab:blue', alpha=0.45, edgecolor='none', label='KOI')
        if not s2_plot.empty:
            sns.histplot(s2_plot, stat='count', bins=bins, binrange=binrange, color='tab:orange', alpha=0.45, edgecolor='none', label='TOI')
        # kde (若樣本數足夠)
        # KDE 同樣針對 core 視圖時使用裁剪後資料
        kde_s1 = s1_plot if (kind == 'core') else s1
        kde_s2 = s2_plot if (kind == 'core') else s2
        if not kde_s1.empty and len(kde_s1) >= min_n_kde and kde_s1.nunique() > 1:
            sns.kdeplot(kde_s1, color='tab:blue', lw=1.5)
        if not kde_s2.empty and len(kde_s2) >= min_n_kde and kde_s2.nunique() > 1:
            sns.kdeplot(kde_s2, color='tab:orange', lw=1.5)
        if xlim:
            # 若上下限相同 (所有值相同) 會觸發警告；人工擴張一個極小範圍
            if xlim[0] == xlim[1]:
                eps = 0.5 if xlim[0] == 0 else abs(xlim[0]) * 0.01 + 1e-9
                plt.xlim(xlim[0]-eps, xlim[1]+eps)
            else:
                plt.xlim(xlim)
        if use_log:
            plt.xscale('log')
        plt.xlabel(feature_label(feature, lang))
        plt.ylabel('Count')
        base_title = feature_label(feature, lang).replace('\n', ' ')
        if lang == 'en':
            extra = ' (core only)' if kind == 'core' else ''
            plt.title(f'{base_title} distribution ({kind}){extra}' + (' [log]' if use_log else ''))
        else:
            extra = ' (核心內數據 / core range only)' if kind == 'core' else ''
            plt.title(f'{base_title} 分佈 ({kind}){extra}' + (' [log]' if use_log else ''))
        plt.legend()
        if xlim and kind != 'core':
            # 僅在 full 視圖顯示核心界線，以免 core 圖內界線重疊造成視覺雜訊
            plt.axvline(core_low, color='k', ls='--', lw=0.8, alpha=0.6)
            plt.axvline(core_high, color='k', ls='--', lw=0.8, alpha=0.6)
        # 添加樣本數標註；full 顯示全域樣本；core 顯示核心範圍樣本。
        def fmt_n(n):
            return f'{int(n):,}'
        if kind == 'core':
            n1 = len(s1_plot) if not s1_plot.empty else 0
            n2 = len(s2_plot) if not s2_plot.empty else 0
            note = f'KOI core n={fmt_n(n1)}\nTOI core n={fmt_n(n2)}'
        else:
            n1 = len(s1) if not s1.empty else 0
            n2 = len(s2) if not s2.empty else 0
            note = f'KOI n={fmt_n(n1)}\nTOI n={fmt_n(n2)}'
        # 置於右上角 (axes 座標系)
        # 樣本數移至左上角
        plt.gca().text(0.01, 0.99, note, ha='left', va='top', transform=plt.gca().transAxes,
                       fontsize=9, bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='gray', alpha=0.6))
        # 底部 caption 說明
        if lang != 'en':
            caption_parts = [
                '長條=樣本數 (bin count); 曲線=平滑機率密度估計 KDE (樣本數足夠才出現)',
            ]
            if kind == 'core':
                caption_parts.append('core: 僅顯示核心範圍 (去除極端尾端)')
            if 'log' in suffix:
                caption_parts.append('log: x 軸對數刻度以展開長尾')
            if 'clipped' in suffix:
                caption_parts.append('clipped: 將最極端尾端裁剪以放大主要結構')
            caption = '；'.join(caption_parts)
            plt.gcf().text(0.5, 0.01, caption, ha='center', va='bottom', fontsize=8, color='dimgray')
        plt.tight_layout()
        fname = out_dir / f'dist_{feature}_{suffix}.png'
        plt.savefig(fname, dpi=160)
        plt.close()

    # Full range
    full_low = combined.min() if not combined.empty else 0
    full_high = combined.max() if not combined.empty else 1
    _plot('full', (full_low, full_high), use_log=False, suffix='full')

    # Core range: 使用 core_low/core_high
    _plot('core', (core_low, core_high), use_log=False, suffix='core')

    # Log scale (判斷是否需要)
    generated_clipped = False
    if full_high > 0 and combined_q['q95'] > 0 and (full_high / max(combined_q['q95'], 1e-9)) > log_threshold:
        _plot('full', (full_low, full_high), use_log=True, suffix='full_log')
        _plot('core', (core_low, core_high), use_log=True, suffix='core_log')

        # 產生 clipped full 圖：將上界裁剪至 p99 (或 core_high 與 p99*1.1 的較大者) 以改善另一資料集壓縮感
        p99 = combined_q['q99']
        clip_upper = max(core_high, p99 * 1.1)  # 給一點餘裕
        # 統計被裁掉的數量
        s1_clip = s1[s1 <= clip_upper]
        s2_clip = s2[s2 <= clip_upper]
        removed_koi = len(s1) - len(s1_clip)
        removed_toi = len(s2) - len(s2_clip)
        # 使用暫時覆蓋的資料呼叫 _plot，並在標題/註記顯示 clipped
        # 為保持函式簡潔，複製最小邏輯（不改動原 _plot 內部）
        plt.figure(figsize=(9,6))
        bins = 40
        if not s1_clip.empty:
            sns.histplot(s1_clip, stat='count', bins=bins, binrange=(full_low, clip_upper), color='tab:blue', alpha=0.45, edgecolor='none', label='KOI')
        if not s2_clip.empty:
            sns.histplot(s2_clip, stat='count', bins=bins, binrange=(full_low, clip_upper), color='tab:orange', alpha=0.45, edgecolor='none', label='TOI')
        # KDE 使用裁剪後資料避免假峰值
        if not s1_clip.empty and len(s1_clip) >= min_n_kde and s1_clip.nunique() > 1:
            sns.kdeplot(s1_clip, color='tab:blue', lw=1.5)
        if not s2_clip.empty and len(s2_clip) >= min_n_kde and s2_clip.nunique() > 1:
            sns.kdeplot(s2_clip, color='tab:orange', lw=1.5)
        plt.xlim(full_low, clip_upper)
        plt.xlabel(feature_label(feature, lang))
        plt.ylabel('Count')
        base_title = feature_label(feature, lang).replace('\n',' ')
        if lang == 'en':
            plt.title(f'{base_title} distribution (full clipped)')
        else:
            plt.title(f'{base_title} 分佈 (full clipped)')
        plt.legend()
        # 樣本與裁剪註記 (僅在 clipped 生成時寫入)
        def fmt_n(n): return f'{int(n):,}'
        note = (f'KOI n={fmt_n(len(s1))} (clip -{removed_koi})\n'
                f'TOI n={fmt_n(len(s2))} (clip -{removed_toi})\n'
                f'upper clipped @ {clip_upper:.3g}')
        plt.gca().text(0.01,0.99,note,ha='left',va='top',transform=plt.gca().transAxes,fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='gray', alpha=0.6))
        if lang != 'en':
            plt.gcf().text(0.5,0.01,'clipped: 上界裁剪於接近 p99 以提升細節；長條=樣本數；曲線=KDE',ha='center',va='bottom',fontsize=8,color='dimgray')
        plt.tight_layout()
        plt.savefig(out_dir / f'dist_{feature}_full_clipped.png', dpi=160)
        plt.close()

    # 若分佈極度集中且長尾，產出 zoom 圖 (左: 全域 或 log / 右: 放大核心/密集區)
    try:
        concentration_ratio = (core_high - core_low) / (full_high - full_low + 1e-12)
        long_tail_ratio = full_high / max(combined_q['q95'], 1e-9)
        need_zoom = (concentration_ratio < 0.35 and long_tail_ratio > log_threshold) or (long_tail_ratio > log_threshold * 2)
        if need_zoom:
            import matplotlib.gridspec as gridspec
            fig = plt.figure(figsize=(12,5.5))
            gs = gridspec.GridSpec(1,2, width_ratios=[2,1.2])
            # 左側全域 (可用 log)
            ax_full = fig.add_subplot(gs[0,0])
            bins_full = 40
            if not s1.empty:
                sns.histplot(s1, stat='count', bins=bins_full, color='tab:blue', alpha=0.45, edgecolor='none', label='KOI')
            if not s2.empty:
                sns.histplot(s2, stat='count', bins=bins_full, color='tab:orange', alpha=0.45, edgecolor='none', label='TOI')
            # 使用 log scale 若觸發
            use_log_full = long_tail_ratio > log_threshold
            if use_log_full:
                ax_full.set_xscale('log')
            ax_full.set_xlabel(feature_label(feature, lang))
            ax_full.set_ylabel('Count')
            base_title = feature_label(feature, lang).replace('\n',' ')
            if lang == 'en':
                ax_full.set_title(f'{base_title} full')
            else:
                ax_full.set_title(f'{base_title} 全域 / full')
            # 樣本數註記
            def fmt_n(n): return f'{int(n):,}'
            ax_full.text(0.01,0.99,f'KOI n={fmt_n(len(s1))}\nTOI n={fmt_n(len(s2))}',ha='left',va='top',transform=ax_full.transAxes,fontsize=9,
                         bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='gray', alpha=0.6))
            ax_full.legend()
            # 右側 zoom: 聚焦 core 區或 p95 區域 (使用更細 bins)
            zoom_high = combined_q['q95']
            ax_zoom = fig.add_subplot(gs[0,1])
            # 裁剪資料
            s1z = s1[(s1 >= full_low) & (s1 <= zoom_high)] if not s1.empty else s1
            s2z = s2[(s2 >= full_low) & (s2 <= zoom_high)] if not s2.empty else s2
            bins_zoom = 70
            if not s1z.empty:
                sns.histplot(s1z, stat='count', bins=bins_zoom, binrange=(full_low, zoom_high), color='tab:blue', alpha=0.50, edgecolor='none', label='KOI')
            if not s2z.empty:
                sns.histplot(s2z, stat='count', bins=bins_zoom, binrange=(full_low, zoom_high), color='tab:orange', alpha=0.50, edgecolor='none', label='TOI')
            # KDE (裁剪後)
            if not s1z.empty and len(s1z) >= min_n_kde and s1z.nunique() > 1:
                sns.kdeplot(s1z, color='tab:blue', lw=1.2)
            if not s2z.empty and len(s2z) >= min_n_kde and s2z.nunique() > 1:
                sns.kdeplot(s2z, color='tab:orange', lw=1.2)
            ax_zoom.set_xlim(full_low, zoom_high)
            ax_zoom.set_xlabel(feature_label(feature, lang))
            ax_zoom.set_ylabel('Count')
            if lang == 'en':
                ax_zoom.set_title('zoom (≤ p95)')
            else:
                ax_zoom.set_title('放大 / zoom (≤ p95)')
            # 樣本與覆蓋比例
            frac1 = len(s1z)/len(s1) if len(s1)>0 else 0
            frac2 = len(s2z)/len(s2) if len(s2)>0 else 0
            ax_zoom.text(0.01,0.99, f'KOI {fmt_n(len(s1z))} ({frac1*100:.1f}%)\nTOI {fmt_n(len(s2z))} ({frac2*100:.1f}%)', ha='left', va='top', transform=ax_zoom.transAxes, fontsize=9,
                         bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='gray', alpha=0.6))
            if lang != 'en':
                fig.text(0.5,0.01,'左: 全域(可能含 log)；右: ≤p95 放大。長條=樣本數；曲線=KDE；灰區=zoom 範圍。',ha='center',va='bottom',fontsize=8,color='dimgray')
            # 在 full 圖加上 zoom 區域指示線
            ax_full.axvspan(full_low, zoom_high, color='grey', alpha=0.08, label='zoom region')
            # 若使用 log，補充一段文字說明縮放
            if use_log_full:
                ax_full.text(0.01,0.02,'log scale', transform=ax_full.transAxes, fontsize=9, color='dimgray')
            fig.tight_layout()
            fig.savefig(out_dir / f'dist_{feature}_full_zoom.png', dpi=170)
            plt.close(fig)
    except Exception as e:
        print(f'[zoom] {feature} skip ({e})')

# ------------------ 相關性矩陣 ------------------

def correlation_matrices(df1: pd.DataFrame, df2: pd.DataFrame, features: List[str], out_dir: Path):
    # 轉成標準化欄位 -> 真實欄位名映射
    # 過濾不可比較欄位
    comp_features = [f for f in features if is_comparable(f)]
    col_map1 = {f: resolve_column(df1, f) for f in comp_features}
    col_map2 = {f: resolve_column(df2, f) for f in comp_features}
    valid_features1 = [f for f,c in col_map1.items() if c and df1[c].notna().sum() > 1]
    valid_features2 = [f for f,c in col_map2.items() if c and df2[c].notna().sum() > 1]

    # 建立純數值 DataFrame (只保留可用欄位)
    df1_std = pd.DataFrame({f: pd.to_numeric(df1[col_map1[f]], errors='coerce') for f in valid_features1})
    df2_std = pd.DataFrame({f: pd.to_numeric(df2[col_map2[f]], errors='coerce') for f in valid_features2})

    corr1 = df1_std.corr(numeric_only=True)
    corr2 = df2_std.corr(numeric_only=True)

    corr1.to_csv(out_dir / 'corr_koi.csv')
    corr2.to_csv(out_dir / 'corr_toi.csv')

    # 差異矩陣 (只對交集特徵)
    common = sorted(set(corr1.columns) & set(corr2.columns))
    if common:
        diff = corr1.loc[common, common] - corr2.loc[common, common]
        diff.to_csv(out_dir / 'corr_diff.csv')
    else:
        diff = None

    # 繪圖
    def heatmap(mat: pd.DataFrame, title: str, fname: str, center: float | None = 0.0, diverge: bool = False):
        plt.figure(figsize=(max(6, 0.6*len(mat.columns)), max(5, 0.6*len(mat.columns))))
        cmap = 'vlag' if diverge else 'viridis'
        sns.heatmap(mat, cmap=cmap, annot=False, square=True, center=center if diverge else None,
                    cbar_kws={'shrink':0.75})
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=170)
    plt.close()
    generated_clipped = True

    if not corr1.empty:
        heatmap(corr1, 'KOI Correlation Matrix', 'corr_heatmap_koi.png')
    if not corr2.empty:
        heatmap(corr2, 'TOI Correlation Matrix', 'corr_heatmap_toi.png')
    if diff is not None and not diff.empty:
        heatmap(diff, 'Correlation Difference (KOI - TOI)', 'corr_heatmap_diff.png', center=0.0, diverge=True)

# ------------------ 主流程 ------------------

def main():
    parser = argparse.ArgumentParser(description='Survey Bias Analysis: KOI vs TOI')
    parser.add_argument('--file1', help='KOI CSV (default Correlation/cumulative_*.csv)')
    parser.add_argument('--file2', help='TOI CSV (default Correlation/TOI_*.csv)')
    parser.add_argument('--features', nargs='*', help='指定要分析的標準化欄位')
    parser.add_argument('--all', action='store_true', help='使用所有可比較欄位 (STANDARD_TO_SOURCES 第三欄為 True 的全部)')
    parser.add_argument('--default-set', action='store_true', help='使用原始預設的小集合 DEFAULT_FEATURES')
    parser.add_argument('--trim-quantiles', nargs=2, type=float, metavar=('LOW','HIGH'), default=[0.01,0.99], help='裁剪分位數 (低,高) 預設 0.01 0.99')
    parser.add_argument('--min-n', type=int, default=30, help='KDE 最低樣本數 (預設30)')
    parser.add_argument('--log-threshold', type=float, default=30.0, help='若 max/p95 > 門檻 則輸出 log 圖 (預設30)')
    parser.add_argument('--out', default='analysis_out', help='輸出資料夾 (預設 analysis_out)')
    parser.add_argument('--stats-plots', action='store_true', help='產出統計值比較圖 (bar) 到 stats/ 下 (預設啟用，除非使用 --no-stats-plots)')
    parser.add_argument('--no-stats-plots', action='store_true', help='停用統計值比較圖')
    parser.add_argument('--font', help='指定中文字型名稱 (若系統存在)', default=None)
    parser.add_argument('--lang', choices=['zh','en'], default='zh', help='輸出語言: zh=中文(含 caption 與中文標籤); en=英文(不含中文說明 caption)')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    default1 = repo_root / 'Correlation' / 'cumulative_2025.09.22_04.01.47.csv'
    default2 = repo_root / 'Correlation' / 'TOI_2025.09.22_03.50.38.csv'
    f1 = Path(args.file1) if args.file1 else default1
    f2 = Path(args.file2) if args.file2 else default2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all and args.default_set:
        raise SystemExit('--all 與 --default-set 不可同時使用')
    if args.all:
        # 所有可比較欄位（第三值 True）
        features = [k for k,(a,b,ok) in STANDARD_TO_SOURCES.items() if ok]
    elif args.default_set:
        features = DEFAULT_FEATURES
    else:
        if not args.features:
            raise SystemExit('請使用 --features 或 --all 或 --default-set')
        features = args.features

    # 讀取 CSV
    df1 = pd.read_csv(f1, comment='#')
    df2 = pd.read_csv(f2, comment='#')

    # 初始化字型 (僅中文模式需要)
    if args.lang == 'zh':
        setup_chinese_font(args.font)

    # 描述統計
    desc1 = descriptive_stats(df1, features)
    desc2 = descriptive_stats(df2, features)
    merged = desc1.merge(desc2, on='feature', how='outer', suffixes=('_koi','_toi'))
    merged.to_csv(out_dir / 'descriptive_stats.csv', index=False)

    # 決定是否生成統計比較圖：預設開啟，除非 --no-stats-plots；若用戶顯式給 --stats-plots 則強制 On
    produce_stats = True
    if args.no_stats_plots:
        produce_stats = False
    if args.stats_plots:
        produce_stats = True

    if produce_stats:
        try:
            stats_dir = out_dir / 'stats'
            stats_dir.mkdir(exist_ok=True)
            # 欄位組: 以 mean, median(50%), IQR, count 為主要呈現；另附 min/max whiskers
            for _, row in merged.iterrows():
                feat = row['feature']
                if not is_comparable(feat):
                    continue
                # 取數值，若缺則跳過
                try:
                    mean_k = row.get('mean_koi'); mean_t = row.get('mean_toi')
                    if pd.isna(mean_k) and pd.isna(mean_t):
                        continue
                    # 構造比較表
                    data_points = []
                    def safe(v):
                        return None if (isinstance(v, float) and math.isnan(v)) else v
                    # Metric labels: remove Chinese to allow pure English output & avoid missing glyph warnings
                    if args.lang == 'en':
                        metrics = [
                            ('mean','Mean'),
                            ('50%','Median'),
                            ('25%','25% Quantile'),
                            ('75%','75% Quantile'),
                            ('std','Std Dev'),
                            ('count','Count'),
                        ]
                    else:
                        metrics = [
                            ('mean','平均值 Mean'),
                            ('50%','中位數 Median'),
                            ('25%','25% 分位 25% Quantile'),
                            ('75%','75% 分位 75% Quantile'),
                            ('std','標準差 Std'),
                            ('count','筆數 Count'),
                        ]
                    for key,label in metrics:
                        vk = safe(row.get(f'{key}_koi'))
                        vt = safe(row.get(f'{key}_toi'))
                        if vk is None and vt is None:
                            continue
                        data_points.append({'metric': label, 'dataset':'KOI', 'value': vk})
                        data_points.append({'metric': label, 'dataset':'TOI', 'value': vt})
                    if not data_points:
                        continue
                    df_plot = pd.DataFrame(data_points)
                    plt.figure(figsize=(9,5))
                    sns.barplot(data=df_plot, x='metric', y='value', hue='dataset', palette=['tab:blue','tab:orange'])
                    base_title = feature_label(feat, args.lang).replace('\n',' ')
                    if args.lang == 'en':
                        plt.title(f'{base_title} statistics comparison', fontsize=14)
                        plt.xlabel('Metric')
                        plt.ylabel('Value')
                    else:
                        plt.title(f'{base_title} 統計比較 / Statistics Comparison', fontsize=14)
                        plt.xlabel('統計指標 / Metric')
                        plt.ylabel('值 Value')
                    plt.xticks(rotation=25, ha='right')
                    # 加上數值標籤 (小字體)，略過 None
                    ax = plt.gca()
                    for patch in ax.patches:
                        height = patch.get_height()
                        if height is None or (isinstance(height, float) and math.isnan(height)):
                            continue
                        label = format_number(height)
                        x = patch.get_x() + patch.get_width()/2
                        ax.text(x, height, label, ha='center', va='bottom', fontsize=8, rotation=0)
                    plt.tight_layout()
                    plt.savefig(stats_dir / f'stats_{feat}.png', dpi=160)
                    plt.close()
                except Exception as ie:
                    print(f'[stats-plot] fail {feat}: {ie}')
        except Exception as e:
            print('產生統計比較圖失敗:', e)

    # 分佈比較圖
    for feat in features:
        s1 = numeric_series(df1, resolve_column(df1, feat))
        s2 = numeric_series(df2, resolve_column(df2, feat))
        plot_distribution(feat, s1, s2, out_dir, args.trim_quantiles[0], args.trim_quantiles[1], args.min_n, args.log_threshold, lang=args.lang)

    # 相關性矩陣
    correlation_matrices(df1, df2, features, out_dir)

    print('完成：輸出已寫入', out_dir)

if __name__ == '__main__':
    main()
