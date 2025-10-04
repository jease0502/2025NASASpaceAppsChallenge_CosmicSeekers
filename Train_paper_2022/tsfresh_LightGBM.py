#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kepler/TESS 系外行星偵測：重現 Malik et al. (2022, MNRAS 513:5505)
================================================================

核心流程（與論文一致的設計理念）
- 輸入：Kepler/TESS 光變曲線（light curves）
- 特徵：以 TSFresh 從時間序列自動萃取（~數百維；論文中 ~789）
- 模型：LightGBM（梯度提升樹）二元分類（CONFIRMED=1, FALSE POSITIVE=0）
- 標籤：來自 exoplanet_data_processed.csv 的 disposition（只取 CONFIRMED 與 FALSE POSITIVE）

注意事項
- 本腳本示範完整「可重現骨架」。實際大量下載與特徵萃取需要較久時間與較大記憶體。
- 你可以用 --max_targets 先跑少量樣本確認流程，再擴大。

環境需求（建議以虛擬環境安裝）
    pip install lightkurve tsfresh lightgbm scikit-learn pandas numpy joblib tqdm astropy

使用方式
    python reproduce_malik2022.py \
    --processed_csv data/exoplanet_data_processed.csv \
    --out_dir ./out_malik2022 \
    --max_targets 100 \
    --lc_dir ./lightkurve_data \
    --offline

# 線上模式（會自動下載光變曲線）
    python reproduce_malik2022.py \
    --processed_csv data/exoplanet_data_processed.csv \
    --out_dir ./out_malik2022 \
    --max_targets 100

輸出
- out_malik2022/
  - cache_lc/  （光變曲線快取）
  - features.csv  （TSFresh 特徵矩陣）
  - labels.csv    （對應標籤）
  - train_report.json （AUC、Precision/Recall、F1 等）

作者：James 專案需求整理 by ChatGPT（繁中註解）
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Memory

# 模型與特徵
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

# TSFresh 特徵
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters, ComprehensiveFCParameters

# 光變曲線下載與處理
import lightkurve as lk


# =============== 工具函式 ===============
def load_local_lc_csv(lightcurve_id: str, source_telescope: str, lc_dir: Path) -> pd.DataFrame:
    """讀取已預載的光變曲線 CSV（含 time,flux 欄位）並回傳 DataFrame。"""
    f = lc_dir / f"{source_telescope}_{lightcurve_id}.csv"
    if not f.exists():
        raise FileNotFoundError(f"Local LC not found: {f}")
    df = pd.read_csv(f)
    # 欄位名修正
    time_col = "time" if "time" in df.columns else next((c for c in df.columns if "time" in c.lower()), None)
    flux_col = "flux" if "flux" in df.columns else next((c for c in df.columns if "flux" in c.lower()), None)
    if not time_col or not flux_col:
        raise ValueError(f"{f} 必須包含 time 與 flux 欄位")
    out = df[[time_col, flux_col]].rename(columns={time_col: "time", flux_col: "flux"}).dropna()
    if len(out) == 0:
        raise ValueError(f"{f} 內容為空")
    return out




def load_processed_data(csv_path: Path) -> pd.DataFrame:
    """讀取 exoplanet_data_processed.csv"""
    df = pd.read_csv(csv_path)
    return df


def filter_labeled_data(df: pd.DataFrame) -> pd.DataFrame:
    """使用所有標籤（CONFIRMED、CANDIDATE、FALSE POSITIVE）進行三元分類。"""
    label_map = {
        "CONFIRMED": 2,
        "CANDIDATE": 1,
        "FALSE POSITIVE": 0
    }
    out = df[df["disposition"].isin(label_map)].copy()
    out["label"] = out["disposition"].map(label_map)
    # 去重：同一顆星/多顆候選的情況，這裡以 row 為單位處理（每個 source_id 作為一個樣本）
    out = out.reset_index(drop=True)
    # 建立樣本 id（穩定的）
    out["sample_id"] = out.index.astype(int)
    # 打印類別分布
    print("類別分布：")
    print(out["disposition"].value_counts())
    return out


def pick_targets(df: pd.DataFrame, max_targets: int = None) -> pd.DataFrame:
    """可選擇子集以先跑小量測試。"""
    if max_targets is not None and max_targets > 0:
        return df.sample(n=min(max_targets, len(df)), random_state=42).reset_index(drop=True)
    return df


# 快取設定（避免重複下載）
CACHE_BASE = None
MEMORY = None

def init_cache(base_dir: Path):
    global CACHE_BASE, MEMORY
    CACHE_BASE = base_dir / "cache_lc"
    CACHE_BASE.mkdir(parents=True, exist_ok=True)
    MEMORY = Memory(location=str(CACHE_BASE), verbose=0)


@MEMORY.cache if MEMORY else (lambda f: f)
def download_kepler_lc_by_kepid(kepid: int) -> lk.LightCurve:
    """下載並回傳 *合併後* 的 PDCSAP_FLUX 光變曲線（可能跨多季度）。
    若找不到 PDCSAP，就退而求其次使用 SAP_FLUX。
    """
    # 使用 KIC ID 檢索
    # 使用新版 API（相容舊版）；新版建議使用 search_lightcurve()
    try:
        search = lk.search_lightcurve(f"KIC {int(kepid)}", mission="Kepler")
    except Exception:
        # 向後相容
        search = lk.search_lightcurvefile(f"KIC {int(kepid)}", mission="Kepler")
    if len(search) == 0:
        raise FileNotFoundError(f"No Kepler LCF for KIC {kepid}")

    lcf_all = search.download_all()
    # 優先選擇 PDCSAP_FLUX（去除系統性趨勢）
    try:
        lc = lcf_all.PDCSAP_FLUX.stitch()
    except Exception:
        lc = None
    if lc is None or len(lc.flux) == 0:
        # 退而求其次
        try:
            lc = lcf_all.SAP_FLUX.stitch()
        except Exception:
            raise FileNotFoundError(f"No usable light curve for KIC {kepid}")

    # 基礎清理：去 NaN、平坦化、正規化
    lc = lc.remove_nans()
    # Kepler long-cadence ~29.4 minutes；window_length 可視情況調（此處選較保守）
    try:
        lc = lc.flatten(window_length=301)
    except Exception:
        pass
    try:
        lc = lc.normalize()
    except Exception:
        pass
    return lc


@MEMORY.cache if MEMORY else (lambda f: f)
def download_tess_lc_by_tic(tic: int) -> lk.LightCurve:
    """下載並回傳 TESS 光變曲線。"""
    try:
        search = lk.search_lightcurve(f"TIC {int(tic)}", mission="TESS")
    except Exception:
        search = lk.search_lightcurvefile(f"TIC {int(tic)}", mission="TESS")
    if len(search) == 0:
        raise FileNotFoundError(f"No TESS LCF for TIC {tic}")

    lcf_all = search.download_all()
    # 優先選擇 PDCSAP_FLUX
    try:
        lc = lcf_all.PDCSAP_FLUX.stitch()
    except Exception:
        lc = None
    if lc is None or len(lc.flux) == 0:
        # 退而求其次
        try:
            lc = lcf_all.SAP_FLUX.stitch()
        except Exception:
            raise FileNotFoundError(f"No usable light curve for TIC {tic}")

    # 基礎清理：去 NaN、平坦化、正規化
    lc = lc.remove_nans()
    try:
        lc = lc.flatten(window_length=301)
    except Exception:
        pass
    try:
        lc = lc.normalize()
    except Exception:
        pass
    return lc


def lc_to_timeseries_df(lc: lk.LightCurve, sample_id: int) -> pd.DataFrame:
    """將 lightkurve 的 LightCurve 轉成 TSFresh 需要的長表格格式：
        columns: [id, time, flux]
        - id: sample_id（對應 KOI row）
        - time: 以 BTJD 天為浮點時間（TSFresh 支援任意遞增時間欄位）
        - flux: 正規化光度
    """
    t = np.array(lc.time.value, dtype=float)
    f = np.array(lc.flux.value if hasattr(lc.flux, 'value') else lc.flux, dtype=float)
    df_ts = pd.DataFrame({
        "id": sample_id,
        "time": t,
        "flux": f
    })
    return df_ts


def build_timeseries_table(df_samples: pd.DataFrame,
                           lc_dir: Path | None = None,
                           offline: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    回傳：
      - ts_long: 長表 (id,time,flux)
      - labels: sample_id,label,lightcurve_id,source_telescope,...（不變）
    """
    all_ts = []
    labels = df_samples[["sample_id", "label", "lightcurve_id", "source_telescope", "source_id"]].copy()
    for _, row in tqdm(df_samples.iterrows(), total=len(df_samples)):
        lightcurve_id = str(row["lightcurve_id"])
        source_telescope = row["source_telescope"]
        sid = int(row["sample_id"])

        # 1) 先試本地 CSV
        if lc_dir is not None:
            try:
                df = load_local_lc_csv(lightcurve_id, source_telescope, lc_dir)
                ts = pd.DataFrame({"id": sid,
                                   "time": df["time"].to_numpy(dtype=float),
                                   "flux": df["flux"].to_numpy(dtype=float)})
                if len(ts):
                    all_ts.append(ts)
                    continue
            except Exception as e:
                if offline:
                    print(f"[SKIP] {source_telescope} {lightcurve_id} 本地無檔或無效（offline 模式）：{e}")
                    continue  # 嚴格離線：直接跳過
                # 非離線：落回線上抓

        # 2) 非離線才嘗試線上抓
        if offline:
            print(f"[SKIP] {source_telescope} {lightcurve_id} 缺本地檔（offline）。")
            continue

        try:
            if source_telescope == "KOI":
                lc = download_kepler_lc_by_kepid(int(lightcurve_id))  # 原本的線上抓邏輯
            elif source_telescope == "TESS":
                lc = download_tess_lc_by_tic(int(lightcurve_id))
            else:
                print(f"[WARN] 不支援的望遠鏡類型: {source_telescope}")
                continue
                
            ts = lc_to_timeseries_df(lc, sid)
            if len(ts):
                all_ts.append(ts)
        except Exception as e:
            print(f"[WARN] {source_telescope} {lightcurve_id} 線上抓失敗：{e}")

    if not all_ts:
        raise RuntimeError("No time series collected. 請確認 lc_dir 內是否有對應的 CSV 或關閉 --offline 允許回退下載。")
    ts_long = pd.concat(all_ts, axis=0, ignore_index=True)
    return ts_long, labels



def extract_tsfresh_features(ts_long: pd.DataFrame, comprehensive: bool = False) -> pd.DataFrame:
    """以 TSFresh 從長表格抽取特徵。
    - comprehensive=True: 使用 ComprehensiveFCParameters（~789 個特徵，論文設定）
    - comprehensive=False: 使用 EfficientFCParameters（~776 個特徵，快速模式）
    - 確保回傳包含 'sample_id' 欄位（兼容單一/多樣本情況）。
    """
    # TSFresh 需要按時間排序
    ts_long = ts_long.sort_values(["id", "time"]).reset_index(drop=True)

    # 檢查資料品質
    print(f"  時間序列資料點數: {len(ts_long)}")
    print(f"  樣本數: {ts_long['id'].nunique()}")
    
    # 移除異常值
    ts_long = ts_long.dropna()
    if len(ts_long) == 0:
        raise ValueError("時間序列資料為空")
    
    # 檢查每個樣本的資料點數
    sample_counts = ts_long.groupby('id').size()
    print(f"  每個樣本資料點數範圍: {sample_counts.min()} - {sample_counts.max()}")
    
    # 移除資料點太少的樣本
    min_points = 50  # 最少需要50個資料點
    valid_samples = sample_counts[sample_counts >= min_points].index
    ts_long = ts_long[ts_long['id'].isin(valid_samples)]
    
    if len(ts_long) == 0:
        raise ValueError(f"沒有樣本有足夠的資料點（至少{min_points}個）")
    
    print(f"  過濾後樣本數: {ts_long['id'].nunique()}")
    print(f"  過濾後資料點數: {len(ts_long)}")

    # 選擇特徵萃取參數
    if comprehensive:
        fc_params = ComprehensiveFCParameters()  # ~789 個特徵（論文設定）
        print("  使用 ComprehensiveFCParameters（~789 個特徵）")
    else:
        fc_params = EfficientFCParameters()  # ~776 個特徵（快速模式）
        print("  使用 EfficientFCParameters（~776 個特徵）")
    
    feats = extract_features(
        ts_long,
        column_id="id",
        column_sort="time",
        column_value="flux",
        default_fc_parameters=fc_params,
        disable_progressbar=False,
        n_jobs=0,
    )

    # 兼容單一樣本時 index/name 可能不同的情況，強制產生 sample_id 欄位
    if isinstance(feats.index, pd.MultiIndex):
        feats = feats.reset_index()
    else:
        feats = feats.reset_index()

    # 嘗試把重建出的索引欄位改名為 sample_id
    if "id" in feats.columns:
        feats = feats.rename(columns={"id": "sample_id"})
    elif "index" in feats.columns:
        feats = feats.rename(columns={"index": "sample_id"})

    # 萬一仍沒有，最後再保底建立
    if "sample_id" not in feats.columns:
        feats.insert(0, "sample_id", range(len(feats)))

    # 移除全為 NaN 的特徵
    feats = feats.dropna(axis=1, how='all')
    print(f"  萃取特徵數: {feats.shape[1] - 1}")  # 減去 sample_id 欄位
    
    return feats


def train_and_eval(X: pd.DataFrame, y: np.ndarray, out_dir: Path) -> Dict:
    """訓練 LightGBM 並輸出評估報告（三元分類版本）。"""
    # 檢查類別分佈
    unique_classes, counts = np.unique(y, return_counts=True)
    class_names = ['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED']
    print(f"  類別分佈: {dict(zip(class_names, counts))}")
    
    # 檢查樣本量是否足夠
    total_samples = len(y)
    min_samples = min(counts)
    
    if total_samples < 50:
        print(f"  警告：樣本量太少（{total_samples}），建議增加 --max_targets")
    if min_samples < 10:
        print(f"  警告：某個類別樣本太少（{min_samples}），可能影響模型性能")
    
    # 如果某個類別樣本太少，不使用分層抽樣
    if min_samples < 2:
        print(f"  警告：某個類別只有 {min_samples} 個樣本，不使用分層抽樣")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

    # 根據樣本量調整模型參數
    if total_samples < 100:
        # 小樣本：使用較保守的參數
        n_estimators = min(100, total_samples // 2)
        learning_rate = 0.1
        max_depth = 3
        print(f"  小樣本模式：n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}")
    else:
        # 正常樣本：使用標準參數
        n_estimators = 1000  # 增加樹的數量以處理多分類
        learning_rate = 0.05
        max_depth = 7  # 增加深度以捕捉更複雜的模式
        print(f"  標準模式：n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}")

    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("lgbm", LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,  # 使用單線程避免記憶體問題
            verbose=-1,
            class_weight='balanced',
            objective='multiclass',  # 多分類設置
            num_class=3  # 三個類別
        )),
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)

    # 計算每個類別的 ROC AUC（one-vs-rest）
    auc_scores = []
    for i in range(3):
        try:
            auc = float(roc_auc_score((y_test == i).astype(int), y_proba[:, i]))
            auc_scores.append(auc)
        except Exception:
            auc_scores.append(float('nan'))
    
    # 使用 macro-averaged 指標
    report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        output_dict=True
    )

    # 存結果
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train_report.json", "w") as f:
        json.dump({
            "auc_scores": {
                "FALSE_POSITIVE": float(auc_scores[0]),
                "CANDIDATE": float(auc_scores[1]),
                "CONFIRMED": float(auc_scores[2])
            },
            "report": report,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "n_features": int(X.shape[1]),
            "total_samples": int(total_samples),
            "min_class_samples": int(min_samples),
            "class_distribution": {
                "FALSE_POSITIVE": int(counts[0]),
                "CANDIDATE": int(counts[1]),
                "CONFIRMED": int(counts[2])
            }
        }, f, indent=2)

    # 也存模型（joblib）
    try:
        import joblib
        joblib.dump(pipe, out_dir / "model_lgbm.joblib")
    except Exception:
        pass

    return {
        "auc_scores": dict(zip(class_names, auc_scores)),
        "report": report
    }


# =============== 主流程 ===============

def main():
    parser = argparse.ArgumentParser(description="Reproduce Malik et al. 2022: TSFresh + LightGBM on Kepler/TESS LCs")
    parser.add_argument("--processed_csv", type=str, required=True, help="exoplanet_data_processed.csv 檔路徑")
    parser.add_argument("--out_dir", type=str, required=True, help="輸出目錄")
    parser.add_argument("--max_targets", type=int, default=200, help="最多處理多少個樣本（先小量測試）")
    parser.add_argument("--lc_dir", type=str, default=None,
                    help="已預先下載的光變曲線 CSV 目錄（包含 KOI_*.csv 和 TESS_*.csv，欄位 time,flux）")
    parser.add_argument("--features_dir", type=str, default=None,
                    help="預處理的 TSFresh 特徵目錄（包含 features_combined.csv）")
    parser.add_argument("--comprehensive", action="store_true",
                        help="使用 ComprehensiveFCParameters（~789 個特徵，論文設定）")
    parser.add_argument("--offline", action="store_true",
                        help="離線模式：只讀 lc_dir，本地缺檔就跳過，不連網")

    args = parser.parse_args()

    processed_csv = Path(args.processed_csv)
    out_dir = Path(args.out_dir)
    init_cache(out_dir)

    print("[1/6] 讀取處理過的資料 ...")
    df_raw = load_processed_data(processed_csv)
    df_lbl = filter_labeled_data(df_raw)
    df_pick = pick_targets(df_lbl, args.max_targets)

    print(f"  樣本數（含標籤）：{len(df_lbl)}，本次處理：{len(df_pick)}")

    # 檢查是否使用預處理的特徵
    if args.features_dir:
        print("[2/6] 使用預處理的 TSFresh 特徵 ...")
        features_path = Path(args.features_dir) / "features_combined.csv"
        if not features_path.exists():
            raise FileNotFoundError(f"預處理特徵檔案不存在: {features_path}")
        
        feats = pd.read_csv(features_path)
        print(f"  載入特徵: {feats.shape[1]} 個特徵，{feats.shape[0]} 個樣本")
        
        # 過濾對應的樣本
        sample_ids = set(df_pick['sample_id'])
        feats = feats[feats['sample_id'].isin(sample_ids)]
        print(f"  過濾後特徵: {feats.shape[0]} 個樣本")
        
        # 建立標籤對應
        labels = df_pick[["sample_id", "label", "lightcurve_id", "source_telescope", "source_id"]].copy()
        
    else:
        print("[2/6] 下載與整理光變曲線 ...（需要網路，建議先用少量測試）")
        ts_long, labels = build_timeseries_table(
                            df_pick,
                            lc_dir=Path(args.lc_dir) if args.lc_dir else None,
                            offline=args.offline
                        )

        print("[3/6] 以 TSFresh 萃取特徵 ...")
        feats = extract_tsfresh_features(ts_long, comprehensive=args.comprehensive)

    print("[4/6] 特徵與標籤對齊 ...")
    data = labels.merge(feats, on="sample_id", how="inner")
    y = data["label"].astype(int).values
    
    # 動態檢查並移除存在的欄位
    columns_to_drop = ["sample_id", "label"]
    available_columns = ["lightcurve_id", "source_telescope", "source_id"]
    
    for col in available_columns:
        if col in data.columns:
            columns_to_drop.append(col)
    
    X = data.drop(columns=columns_to_drop)  # 只保留 TSFresh 特徵
    
    # 清理特徵資料：只保留數值特徵
    print(f"  原始特徵數: {X.shape[1]}")
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_columns]
    print(f"  數值特徵數: {X.shape[1]}")
    
    # 移除全為 NaN 的特徵
    X = X.dropna(axis=1, how='all')
    print(f"  清理後特徵數: {X.shape[1]}")
    
    if X.shape[1] == 0:
        raise ValueError("沒有可用的數值特徵進行訓練")

    print("[5/6] 訓練與評估 LightGBM（三元分類）...")
    metrics = train_and_eval(X, y, out_dir)
    print("\n各類別 AUC 分數：")
    for class_name, auc in metrics['auc_scores'].items():
        print(f"  {class_name}: {auc:.3f}")
    
    print("\n分類報告：")
    for class_name in ['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED']:
        class_metrics = metrics['report'][class_name]
        print(f"\n{class_name}:")
        print(f"  精確度: {class_metrics['precision']:.3f}")
        print(f"  召回率: {class_metrics['recall']:.3f}")
        print(f"  F1分數: {class_metrics['f1-score']:.3f}")

    print("[6/6] 輸出檔案 ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    feats.to_csv(out_dir / "features.csv", index=False)
    labels.to_csv(out_dir / "labels.csv", index=False)
    print(f"完成，輸出位於：{out_dir}")


if __name__ == "__main__":
    main()




