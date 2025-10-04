#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preload Light Curves from exoplanet_data_processed.csv
=====================================================

- 讀取 exoplanet_data_processed.csv
- 根據 source_telescope (KOI/TESS) 和 lightcurve_id 下載光變曲線
- 支援 Kepler 和 TESS 資料
- 平行下載（ThreadPoolExecutor）
- 每顆星輸出 CSV：time,flux（UTF-8）

requirements:
    pip install lightkurve pandas numpy tqdm

usage:
    python preload_data_v2.py \
        --processed_csv exoplanet_data_processed.csv \
        --data_dir ./lightkurve_data \
        --max_targets 50 \
        --max_workers 4 \
        --window_days 30 \
        --downsample 20000
"""

from __future__ import annotations

import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
import time
import sys
from typing import List, Tuple, Optional, Dict, Any
import threading

import numpy as np
import pandas as pd

# 靜音不必要的警告（不影響 LC 下載/處理）
warnings.filterwarnings(
    "ignore",
    message=".*tpfmodel submodule is not available without oktopus.*"
)
warnings.filterwarnings("ignore", module="lightkurve.prf")
warnings.filterwarnings("ignore", category=UserWarning, module="lightkurve")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="lightkurve")
warnings.filterwarnings("ignore", message=".*PDCSAP_FLUX.*deprecated.*")
warnings.filterwarnings("ignore", message=".*files available to download.*")
warnings.filterwarnings("ignore", message=".*File may have been truncated.*")
warnings.filterwarnings("ignore", message=".*UnitsWarning.*")

try:
    import lightkurve as lk
except Exception as e:
    print("[FATAL] lightkurve 未安裝或無法匯入：", e, file=sys.stderr)
    sys.exit(1)


# 線程安全的輸出鎖
print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    """線程安全的 print 函數"""
    with print_lock:
        try:
            print(*args, **kwargs)
        except (ValueError, OSError):
            # 如果 stdout 被關閉，忽略錯誤
            pass


# -------------------------
# 讀取處理過的資料
# -------------------------
def load_processed_data(csv_path: Path) -> pd.DataFrame:
    """讀取 exoplanet_data_processed.csv"""
    df = pd.read_csv(csv_path)
    return df


def select_labeled_samples(
    df: pd.DataFrame, max_targets: Optional[int] = None
) -> pd.DataFrame:
    """只保留 CONFIRMED / FALSE POSITIVE；建立簡化樣本表。"""
    label_map = {"CONFIRMED": 1, "FALSE POSITIVE": 0}
    sub = df[df["disposition"].isin(label_map)].copy()

    # 只保留需要的欄位
    keep_cols = [
        "source_id", "source_telescope", "disposition", "lightcurve_id",
        "period", "transit_midpoint"
    ]
    for c in keep_cols:
        if c not in sub.columns:
            raise KeyError(f"處理過的 CSV 缺少必要欄位：{c}")
    
    sub = sub[keep_cols].dropna(subset=["lightcurve_id"])

    # 隨機取樣（若指定 max_targets）
    if max_targets and max_targets > 0:
        sub = sub.sample(n=min(max_targets, len(sub)), random_state=42).reset_index(drop=True)
    else:
        sub = sub.reset_index(drop=True)

    # 確保型別
    sub["lightcurve_id"] = sub["lightcurve_id"].astype(str)
    return sub


# -------------------------
# Lightkurve 下載與處理
# -------------------------
def _find_bad_fits_path_from_error(msg: str) -> Optional[Path]:
    """從錯誤訊息找出壞掉的 .fits 路徑（macOS/Linux 通用）。"""
    # 嘗試多種模式匹配損壞的文件路徑
    patterns = [
        r"(/[^\\\s]+?\.fits)",  # 基本 .fits 文件
        r"(lightkurve_data/mastDownload/[^\\\s]+?\.fits)",  # 完整路徑
        r"(mastDownload/[^\\\s]+?\.fits)"  # 相對路徑
    ]
    
    for pattern in patterns:
        m = re.search(pattern, msg)
        if m:
            p = Path(m.group(1))
            if p.exists():
                return p
    return None


def _download_kepler_lc(lightcurve_id: str, download_dir: Path):
    """下載 Kepler 光變曲線"""
    res = lk.search_lightcurve(
        f"KIC {lightcurve_id}",
        mission="Kepler",
        exptime=1800,       # long-cadence
        author="Kepler"
    )
    if len(res) == 0:
        raise FileNotFoundError(f"No Kepler LC search result for KIC {lightcurve_id}")
    # 只抓一個檔案
    lcf = res.download(download_dir=str(download_dir))
    return lcf


def _download_tess_lc(lightcurve_id: str, download_dir: Path):
    """下載 TESS 光變曲線"""
    res = lk.search_lightcurve(
        f"TIC {lightcurve_id}",
        mission="TESS"
    )
    if len(res) == 0:
        raise FileNotFoundError(f"No TESS LC search result for TIC {lightcurve_id}")
    # 只抓一個檔案
    lcf = res.download(download_dir=str(download_dir))
    return lcf


def safe_download_lcfile(
    lightcurve_id: str, 
    source_telescope: str, 
    download_dir: Path, 
    retries: int = 1
):
    """下載光變曲線檔案，若遇壞檔錯誤，刪除後重試一次。"""
    try:
        if source_telescope == "KOI":
            return _download_kepler_lc(lightcurve_id, download_dir)
        elif source_telescope == "TESS":
            return _download_tess_lc(lightcurve_id, download_dir)
        else:
            raise ValueError(f"不支援的望遠鏡類型: {source_telescope}")
    except Exception as e:
        if retries <= 0:
            raise
        
        # 清理損壞的文件和目錄
        bad = _find_bad_fits_path_from_error(str(e))
        if bad and bad.exists():
            try:
                # 刪除損壞的文件
                bad.unlink()
                # 如果父目錄為空，也刪除它
                parent_dir = bad.parent
                if parent_dir.exists() and not any(parent_dir.iterdir()):
                    parent_dir.rmdir()
                time.sleep(1.0)  # 增加等待時間
            except Exception:
                pass
        
        # 清理整個目標目錄（如果存在）
        target_dir_pattern = download_dir / f"mastDownload/*/{lightcurve_id}_*"
        try:
            import glob
            for dir_path in glob.glob(str(target_dir_pattern)):
                import shutil
                shutil.rmtree(dir_path)
        except Exception:
            pass
        
        # 再試一次
        return safe_download_lcfile(lightcurve_id, source_telescope, download_dir, retries=retries - 1)


def lcfile_to_lightcurve(lcf) -> "lk.LightCurve":
    """從 LightCurveFile 取 PDCSAP_FLUX，若無則退回 SAP_FLUX；再做基本處理。"""
    lc = None
    
    # 嘗試獲取 PDCSAP_FLUX（使用新的 API）
    try:
        # 新版本 lightkurve 的用法
        if hasattr(lcf, 'PDCSAP_FLUX'):
            lc = lcf.PDCSAP_FLUX
        elif hasattr(lcf, 'get_lightcurve'):
            # 嘗試使用新的 API
            lc = lcf.get_lightcurve('PDCSAP_FLUX')
        else:
            # 舊版本用法
            lc = lcf.PDCSAP_FLUX
    except Exception as e:
        safe_print(f"    PDCSAP_FLUX 獲取失敗: {e}")
        lc = None
    
    # 如果 PDCSAP_FLUX 失敗，嘗試 SAP_FLUX
    if (lc is None) or (len(lc.flux) == 0):
        try:
            if hasattr(lcf, 'SAP_FLUX'):
                lc = lcf.SAP_FLUX
            elif hasattr(lcf, 'get_lightcurve'):
                lc = lcf.get_lightcurve('SAP_FLUX')
            else:
                lc = lcf.SAP_FLUX
        except Exception as e:
            safe_print(f"    SAP_FLUX 獲取失敗: {e}")
            lc = None
    
    if lc is None:
        raise FileNotFoundError("No usable PDCSAP/SAP light curve in file")

    # 基礎清理
    try:
        lc = lc.remove_nans()
    except Exception as e:
        safe_print(f"    移除 NaN 失敗: {e}")
        raise
    
    # 嘗試 flatten（可選）
    try:
        lc = lc.flatten(window_length=301)
    except Exception as e:
        safe_print(f"    Flatten 失敗，跳過: {e}")
        pass
    
    # 嘗試 normalize（可選）
    try:
        lc = lc.normalize()
    except Exception as e:
        safe_print(f"    Normalize 失敗，跳過: {e}")
        pass
    
    return lc


def apply_time_window_and_downsample(
    lc: "lk.LightCurve",
    transit_midpoint: Optional[float],
    window_days: Optional[float],
    downsample: Optional[int]
) -> "lk.LightCurve":
    """依 transit_midpoint 做 ±window_days 時窗截取，並可選下採樣。"""
    # 截取時間窗
    if transit_midpoint is not None and window_days and window_days > 0:
        t = lc.time.value.astype(float)
        mask = (t >= transit_midpoint - window_days) & (t <= transit_midpoint + window_days)
        if mask.any():
            lc = lc[mask]

    # 下採樣（均勻取樣）
    if downsample and downsample > 0 and len(lc.time.value) > downsample:
        idx = np.linspace(0, len(lc.time.value) - 1, downsample).astype(int)
        lc = lc[idx]

    return lc


def save_lightcurve_csv(lc: "lk.LightCurve", out_csv: Path):
    """將光變曲線保存為 CSV 格式，處理可能的欄位名稱問題"""
    try:
        df = lc.to_pandas()
        
        # 檢查並處理欄位名稱
        if 'time' not in df.columns:
            # 嘗試其他可能的时间欄位名稱
            time_cols = [col for col in df.columns if 'time' in col.lower()]
            if time_cols:
                df = df.rename(columns={time_cols[0]: 'time'})
            else:
                # 如果沒有找到時間欄位，使用索引
                df['time'] = df.index
        
        if 'flux' not in df.columns:
            # 嘗試其他可能的流量欄位名稱
            flux_cols = [col for col in df.columns if 'flux' in col.lower()]
            if flux_cols:
                df = df.rename(columns={flux_cols[0]: 'flux'})
            else:
                raise ValueError("找不到 flux 欄位")
        
        # 只保留 time 和 flux 欄位
        df_subset = df[["time", "flux"]].copy()
        
        # 移除 NaN 值
        df_subset = df_subset.dropna()
        
        if len(df_subset) == 0:
            raise ValueError("處理後的光變曲線數據為空")
            
        out_csv.write_text(df_subset.to_csv(index=False))
        
    except Exception as e:
        raise ValueError(f"保存光變曲線 CSV 失敗: {e}")


def download_one_target(
    source_id: str,
    lightcurve_id: str,
    source_telescope: str,
    out_dir: Path,
    transit_midpoint: Optional[float],
    window_days: Optional[float],
    downsample: Optional[int],
    retry: int = 1
) -> str:
    """下載單一目標的 LC → 處理 → 存 CSV。回傳 source_id。"""
    csv_path = out_dir / f"{source_telescope}_{lightcurve_id}.csv"
    if csv_path.exists():
        safe_print(f"  {source_telescope} {lightcurve_id} 已存在，跳過")
        return source_id

    try:
        lcf = safe_download_lcfile(lightcurve_id, source_telescope, out_dir, retries=retry)
        lc = lcfile_to_lightcurve(lcf)
        lc = apply_time_window_and_downsample(lc, transit_midpoint, window_days, downsample)
        save_lightcurve_csv(lc, csv_path)
        return source_id
    except Exception as e:
        # 清理可能損壞的文件
        if csv_path.exists():
            try:
                csv_path.unlink()
            except:
                pass
        raise


def bulk_download(
    targets: List[Dict[str, Any]],
    data_dir: str,
    max_workers: int = 4,
    window_days: Optional[float] = None,
    downsample: Optional[int] = None,
    retry: int = 1
) -> Tuple[List[str], List[str]]:
    """平行下載多個目標。"""
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok, fail = [], []
    total = len(targets)
    completed = 0
    
    safe_print(f"  開始平行下載 {total} 個目標，使用 {max_workers} 個線程...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {}
        for target in targets:
            fut = ex.submit(
                download_one_target,
                target["source_id"],
                target["lightcurve_id"],
                target["source_telescope"],
                out_dir,
                target["transit_midpoint"],
                window_days,
                downsample,
                retry
            )
            futs[fut] = target["source_id"]

        for fut in as_completed(futs):
            source_id = futs[fut]
            completed += 1
            try:
                fut.result()
                ok.append(source_id)
                safe_print(f"  [{completed}/{total}] ✓ {source_id} 成功")
            except Exception as e:
                safe_print(f"  [{completed}/{total}] ✗ {source_id} 失敗: {e}")
                fail.append(source_id)

    safe_print(f"  下載完成：成功 {len(ok)} 個，失敗 {len(fail)} 個")
    return ok, fail


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Preload light curves from exoplanet_data_processed.csv")
    p.add_argument("--processed_csv", required=True, type=str, help="exoplanet_data_processed.csv 路徑")
    p.add_argument("--data_dir", required=True, type=str, help="輸出資料夾（光變曲線 CSV 與 .fits 快取）")
    p.add_argument("--max_targets", type=int, default=50, help="最多處理幾個目標（預設 50）")
    p.add_argument("--max_workers", type=int, default=4, help="平行下載 worker 數（預設 4）")
    p.add_argument("--window_days", type=float, default=None, help="依 transit_midpoint 做 ±window_days 天的截取")
    p.add_argument("--downsample", type=int, default=None, help="每個目標最多保留多少點（均勻抽樣）")
    p.add_argument("--retry", type=int, default=1, help="壞檔刪除後重試次數（預設 1）")
    return p.parse_args()


def main():
    args = parse_args()
    processed_csv = Path(args.processed_csv)
    data_dir = Path(args.data_dir)

    safe_print("[1/3] 讀取處理過的資料 ...")
    df = load_processed_data(processed_csv)
    sub = select_labeled_samples(df, max_targets=args.max_targets)
    safe_print(f"  樣本數（含標籤）：{len(df[df['disposition'].isin(['CONFIRMED','FALSE POSITIVE'])])}，本次下載：{len(sub)}")

    # 準備目標列表
    targets = []
    for _, row in sub.iterrows():
        target = {
            "source_id": row["source_id"],
            "lightcurve_id": str(row["lightcurve_id"]),
            "source_telescope": row["source_telescope"],
            "transit_midpoint": float(row["transit_midpoint"]) if pd.notna(row["transit_midpoint"]) else None
        }
        targets.append(target)

    safe_print("[2/3] 下載光變曲線數據 ...")
    ok, fail = bulk_download(
        targets,
        data_dir=str(data_dir),
        max_workers=args.max_workers,
        window_days=args.window_days,
        downsample=args.downsample,
        retry=args.retry
    )

    safe_print("[3/3] 完成")
    safe_print(f"  成功：{len(ok)} ；失敗：{len(fail)}")
    if fail:
        safe_print("  失敗目標：", fail)


if __name__ == "__main__":
    main()