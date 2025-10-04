# 光變曲線數據預載系統

## 概述

`preload_data.py` 是系外行星項目中的光變曲線數據下載和預處理工具。該工具能夠從處理過的系外行星數據中提取目標信息，並批量下載對應的克普勒 (Kepler) 和 TESS 光變曲線數據，為後續的特徵萃取和機器學習提供標準化的數據源。

## 🎯 核心功能

### 1. 智能數據下載
- **多源支持**: 同時支持克普勒 (KOI) 和 TESS 數據
- **並行下載**: 多線程同時下載多個目標
- **智能重試**: 自動處理損壞文件和網路錯誤
- **快取機制**: 避免重複下載相同數據

### 2. 數據預處理
- **格式標準化**: 統一輸出 CSV 格式
- **時間窗口截取**: 基於凌日中點的時間窗口截取
- **下採樣**: 可選的數據點下採樣
- **品質控制**: 自動過濾低品質數據

### 3. 錯誤處理
- **損壞文件清理**: 自動檢測和刪除損壞的 FITS 文件
- **網路重試**: 處理網路連接問題
- **線程安全**: 多線程環境下的安全輸出

## 📋 系統要求

### 依賴套件
```bash
pip install lightkurve pandas numpy tqdm
```

### 硬體要求
- **記憶體**: 建議 4GB+ RAM
- **儲存**: 每個光變曲線約 1-10MB
- **網路**: 需要穩定的網路連接

## 🚀 使用方法

### 基本命令
```bash
python preload_data.py \
    --processed_csv data/exoplanet_data_processed.csv \
    --data_dir ./lightkurve_data \
    --max_targets 100 \
    --max_workers 4
```

### 參數說明

| 參數 | 類型 | 必需 | 預設值 | 說明 |
|------|------|------|--------|------|
| `--processed_csv` | str | ✓ | - | 處理過的數據 CSV 文件路徑 |
| `--data_dir` | str | ✓ | - | 輸出數據目錄 |
| `--max_targets` | int | ✗ | 50 | 最大處理目標數 |
| `--max_workers` | int | ✗ | 4 | 並行下載線程數 |
| `--window_days` | float | ✗ | None | 時間窗口（天） |
| `--downsample` | int | ✗ | None | 下採樣點數 |
| `--retry` | int | ✗ | 1 | 重試次數 |

### 使用範例

#### 1. 基本下載
```bash
python preload_data.py \
    --processed_csv data/exoplanet_data_processed.csv \
    --data_dir ./lightkurve_data \
    --max_targets 50
```

#### 2. 高效並行下載
```bash
python preload_data.py \
    --processed_csv data/exoplanet_data_processed.csv \
    --data_dir ./lightkurve_data \
    --max_targets 200 \
    --max_workers 8
```

#### 3. 時間窗口截取
```bash
python preload_data.py \
    --processed_csv data/exoplanet_data_processed.csv \
    --data_dir ./lightkurve_data \
    --window_days 30 \
    --downsample 10000
```

#### 4. 大量數據處理
```bash
python preload_data.py \
    --processed_csv data/exoplanet_data_processed.csv \
    --data_dir ./lightkurve_data \
    --max_targets 1000 \
    --max_workers 16 \
    --downsample 20000
```

## 🔧 核心函數說明

### 數據讀取函數

#### `load_processed_data(csv_path: Path) -> pd.DataFrame`
**功能**: 讀取處理過的系外行星數據
**參數**:
- `csv_path`: CSV 文件路徑
**返回**: 包含所有處理後數據的 DataFrame
**特點**: 自動處理編碼和格式問題

#### `select_labeled_samples(df: pd.DataFrame, max_targets: Optional[int]) -> pd.DataFrame`
**功能**: 選擇標記的樣本並建立目標列表
**參數**:
- `df`: 原始數據框
- `max_targets`: 最大目標數（可選）
**返回**: 包含必要欄位的樣本 DataFrame
**保留欄位**:
- `source_id`: 數據源 ID
- `source_telescope`: 望遠鏡類型 (KOI/TESS)
- `disposition`: 行星狀態
- `lightcurve_id`: 光變曲線 ID
- `period`: 軌道週期
- `transit_midpoint`: 凌日中點時間

### 光變曲線下載函數

#### `_download_kepler_lc(lightcurve_id: str, download_dir: Path)`
**功能**: 下載克普勒光變曲線
**參數**:
- `lightcurve_id`: 克普勒星體 ID
- `download_dir`: 下載目錄
**返回**: LightCurveFile 對象
**特點**:
- 使用 KIC ID 檢索
- 優先選擇 long-cadence 數據
- 自動處理多個季度的數據

#### `_download_tess_lc(lightcurve_id: str, download_dir: Path)`
**功能**: 下載 TESS 光變曲線
**參數**:
- `lightcurve_id`: TESS 星體 ID
- `download_dir`: 下載目錄
**返回**: LightCurveFile 對象
**特點**:
- 使用 TIC ID 檢索
- 支持多個 Sector 的數據
- 自動處理不同觀測模式

#### `safe_download_lcfile(lightcurve_id: str, source_telescope: str, download_dir: Path, retries: int)`
**功能**: 安全下載光變曲線文件，包含錯誤處理
**參數**:
- `lightcurve_id`: 光變曲線 ID
- `source_telescope`: 望遠鏡類型
- `download_dir`: 下載目錄
- `retries`: 重試次數
**返回**: LightCurveFile 對象
**錯誤處理**:
- 自動檢測損壞的 FITS 文件
- 清理損壞文件後重試
- 處理網路連接問題

### 數據處理函數

#### `lcfile_to_lightcurve(lcf) -> lk.LightCurve`
**功能**: 從 LightCurveFile 提取可用的光變曲線
**參數**:
- `lcf`: LightCurveFile 對象
**返回**: 處理後的 LightCurve 對象
**處理流程**:
1. 優先選擇 PDCSAP_FLUX（去除系統性趨勢）
2. 回退到 SAP_FLUX（原始光度）
3. 移除 NaN 值
4. 可選的平坦化處理
5. 可選的正規化處理

#### `apply_time_window_and_downsample(lc: lk.LightCurve, transit_midpoint: Optional[float], window_days: Optional[float], downsample: Optional[int]) -> lk.LightCurve`
**功能**: 應用時間窗口截取和下採樣
**參數**:
- `lc`: 原始光變曲線
- `transit_midpoint`: 凌日中點時間
- `window_days`: 時間窗口（天）
- `downsample`: 下採樣點數
**返回**: 處理後的光變曲線
**處理邏輯**:
- 時間窗口: `[transit_midpoint - window_days, transit_midpoint + window_days]`
- 下採樣: 均勻選擇指定數量的數據點

#### `save_lightcurve_csv(lc: lk.LightCurve, out_csv: Path)`
**功能**: 將光變曲線保存為 CSV 格式
**參數**:
- `lc`: LightCurve 對象
- `out_csv`: 輸出 CSV 文件路徑
**輸出格式**:
- 欄位: `time`, `flux`
- 編碼: UTF-8
- 格式: CSV (無索引)

### 批量處理函數

#### `download_one_target(source_id: str, lightcurve_id: str, source_telescope: str, out_dir: Path, transit_midpoint: Optional[float], window_days: Optional[float], downsample: Optional[int], retry: int) -> str`
**功能**: 下載單一目標的完整流程
**參數**:
- `source_id`: 數據源 ID
- `lightcurve_id`: 光變曲線 ID
- `source_telescope`: 望遠鏡類型
- `out_dir`: 輸出目錄
- `transit_midpoint`: 凌日中點時間
- `window_days`: 時間窗口
- `downsample`: 下採樣點數
- `retry`: 重試次數
**返回**: 成功處理的 source_id
**輸出文件**: `{source_telescope}_{lightcurve_id}.csv`

#### `bulk_download(targets: List[Dict[str, Any]], data_dir: str, max_workers: int, window_days: Optional[float], downsample: Optional[int], retry: int) -> Tuple[List[str], List[str]]`
**功能**: 批量並行下載多個目標
**參數**:
- `targets`: 目標列表
- `data_dir`: 數據目錄
- `max_workers`: 最大線程數
- `window_days`: 時間窗口
- `downsample`: 下採樣點數
- `retry`: 重試次數
**返回**: (成功列表, 失敗列表)
**特點**:
- 使用 ThreadPoolExecutor 並行處理
- 線程安全的進度輸出
- 自動錯誤處理和重試

## 📊 數據處理流程

### 1. 數據準備階段
```python
# 讀取處理過的數據
df = load_processed_data(processed_csv)

# 選擇標記樣本
sub = select_labeled_samples(df, max_targets)

# 建立目標列表
targets = []
for _, row in sub.iterrows():
    target = {
        "source_id": row["source_id"],
        "lightcurve_id": str(row["lightcurve_id"]),
        "source_telescope": row["source_telescope"],
        "transit_midpoint": float(row["transit_midpoint"]) if pd.notna(row["transit_midpoint"]) else None
    }
    targets.append(target)
```

### 2. 並行下載階段
```python
# 使用線程池並行下載
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {}
    for target in targets:
        future = executor.submit(
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
        futures[future] = target["source_id"]
```

### 3. 結果處理階段
```python
# 收集成功和失敗的結果
ok, fail = [], []
for future in as_completed(futures):
    source_id = futures[future]
    try:
        future.result()
        ok.append(source_id)
    except Exception as e:
        fail.append(source_id)
```

## 🛠️ 錯誤處理機制

### 1. 損壞文件處理
```python
def _find_bad_fits_path_from_error(msg: str) -> Optional[Path]:
    """從錯誤訊息中找出損壞的 FITS 文件路徑"""
    patterns = [
        r"(/[^\\\s]+?\.fits)",  # 基本 .fits 文件
        r"(lightkurve_data/mastDownload/[^\\\s]+?\.fits)",  # 完整路徑
        r"(mastDownload/[^\\\s]+?\.fits)"  # 相對路徑
    ]
    
    for pattern in patterns:
        match = re.search(pattern, msg)
        if match:
            path = Path(match.group(1))
            if path.exists():
                return path
    return None
```

### 2. 自動重試機制
```python
def safe_download_lcfile(lightcurve_id: str, source_telescope: str, download_dir: Path, retries: int):
    """安全下載，包含重試機制"""
    try:
        # 嘗試下載
        return _download_lc(lightcurve_id, source_telescope, download_dir)
    except Exception as e:
        if retries <= 0:
            raise
        
        # 清理損壞文件
        bad_file = _find_bad_fits_path_from_error(str(e))
        if bad_file and bad_file.exists():
            bad_file.unlink()
            # 清理空目錄
            parent_dir = bad_file.parent
            if parent_dir.exists() and not any(parent_dir.iterdir()):
                parent_dir.rmdir()
        
        # 重試
        return safe_download_lcfile(lightcurve_id, source_telescope, download_dir, retries - 1)
```

### 3. 線程安全輸出
```python
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
```

## 📈 性能優化

### 1. 並行處理
- **多線程下載**: 同時下載多個目標
- **線程池管理**: 自動管理線程生命週期
- **負載平衡**: 根據系統資源調整線程數

### 2. 記憶體優化
- **流式處理**: 逐個處理目標，避免大量數據載入記憶體
- **及時清理**: 處理完成後立即釋放資源
- **快取策略**: 智能快取下載的數據

### 3. 網路優化
- **重試機制**: 處理網路不穩定問題
- **錯誤恢復**: 自動清理損壞文件
- **進度追蹤**: 實時顯示下載進度

## 📁 輸出文件格式

### 光變曲線 CSV 格式
```csv
time,flux
2454833.0,1.000000
2454833.5,0.999500
2454834.0,1.000200
...
```

### 文件命名規則
- **克普勒數據**: `KOI_{lightcurve_id}.csv`
- **TESS 數據**: `TESS_{lightcurve_id}.csv`

### 目錄結構
```
lightkurve_data/
├── KOI_123456.csv
├── KOI_123457.csv
├── TESS_987654.csv
├── TESS_987655.csv
└── mastDownload/  # LightKurve 快取目錄
    ├── Kepler/
    └── TESS/
```

## 🔍 故障排除

### 常見問題

#### 1. 下載失敗
```bash
# 錯誤: FileNotFoundError: No Kepler LCF for KIC xxx
# 解決: 檢查星體 ID 是否正確，或該星體是否在克普勒觀測範圍內
```

#### 2. 記憶體不足
```bash
# 錯誤: MemoryError
# 解決: 減少 max_workers 參數，或增加系統記憶體
```

#### 3. 網路問題
```bash
# 錯誤: ConnectionError
# 解決: 檢查網路連接，或使用 --retry 增加重試次數
```

### 調試技巧

#### 1. 啟用詳細輸出
```python
# 在函數中添加調試信息
safe_print(f"正在下載 {source_telescope} {lightcurve_id}")
safe_print(f"目標目錄: {out_dir}")
```

#### 2. 檢查數據品質
```python
# 檢查下載的數據
df = pd.read_csv(csv_path)
print(f"數據點數: {len(df)}")
print(f"時間範圍: {df['time'].min()} - {df['time'].max()}")
print(f"流量範圍: {df['flux'].min()} - {df['flux'].max()}")
```

#### 3. 逐步測試
```bash
# 先測試少量樣本
python preload_data.py --max_targets 5
```

## 🚀 進階使用

### 1. 自定義時間窗口
```python
# 根據軌道週期調整時間窗口
window_days = period * 2  # 2 個軌道週期
```

### 2. 智能下採樣
```python
# 根據數據密度調整下採樣
if len(lc.time) > 50000:
    downsample = 20000
elif len(lc.time) > 10000:
    downsample = 5000
else:
    downsample = None
```

### 3. 批量處理腳本
```bash
#!/bin/bash
# 分批處理大量數據
for batch in {1..10}; do
    python preload_data.py \
        --processed_csv data/exoplanet_data_processed.csv \
        --data_dir ./lightkurve_data_batch_$batch \
        --max_targets 100 \
        --max_workers 8
done
```

## 📚 相關工具

### 與其他組件的整合

#### 1. 數據預處理
```bash
# 先運行數據預處理
cd data/
python data_preprocessing_augment.py

# 再下載光變曲線
python ../preload_data.py \
    --processed_csv exoplanet_data_processed.csv \
    --data_dir ../lightkurve_data
```

#### 2. 特徵萃取
```bash
# 使用下載的光變曲線進行特徵萃取
python tsfresh_LightGBM.py \
    --processed_csv data/exoplanet_data_processed.csv \
    --out_dir ./results \
    --lc_dir ./lightkurve_data \
    --offline
```

### 數據品質檢查
```python
# 檢查下載的數據品質
import pandas as pd
import numpy as np

def check_lightcurve_quality(csv_path):
    df = pd.read_csv(csv_path)
    
    # 基本統計
    print(f"數據點數: {len(df)}")
    print(f"時間範圍: {df['time'].max() - df['time'].min():.2f} 天")
    print(f"流量標準差: {df['flux'].std():.6f}")
    
    # 檢查異常值
    flux_std = df['flux'].std()
    flux_mean = df['flux'].mean()
    outliers = np.abs(df['flux'] - flux_mean) > 3 * flux_std
    print(f"異常值數量: {outliers.sum()}")
    
    return len(df) > 100 and outliers.sum() < len(df) * 0.1
```

---

**注意**: 本工具需要穩定的網路連接和充足的儲存空間。建議在網路條件良好的環境下運行，並定期檢查下載的數據品質。
