# -*- coding: utf-8 -*-
"""
make_evidence.py
產出 5 種證據圖與表,證明 K2 不適合納入本專案訓練/驗證。
usage:
  python make_evidence.py --std_csv standardized_all.csv --out_dir ./evidence_out
"""
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

# --------- 參數 ---------
CORE = ["planet_period","planet_transit_epoch","planet_transit_duration",
        "planet_transit_depth","planet_radius","star_eff_temp","star_logg",
        "star_radius","ra_deg","dec_deg","Vmag"]
KOI_ONLY = ["disposition_score","fp_flag_nt","fp_flag_ss","fp_flag_co","fp_flag_ec","disposition_pipeline"]

# 欄位中文對應
FIELD_CN = {
    "planet_period": "軌道週期",
    "planet_transit_epoch": "凌日時刻",
    "planet_transit_duration": "凌日持續時間",
    "planet_transit_depth": "凌日深度",
    "planet_radius": "行星半徑",
    "star_eff_temp": "恆星有效溫度",
    "star_logg": "恆星表面重力",
    "star_radius": "恆星半徑",
    "ra_deg": "赤經",
    "dec_deg": "赤緯",
    "Vmag": "星等",
    "disposition_score": "分類分數",
    "fp_flag_nt": "非凌日標記",
    "fp_flag_ss": "次食標記",
    "fp_flag_co": "質心偏移標記",
    "fp_flag_ec": "星曆匹配標記",
    "disposition_pipeline": "管線分類"
}

SHIFT_FEATS = ["planet_period","planet_transit_depth","planet_radius","star_eff_temp","star_radius","star_logg","Vmag"]
RELERR_SPECS = [
    ("planet_radius","planet_radius_err_pos","planet_radius_err_neg","e3_relerr_planet_radius.png"),
    ("star_radius","star_radius_err_pos","star_radius_err_neg","e3_relerr_star_radius.png"),
    ("planet_transit_duration","planet_transit_duration_err_pos","planet_transit_duration_err_neg","e3_relerr_trandur.png"),
    ("planet_transit_depth","planet_transit_depth_err_pos","planet_transit_depth_err_neg","e3_relerr_trandep.png"),
]

# --------- 輔助 ---------
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def wasserstein_1d(x, y):
    x = np.sort(np.asarray(x, float)); y = np.sort(np.asarray(y, float))
    if len(x)==0 or len(y)==0: return np.nan
    grid = np.union1d(x, y)
    Fx = np.searchsorted(x, grid, side="right")/len(x)
    Fy = np.searchsorted(y, grid, side="right")/len(y)
    steps = np.diff(np.concatenate([grid, grid[-1:]]))
    return np.sum(np.abs(Fx - Fy) * steps)

def log1p_pos(a):
    a = pd.to_numeric(a, errors="coerce")
    a = a[a>0]
    return np.log10(a)

def rel_err(df, val, epos, eneg):
    v = pd.to_numeric(df.get(val), errors="coerce").abs()
    ep = pd.to_numeric(df.get(epos), errors="coerce").abs()
    en = pd.to_numeric(df.get(eneg), errors="coerce").abs()
    r = (ep + en) / (2.0 * v)
    r = r.replace([np.inf,-np.inf], np.nan).dropna()
    return r.values

def bar_save(xlabels, series_dict, title, out_png, ylabel="", title_cn="", ylabel_cn=""):
    """繪製長條圖，同時生成中英文版本"""
    x = np.arange(len(xlabels))
    keys = list(series_dict.keys())
    width = 0.8 / max(1,len(keys))
    
    # 中文標籤對應
    xlabels_cn = [FIELD_CN.get(label, label) for label in xlabels]
    
    # # 生成雙語版本（原本的版本）
    # fig = plt.figure(figsize=(14,7))
    # for i,k in enumerate(keys):
    #     plt.bar(x + i*width, series_dict[k], width, label=k, alpha=0.8)
    # 
    # plt.xticks(x + (len(keys)-1)*width/2, xlabels_cn, rotation=45, ha="right", fontsize=10)
    # 
    # if ylabel_cn:
    #     plt.ylabel(ylabel_cn, fontsize=12)
    # elif ylabel:
    #     plt.ylabel(ylabel, fontsize=12)
    # 
    # if title_cn:
    #     plt.title(f"{title_cn}\n{title}", fontsize=14, pad=15)
    # else:
    #     plt.title(title, fontsize=14, pad=15)
    # 
    # plt.legend(fontsize=11)
    # plt.grid(axis='y', alpha=0.3, linestyle='--')
    # plt.tight_layout()
    # plt.savefig(out_png, dpi=150, bbox_inches='tight')
    # plt.close(fig)
    
    # # 生成純中文版本
    # if title_cn:
    #     fig_cn = plt.figure(figsize=(14,7))
    #     for i,k in enumerate(keys):
    #         plt.bar(x + i*width, series_dict[k], width, label=k, alpha=0.8)
    #     
    #     plt.xticks(x + (len(keys)-1)*width/2, xlabels_cn, rotation=45, ha="right", fontsize=10)
    #     plt.ylabel(ylabel_cn if ylabel_cn else ylabel, fontsize=12)
    #     plt.title(title_cn, fontsize=14, pad=15)
    #     plt.legend(fontsize=11)
    #     plt.grid(axis='y', alpha=0.3, linestyle='--')
    #     plt.tight_layout()
    #     out_png_cn = str(out_png).replace('.png', '_cn.png')
    #     plt.savefig(out_png_cn, dpi=150, bbox_inches='tight')
    #     plt.close(fig_cn)
    
    # 生成純英文版本
    fig_en = plt.figure(figsize=(14,7))
    for i,k in enumerate(keys):
        plt.bar(x + i*width, series_dict[k], width, label=k, alpha=0.8)
    
    plt.xticks(x + (len(keys)-1)*width/2, xlabels, rotation=45, ha="right", fontsize=10)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, pad=15)
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    out_png_en = str(out_png).replace('.png', '_en.png')
    plt.savefig(out_png_en, dpi=150, bbox_inches='tight')
    plt.close(fig_en)

# --------- 主流程 ---------
def main(std_csv, out_dir):
    ensure_dir(out_dir)
    df = pd.read_csv(std_csv, engine="c", comment="#", low_memory=False)
    if "__dataset__" not in df.columns:
        raise ValueError("需要欄位 __dataset__(值必須為 KOI/TESS/K2)")

    ds_vals = ["KOI","TESS","K2"]
    # ========== (1) 缺失率 ==========
    print("\n" + "="*60)
    print("證據 (1): 欄位完備度與一致性")
    print("="*60)
    
    miss_tbl = []
    for ds in ds_vals:
        g = df[df["__dataset__"]==ds]
        if g.empty: continue
        m = g[CORE].isna().mean().rename("missing_rate").to_frame()
        m["dataset"] = ds
        miss_tbl.append(m)
    miss_df = pd.concat(miss_tbl).reset_index().rename(columns={"index":"feature"})
    miss_df.to_csv(Path(out_dir,"e1_missingness.csv"), index=False)
    print(f"已儲存: e1_missingness.csv")
    
    # plot
    pv = miss_df.pivot(index="feature", columns="dataset", values="missing_rate").reindex(CORE)
    bar_save(pv.index.tolist(),
             {c: pv[c].values for c in pv.columns if c in ["KOI","TESS","K2"]},
             "Evidence (1): Core feature missingness", 
             Path(out_dir,"e1_missingness.png"),
             ylabel="Missing rate",
             title_cn="證據 (1): 核心欄位缺失率比較",
             ylabel_cn="缺失率 (0-1)")
    print(f"已儲存: e1_missingness.png")

    # # KOI-only 可用率
    # koi = df[df["__dataset__"]=="KOI"]
    # if not koi.empty:
    #     available_koi_cols = [col for col in KOI_ONLY if col in koi.columns]
    #     if available_koi_cols:
    #         koi_avail = koi[available_koi_cols].notna().mean().sort_values(ascending=False)
    #         bar_save(koi_avail.index.tolist(), {"KOI": koi_avail.values},
    #                  "Evidence (1): KOI-only high-value fields availability",
    #                  Path(out_dir, "e1_koi_only_availability.png"),
    #                  ylabel="Availability",
    #                  title_cn="證據 (1): KOI 專屬高價值欄位可用率",
    #                  ylabel_cn="可用率 (0-1)")
    #         print(f"已儲存: e1_koi_only_availability.png")

    #     # # ========== (2) 分佈相容性 ==========
    # print("\n" + "="*60)
    # print("證據 (2): 分佈相容性 (Covariate Shift)")
    # print("="*60)
    # 
    # rows = []
    # both = df[df["__dataset__"].isin(["KOI","TESS"])].copy()
    # for f in SHIFT_FEATS:
    #     if f not in df.columns:
    #         continue
    #     # KOI vs TOI
    #     a = df[df["__dataset__"]=="KOI"][f]
    #     b = df[df["__dataset__"]=="TESS"][f]
    #     A = log1p_pos(a) if f in ["planet_period","planet_transit_depth","planet_radius","Vmag"] else pd.to_numeric(a, errors="coerce").dropna()
    #     B = log1p_pos(b) if f in ["planet_period","planet_transit_depth","planet_radius","Vmag"] else pd.to_numeric(b, errors="coerce").dropna()
    #     w = wasserstein_1d(A,B) if (len(A)>=50 and len(B)>=50) else np.nan
    #     p = ks_2samp(A,B).pvalue if (len(A)>=50 and len(B)>=50) else np.nan
    #     rows.append({"pair":"KOI_vs_TOI","feature":f,"wasserstein":w,"ks_p":p})
    # 
    #     # K2 vs KOI∪TOI
    #     a2 = df[df["__dataset__"]=="K2"][f]
    #     b2 = both[f]
    #     A2 = log1p_pos(a2) if f in ["planet_period","planet_transit_depth","planet_radius","Vmag"] else pd.to_numeric(a2, errors="coerce").dropna()
    #     B2 = log1p_pos(b2) if f in ["planet_period","planet_transit_depth","planet_radius","Vmag"] else pd.to_numeric(b2, errors="coerce").dropna()
    #     w2 = wasserstein_1d(A2,B2) if (len(A2)>=50 and len(B2)>=50) else np.nan
    #     p2 = ks_2samp(A2,B2).pvalue if (len(A2)>=50 and len(B2)>=50) else np.nan
    #     rows.append({"pair":"K2_vs_KOI_TOI","feature":f,"wasserstein":w2,"ks_p":p2})
    # 
    # shift_df = pd.DataFrame(rows)
    # shift_df.to_csv(Path(out_dir,"e2_shift_metrics.csv"), index=False)
    # print(f"已儲存: e2_shift_metrics.csv")
    # 
    # # 中文對應
    # pair_cn = {
    #     "KOI_vs_TOI": "KOI vs TOI 分佈對比",
    #     "K2_vs_KOI_TOI": "K2 vs KOI∪TOI 分佈對比"
    # }
    # 
    # for pair in ["KOI_vs_TOI","K2_vs_KOI_TOI"]:
    #     sub = shift_df[shift_df["pair"]==pair]
    #     bar_save(sub["feature"].tolist(), {"Wasserstein 距離": sub["wasserstein"].tolist()},
    #              f"Evidence (2): Covariate shift — {pair}",
    #              Path(out_dir, f"e2_shift_{pair}.png"),
    #              ylabel="Wasserstein distance",
    #              title_cn=f"證據 (2): 分佈差異 — {pair_cn[pair]}",
    #              ylabel_cn="Wasserstein 距離 (數值越大代表分佈差異越大)")
    #     print(f"已儲存: e2_shift_{pair}.png")

    # # ========== (3) 相對誤差(異方差) ==========
    
    # # 中文對應
    # pair_cn = {
    #     "KOI_vs_TOI": "KOI vs TOI 分佈對比",
    #     "K2_vs_KOI_TOI": "K2 vs KOI∪TOI 分佈對比"
    # }
    # 
    # for pair in ["KOI_vs_TOI","K2_vs_KOI_TOI"]:
    #     sub = shift_df[shift_df["pair"]==pair]
    #     bar_save(sub["feature"].tolist(), {"Wasserstein 距離": sub["wasserstein"].tolist()},
    #              f"Evidence (2): Covariate shift — {pair}",
    #              Path(out_dir, f"e2_shift_{pair}.png"),
    #              ylabel="Wasserstein distance",
    #              title_cn=f"證據 (2): 分佈差異 — {pair_cn[pair]}",
    #              ylabel_cn="Wasserstein 距離 (數值越大代表分佈差異越大)")
    #     print(f"已儲存: e2_shift_{pair}.png")

    #     # # ========== (3) 相對誤差(異方差) ==========
    # print("\n" + "="*60)
    # print("證據 (3): 測量精度與異方差")
    # print("="*60)
    # 
    # # 欄位中文名稱
    # val_cn = {
    #     "planet_radius": "行星半徑",
    #     "star_radius": "恆星半徑",
    #     "planet_transit_duration": "凌日持續時間",
    #     "planet_transit_depth": "凌日深度"
    # }
    # 
    # for val,ep,en,fn in RELERR_SPECS:
    #     if val not in df.columns:
    #         continue
    #     data = {}; ok=False
    #     for ds in ds_vals:
    #         g = df[df["__dataset__"]==ds]
    #         if g.empty: continue
    #         arr = rel_err(g, val, ep, en)
    #         if len(arr) >= 30:
    #             data[ds] = arr; ok=True
    #     if not ok: continue
    #     
    #     # 生成雙語版本（原本的版本）
    #     fig = plt.figure(figsize=(8,6))
    #     labels = list(data.keys())
    #     bp = plt.boxplot([data[k] for k in labels], labels=labels, showmeans=True, 
    #                      patch_artist=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
    #     
    #     colors = ['lightblue', 'lightgreen', 'lightyellow']
    #     for patch, color in zip(bp['boxes'], colors[:len(labels)]):
    #         patch.set_facecolor(color)
    #         patch.set_alpha(0.7)
    #     
    #     plt.ylabel("相對誤差 = (|上誤差|+|下誤差|)/(2×|數值|)", fontsize=11)
    #     plt.xlabel("資料集", fontsize=11)
    #     plt.title(f"證據 (3): 測量精度比較 — {val_cn.get(val, val)}\nEvidence (3): Relative uncertainty — {val}", 
    #               fontsize=13, pad=15)
    #     plt.grid(axis='y', alpha=0.3, linestyle='--')
    #     plt.tight_layout()
    #     plt.savefig(Path(out_dir, fn), dpi=150, bbox_inches='tight')
    #     plt.close(fig)
    #     print(f"已儲存: {fn}")
    #     
    #     # 生成純中文版本
    #     fig_cn = plt.figure(figsize=(8,6))
    #     bp = plt.boxplot([data[k] for k in labels], labels=labels, showmeans=True, 
    #                      patch_artist=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
    #     
    #     for patch, color in zip(bp['boxes'], colors[:len(labels)]):
    #         patch.set_facecolor(color)
    #         patch.set_alpha(0.7)
    #     
    #     plt.ylabel("相對誤差 = (|上誤差|+|下誤差|)/(2×|數值|)", fontsize=11)
    #     plt.xlabel("資料集", fontsize=11)
    #     plt.title(f"證據 (3): 測量精度比較 — {val_cn.get(val, val)}", fontsize=13, pad=15)
    #     plt.grid(axis='y', alpha=0.3, linestyle='--')
    #     plt.tight_layout()
    #     fn_cn = fn.replace('.png', '_cn.png')
    #     plt.savefig(Path(out_dir, fn_cn), dpi=150, bbox_inches='tight')
    #     plt.close(fig_cn)
    #     print(f"已儲存: {fn_cn}")
    #     
    #     # 生成純英文版本
    #     fig_en = plt.figure(figsize=(8,6))
    #     bp = plt.boxplot([data[k] for k in labels], labels=labels, showmeans=True, 
    #                      patch_artist=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
    #     
    #     for patch, color in zip(bp['boxes'], colors[:len(labels)]):
    #         patch.set_facecolor(color)
    #         patch.set_alpha(0.7)
    #     
    #     plt.ylabel("Relative uncertainty = (|upper err|+|lower err|)/(2×|value|)", fontsize=11)
    #     plt.xlabel("Dataset", fontsize=11)
    #     plt.title(f"Evidence (3): Relative uncertainty — {val}", fontsize=13, pad=15)
    #     plt.grid(axis='y', alpha=0.3, linestyle='--')
    #     plt.tight_layout()
    #     fn_en = fn.replace('.png', '_en.png')
    #     plt.savefig(Path(out_dir, fn_en), dpi=150, bbox_inches='tight')
    #     plt.close(fig_en)
    #     print(f"已儲存: {fn_en}")

    # # ========== (4) 外部驗證(可選,若有標籤就做) ==========

    # # ========== (4) 外部驗證(可選,若有標籤就做) ==========
    # print("\n" + "="*60)
    # print("證據 (4): 模型外部驗證")
    # print("="*60)
    # 
    # have_label = "disposition" in df.columns or "disposition_pipeline" in df.columns
    if False:  # have_label:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss
        try:
            import lightgbm as lgb
        except Exception:
            lgb = None
            print("警告: 未安裝 lightgbm,跳過模型訓練")

        if lgb is not None:
            # y 定義
            if "disposition" in df.columns:
                lab = df["disposition"].astype(str).str.upper()
                y = lab.map({"CONFIRMED":1, "FALSE POSITIVE":0})
            else:
                lab = df["disposition_pipeline"].astype(str).str.upper()
                y = lab.map({"CANDIDATE":1, "FALSE POSITIVE":0})
            
            # 過濾掉沒有標籤的資料（TESS 大部分沒有 disposition）
            valid_mask = y.notna()
            print(f"有效標籤數: {valid_mask.sum()} / {len(df)} ({valid_mask.mean():.1%})")
            
            dfY = pd.DataFrame({"__dataset__": df["__dataset__"], "_y": y})[valid_mask]
            data = df[valid_mask].copy()

            # 選特徵（移除 Vmag（在不同任務使用不同波段）和 planet_transit_depth（K2 100% 缺失））
            FEATS = ["planet_period","planet_radius","star_eff_temp","star_logg","star_radius"]
            FEATS = [f for f in FEATS if f in df.columns]
            X = data[FEATS].apply(pd.to_numeric, errors="coerce")
            M = pd.concat([X, dfY], axis=1).dropna()
            
            print(f"刪除缺失值後資料筆數: {len(M)}")
            print(f"資料集分佈: {M['__dataset__'].value_counts().to_dict()}")
            
            if len(M) < 500:
                print(f"警告: 有效資料不足 ({len(M)} 筆)，跳過模型訓練")
            else:
                def run_exp(train_mask, test_mask, tag):
                    Xtr, ytr = M.loc[train_mask, FEATS], M.loc[train_mask, "_y"].astype(int)
                    Xte, yte = M.loc[test_mask, FEATS], M.loc[test_mask, "_y"].astype(int)
                    if len(Xtr)<200 or len(Xte)<200: 
                        print(f"  {tag}: 樣本不足 (Train:{len(Xtr)}, Test:{len(Xte)})")
                        return None
                    print(f"  {tag}: 訓練中... (Train:{len(Xtr)}, Test:{len(Xte)})")
                    model = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, max_depth=-1, subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1)
                    model.fit(Xtr, ytr)
                    p = model.predict_proba(Xte)[:,1]
                    ypred = (p>=0.5).astype(int)
                    out = {
                        "tag": tag,
                        "ROC_AUC": roc_auc_score(yte, p),
                        "PR_AUC": average_precision_score(yte, p),
                        "F1": f1_score(yte, ypred),
                        "Brier(ECE proxy)": brier_score_loss(yte, p)
                    }
                    return out

                # 實驗設計：比較「有/無 K2」對模型在 TOI 上的泛化能力
                # A: 只用 KOI 訓練 → TOI 測試
                A = run_exp((M["__dataset__"]=="KOI"),
                            (M["__dataset__"]=="TESS"),
                            "A: Train=KOI, Test=TOI")
                
                # B: KOI+TOI 訓練 → TOI holdout 測試（防止 overfitting）
                toi_data = M[M["__dataset__"]=="TESS"]
                koi_data = M[M["__dataset__"]=="KOI"]
                if len(toi_data) >= 400 and len(koi_data) >= 200:
                    toi_test_idx = toi_data.sample(frac=0.3, random_state=42).index
                    trainB_mask = (M["__dataset__"]=="KOI") | (M.index.isin(toi_data.index.difference(toi_test_idx)))
                    B = run_exp(trainB_mask,
                                M.index.isin(toi_test_idx),
                                "B: Train=KOI+TOI(70%), Test=TOI(30%)")
                else:
                    B = None
                
                # C: KOI+K2+TOI 訓練 → TOI holdout 測試（檢驗 K2 是否有幫助）
                if len(toi_data) >= 400:
                    toi_test_idx = toi_data.sample(frac=0.3, random_state=42).index
                    trainC_mask = (M["__dataset__"].isin(["KOI","K2"])) | (M.index.isin(toi_data.index.difference(toi_test_idx)))
                    C = run_exp(trainC_mask,
                                M.index.isin(toi_test_idx),
                                "C: Train=All(KOI+K2+TOI), Test=TOI(30%)")
                else:
                    C = None

                res = [r for r in [A,B,C] if r is not None]
                if res:
                    res_df = pd.DataFrame(res)
                    res_df.to_csv(Path(out_dir,"e4_generalization_metrics.csv"), index=False)
                    print(f"已儲存: e4_generalization_metrics.csv")
                    
                    # 畫每個指標的對比
                    metric_cn = {
                        "ROC_AUC": "ROC 曲線下面積",
                        "PR_AUC": "精確率-召回率曲線下面積",
                        "F1": "F1 分數",
                        "Brier(ECE proxy)": "Brier 分數 (校準誤差代理)"
                    }
                    
                    for metric in ["ROC_AUC","PR_AUC","F1","Brier(ECE proxy)"]:
                        sub = res_df[["tag",metric]]
                        
                        # 生成雙語版本
                        fig = plt.figure(figsize=(9,5))
                        bars = plt.bar(sub["tag"], sub[metric], alpha=0.8, color=['skyblue', 'lightcoral', 'lightgreen'])
                        
                        for bar in bars:
                            height = bar.get_height()
                            plt.text(bar.get_x() + bar.get_width()/2., height,
                                    f'{height:.3f}',
                                    ha='center', va='bottom', fontsize=10)
                        
                        plt.ylabel(f"{metric_cn.get(metric, metric)}", fontsize=12)
                        plt.title(f"證據 (4): 模型泛化能力 — {metric_cn.get(metric, metric)}\nEvidence (4): {metric}", 
                                 fontsize=13, pad=15)
                        plt.xticks(rotation=15, ha="right", fontsize=10)
                        plt.grid(axis='y', alpha=0.3, linestyle='--')
                        plt.tight_layout()
                        plt.savefig(Path(out_dir, f"e4_{metric}.png"), dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        print(f"已儲存: e4_{metric}.png")
                        
                        # 生成純中文版本
                        fig_cn = plt.figure(figsize=(9,5))
                        bars = plt.bar(sub["tag"], sub[metric], alpha=0.8, color=['skyblue', 'lightcoral', 'lightgreen'])
                        
                        for bar in bars:
                            height = bar.get_height()
                            plt.text(bar.get_x() + bar.get_width()/2., height,
                                    f'{height:.3f}',
                                    ha='center', va='bottom', fontsize=10)
                        
                        plt.ylabel(f"{metric_cn.get(metric, metric)}", fontsize=12)
                        plt.title(f"證據 (4): 模型泛化能力 — {metric_cn.get(metric, metric)}", fontsize=13, pad=15)
                        plt.xticks(rotation=15, ha="right", fontsize=10)
                        plt.grid(axis='y', alpha=0.3, linestyle='--')
                        plt.tight_layout()
                        plt.savefig(Path(out_dir, f"e4_{metric}_cn.png"), dpi=150, bbox_inches='tight')
                        plt.close(fig_cn)
                        print(f"已儲存: e4_{metric}_cn.png")
                        
                        # 生成純英文版本
                        fig_en = plt.figure(figsize=(9,5))
                        bars = plt.bar(sub["tag"], sub[metric], alpha=0.8, color=['skyblue', 'lightcoral', 'lightgreen'])
                        
                        for bar in bars:
                            height = bar.get_height()
                            plt.text(bar.get_x() + bar.get_width()/2., height,
                                    f'{height:.3f}',
                                    ha='center', va='bottom', fontsize=10)
                        
                        plt.ylabel(f"{metric}", fontsize=12)
                        plt.title(f"Evidence (4): {metric}", fontsize=13, pad=15)
                        plt.xticks(rotation=15, ha="right", fontsize=10)
                        plt.grid(axis='y', alpha=0.3, linestyle='--')
                        plt.tight_layout()
                        plt.savefig(Path(out_dir, f"e4_{metric}_en.png"), dpi=150, bbox_inches='tight')
                        plt.close(fig_en)
                        print(f"已儲存: e4_{metric}_en.png")
    else:
        print("警告: 沒有 disposition 欄位,跳過模型驗證")

    # # ========== (5) 工程可行性(可用率 proxy) ==========
    # print("\n" + "="*60)
    # print("證據 (5): 工程可行性與失敗率")
    # print("="*60)
    # 
    # ops = []
    # for ds in ds_vals:
    #     g = df[df["__dataset__"]==ds]
    #     if g.empty: continue
    #     id_cols = [c for c in ["kepid","tid","epic"] if c in g.columns]
    #     id_cov = g[id_cols].notna().any(axis=1).mean() if id_cols else 0.0
    #     core_cols = [c for c in ["planet_period","planet_transit_epoch"] if c in g.columns]
    #     core_cov = g[core_cols].notna().mean(axis=1).mean() if core_cols else 0.0
    #     ops.append({"dataset": ds, "Identifier coverage": id_cov, "Core fields coverage": core_cov})
    # ops_df = pd.DataFrame(ops)
    # ops_df.to_csv(Path(out_dir,"e5_ops.csv"), index=False)
    # print(f"已儲存: e5_ops.csv")
    # 
    # if not ops_df.empty:
    #     x = np.arange(len(ops_df))
    #     
    #     # 生成雙語版本
    #     fig = plt.figure(figsize=(10,6))
    #     bars1 = plt.bar(x-0.2, ops_df["Identifier coverage"], width=0.4, 
    #                    label="識別子覆蓋率 (Identifier)", alpha=0.8, color='steelblue')
    #     bars2 = plt.bar(x+0.2, ops_df["Core fields coverage"], width=0.4, 
    #                    label="核心欄位覆蓋率 (Period + Epoch)", alpha=0.8, color='coral')
    #     
    #     for bars in [bars1, bars2]:
    #         for bar in bars:
    #             height = bar.get_height()
    #             plt.text(bar.get_x() + bar.get_width()/2., height,
    #                     f'{height:.1%}',
    #                     ha='center', va='bottom', fontsize=10)
    #     
    #     plt.xticks(x, ops_df["dataset"], fontsize=12)
    #     plt.ylim(0, 1.15)
    #     plt.ylabel("可用率 (0-1)", fontsize=12)
    #     plt.xlabel("資料集", fontsize=12)
    #     plt.title("證據 (5): 工程可行性指標\nEvidence (5): Ops feasibility proxies", 
    #              fontsize=14, pad=15)
    #     plt.legend(fontsize=11, loc='upper right')
    #     plt.grid(axis='y', alpha=0.3, linestyle='--')
    #     plt.tight_layout()
    #     plt.savefig(Path(out_dir,"e5_ops.png"), dpi=150, bbox_inches='tight')
    #     plt.close(fig)
    #     print(f"已儲存: e5_ops.png")
    #     
    #     # 生成純中文版本
    #     fig_cn = plt.figure(figsize=(10,6))
    #     bars1 = plt.bar(x-0.2, ops_df["Identifier coverage"], width=0.4, 
    #                    label="識別子覆蓋率", alpha=0.8, color='steelblue')
    #     bars2 = plt.bar(x+0.2, ops_df["Core fields coverage"], width=0.4, 
    #                    label="核心欄位覆蓋率 (週期 + 時刻)", alpha=0.8, color='coral')
    #     
    #     for bars in [bars1, bars2]:
    #         for bar in bars:
    #             height = bar.get_height()
    #             plt.text(bar.get_x() + bar.get_width()/2., height,
    #                     f'{height:.1%}',
    #                     ha='center', va='bottom', fontsize=10)
    #     
    #     plt.xticks(x, ops_df["dataset"], fontsize=12)
    #     plt.ylim(0, 1.15)
    #     plt.ylabel("可用率 (0-1)", fontsize=12)
    #     plt.xlabel("資料集", fontsize=12)
    #     plt.title("證據 (5): 工程可行性指標", fontsize=14, pad=15)
    #     plt.legend(fontsize=11, loc='upper right')
    #     plt.grid(axis='y', alpha=0.3, linestyle='--')
    #     plt.tight_layout()
    #     plt.savefig(Path(out_dir,"e5_ops_cn.png"), dpi=150, bbox_inches='tight')
    #     plt.close(fig_cn)
    #     print(f"已儲存: e5_ops_cn.png")
    #     
    #     # 生成純英文版本
    #     fig_en = plt.figure(figsize=(10,6))
    #     bars1 = plt.bar(x-0.2, ops_df["Identifier coverage"], width=0.4, 
    #                    label="Identifier coverage", alpha=0.8, color='steelblue')
    #     bars2 = plt.bar(x+0.2, ops_df["Core fields coverage"], width=0.4, 
    #                    label="Core fields (Period + Epoch)", alpha=0.8, color='coral')
    #     
    #     for bars in [bars1, bars2]:
    #         for bar in bars:
    #             height = bar.get_height()
    #             plt.text(bar.get_x() + bar.get_width()/2., height,
    #                     f'{height:.1%}',
    #                     ha='center', va='bottom', fontsize=10)
    #     
    #     plt.xticks(x, ops_df["dataset"], fontsize=12)
    #     plt.ylim(0, 1.15)
    #     plt.ylabel("Coverage (0-1)", fontsize=12)
    #     plt.xlabel("Dataset", fontsize=12)
    #     plt.title("Evidence (5): Ops feasibility proxies", fontsize=14, pad=15)
    #     plt.legend(fontsize=11, loc='upper right')
    #     plt.grid(axis='y', alpha=0.3, linestyle='--')
    #     plt.tight_layout()
    #     plt.savefig(Path(out_dir,"e5_ops_en.png"), dpi=150, bbox_inches='tight')
    #     plt.close(fig_en)
    #     print(f"已儲存: e5_ops_en.png")

    # # ===== 總結(門檻判讀) =====
    # print("\n" + "="*60)
    # print("判讀結果摘要")
    # print("="*60)
    # 
    # verdict_lines = []
    # # 缺失率門檻
    # core_miss = miss_df[miss_df["feature"].isin(CORE)].pivot(index="feature", columns="dataset", values="missing_rate")
    # ok_koi = (core_miss["KOI"]<0.15).mean() if "KOI" in core_miss.columns else 0
    # ok_toi = (core_miss["TESS"]<0.15).mean() if "TESS" in core_miss.columns else 0
    # bad_k2 = (core_miss.get("K2", pd.Series(1, index=core_miss.index))>0.30).mean()
    # verdict_lines.append(f"*Data Availability* KOI<15% ok={ok_koi:.2f}, TOI<15% ok={ok_toi:.2f}, K2>30% bad={bad_k2:.2f}")
    # 
    # # Covariate shift門檻:W1 > 0.2 視為偏移大(簡化)
    # big_shift = shift_df[shift_df["pair"]=="K2_vs_KOI_TOI"]["wasserstein"].gt(0.2).mean()
    # small_shift = shift_df[shift_df["pair"]=="KOI_vs_TOI"]["wasserstein"].lt(0.1).mean()
    # verdict_lines.append(f"*Covariate Shift* KOI↔TOI small={small_shift:.2f}, K2 vs KOI∪TOI big={big_shift:.2f}")
    # 
    # # 相對誤差:若 K2 中位數比 KOI/TOI 高 50%(簡化用均值比較)
    # rel_flags = []
    # for val,ep,en,_ in RELERR_SPECS:
    #     if val not in df.columns:
    #         continue
    #     ds2arr = {}
    #     for ds in ds_vals:
    #         arr = rel_err(df[df["__dataset__"]==ds], val, ep, en)
    #         if len(arr)>=30: ds2arr[ds] = np.median(arr)
    #     if {"KOI","TESS","K2"} <= set(ds2arr):
    #         k2_bad = ds2arr["K2"] > 1.5*min(ds2arr["KOI"], ds2arr["TESS"])
    #         rel_flags.append((val, k2_bad))
    # rel_bad_rate = np.mean([b for _,b in rel_flags]) if rel_flags else np.nan
    # verdict_lines.append(f"*Uncertainty* K2 median rel.err > 1.5× KOI/TOI in {rel_bad_rate:.2f} of tracked vars")
    # 
    # # Ops:K2 低於 0.85 視為高風險(可自行調)
    # if not ops_df.empty:
    #     risk_id = (ops_df.set_index("dataset").loc["K2","Identifier coverage"] if "K2" in ops_df["dataset"].values else 0.0) < 0.85
    #     risk_core = (ops_df.set_index("dataset").loc["K2","Core fields coverage"] if "K2" in ops_df["dataset"].values else 0.0) < 0.85
    #     verdict_lines.append(f"*Ops* K2 identifier/core-coverage risk = {risk_id or risk_core}")
    # 
    # verdict_text = "\n".join(verdict_lines)
    # Path(out_dir,"verdict.md").write_text(verdict_text, encoding="utf-8")
    # print(f"\n已儲存: verdict.md")
    # 
    # print("\n" + verdict_text)
    
    print("\n" + "="*60)
    print("完成! 所有輸出檔案:")
    print("="*60)
    for f in sorted(os.listdir(out_dir)):
        print(f"  - {f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--std_csv", required=True, help="整合後的標準化 CSV(含 __dataset__)")
    ap.add_argument("--out_dir", default="./evidence_out")
    args = ap.parse_args()
    main(args.std_csv, args.out_dir)
