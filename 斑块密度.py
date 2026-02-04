import os
import rasterio
import numpy as np
import pandas as pd
from scipy.ndimage import label

# ===================== 基础函数 =====================

def read_raster(path):
    with rasterio.open(path) as src:
        data = src.read(1)
        pixel_area = src.res[0] * src.res[1]
    return data, pixel_area


def identify_patches(binary_map):
    structure = np.ones((3, 3))
    labeled, num = label(binary_map, structure=structure)
    return labeled, num


def calc_PD(n_patches, area_km2):
    if area_km2 == 0:
        return np.nan
    return n_patches / area_km2


def calc_FI(n_patches, area_km2):
    if n_patches == 0 or area_km2 == 0:
        return np.nan
    avg_patch_area = area_km2 / n_patches
    PD = n_patches / area_km2
    return PD / avg_patch_area


# ===================== 路径与情景 =====================

lucc_dir = r"D:\SD_PLUS_Model\paper_result\landuse"
hq_dir   = r"D:\SD_PLUS_Model\paper_result\quality_all"

scenarios = [
    "PES_2030", "ESP_2030", "SCS_2030",
    "PES_2040", "ESP_2040", "SCS_2040"
]

# ===================== 类别定义 =====================

lucc_classes = {
    1: "Cropland",
    2: "Forest",
    3: "Grassland",
    4: "Water",
    5: "Construction",
    6: "Unused"
}

results = []

# ===================== 主循环 =====================

for scen in scenarios:

    print(f"Processing {scen} ...")

    lucc, pixel_area = read_raster(os.path.join(lucc_dir, f"{scen}.tif"))
    hq, _ = read_raster(os.path.join(hq_dir, f"{scen}.tif"))

    for code, cls_name in lucc_classes.items():

        # ---------- LUCC PD / FI ----------
        lucc_bin = (lucc == code).astype(np.uint8)
        if np.sum(lucc_bin) == 0:
            continue

        _, n_lucc = identify_patches(lucc_bin)
        area_lucc = np.sum(lucc_bin) * pixel_area / 1e6

        LUCC_PD = calc_PD(n_lucc, area_lucc)
        LUCC_FI = calc_FI(n_lucc, area_lucc)

        # ---------- HQ PD / FI（限定在该 LUCC 类型内） ----------
        hq_masked = np.where(lucc_bin == 1, hq, np.nan)
        hq_bin = ((hq_masked >= 0) & (hq_masked <= 1)).astype(np.uint8)

        if np.sum(hq_bin) > 0:
            _, n_hq = identify_patches(hq_bin)
            area_hq = np.sum(hq_bin) * pixel_area / 1e6
            HQ_PD = calc_PD(n_hq, area_hq)
            HQ_FI = calc_FI(n_hq, area_hq)
        else:
            HQ_PD, HQ_FI = np.nan, np.nan

        results.append({
            "Scenario": scen,
            "Class": cls_name,
            "LUCC_PD": LUCC_PD,
            "LUCC_FI": LUCC_FI,
            "HQ_PD": HQ_PD,
            "HQ_FI": HQ_FI
        })

# ===================== 输出结果 =====================

df = pd.DataFrame(results)

out_csv = r"D:\SD_PLUS_Model\paper_result\LUCC_HQ_PD_FI_merged.csv"
df.to_csv(out_csv, index=False, encoding="utf-8-sig")

print("✅ 已按指定结构输出结果：", out_csv)
