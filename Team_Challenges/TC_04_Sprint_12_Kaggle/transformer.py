import numpy as np
import pandas as pd

def transformer_screenresolution(df: pd.DataFrame, col: str = "ScreenResolution"):
    """
    Extrae:
      - Res_X: resolución horizontal (int)
      - Res_Y: resolución vertical (int)
      - Touchscreen: 1 si contiene 'Touchscreen' (case-insensitive), sino 0
    """
    s = df[col].fillna("").astype(str).str.strip()
    s_low = s.str.lower()

    # Touchscreen 
    Touchscreen = s_low.str.contains("touchscreen", regex=False).astype(int)

    # Extraer resolución XxY 
    extracted = s_low.str.extract(r"(\d{3,4})x(\d{3,4})")
    Res_X = pd.to_numeric(extracted[0], errors="coerce").fillna(0).astype(int)
    Res_Y = pd.to_numeric(extracted[1], errors="coerce").fillna(0).astype(int)

    out = pd.DataFrame({
        "Res_X": Res_X,
        "Res_Y": Res_Y,
        "Touchscreen": Touchscreen
    }, index=df.index)

    return out


def transformer_screenresolution_hd(df: pd.DataFrame, col: str = "ScreenResolution"):
    """
    Devuelve:
      - Res_X, Res_Y: resolución en píxeles (int, 0 si no encuentra patrón ###x###)
      - Touchscreen: 0/1 si aparece la palabra Touchscreen
      - Screen_Qual_Ord:
            0 = sin etiqueta (no aparece HD/FHD/UHD/4K)
            1 = HD (incluye HD y HD+)
            2 = Full HD (Full HD o FHD)
            3 = Ultra HD/4K/UHD (y también Quad HD+ si aparece)
    """
    s = df[col].fillna("").astype(str).str.strip()
    s_low = s.str.lower()

    # Touchscreen 
    Touchscreen = s_low.str.contains("touchscreen", regex=False).astype(int)

    # Resolución XxY
    res = s_low.str.extract(r"(\d{3,4})x(\d{3,4})", expand=True)
    Res_X = pd.to_numeric(res[0], errors="coerce").fillna(0).astype(int)
    Res_Y = pd.to_numeric(res[1], errors="coerce").fillna(0).astype(int)

    # Ordinal por etiqueta (orden importa: primero lo más alto)
    Screen_Qual_Ord = np.select(
        [
            s_low.str.contains(r"4k|ultra\s*hd|uhd", regex=True),
            s_low.str.contains(r"quad\s*hd", regex=True),          # en tu dataset aparece "Quad HD+"
            s_low.str.contains(r"full\s*hd|\bfhd\b", regex=True),
            s_low.str.contains(r"\bhd\b|hd\+", regex=True),        # HD o HD+
        ],
        [3, 3, 2, 1],
        default=0
    ).astype(int)

    out = pd.DataFrame(
        {
            "Res_X": Res_X,
            "Res_Y": Res_Y,
            "Touchscreen": Touchscreen,
            "Screen_Qual_Ord": Screen_Qual_Ord
        },
        index=df.index
    )

    return out


def transformer_cpu(df: pd.DataFrame, col: str = "Cpu"):
    """
    Toma el texto de la CPU y lo convierte en variables numéricas/binarias para que el modelo entienda 
    “marca”, “familia/gama” y “velocidad”
    
    """
    s = df[col].fillna("").astype(str).str.strip()
    s_low = s.str.lower()

    # ---- marca ----
    CPU_Intel = s_low.str.contains("intel", regex=False).astype(int)
    CPU_AMD   = s_low.str.contains("amd", regex=False).astype(int)
    CPU_Other = ((CPU_Intel == 0) & (CPU_AMD == 0)).astype(int)

    # ---- ordinal por familia ----
    # 1) Intel Core i3/i5/i7/i9 -> guardar 3/5/7/9
    core = pd.to_numeric(
        s_low.str.extract(r"core\s+i\s*([3579])", expand=False),
        errors="coerce"
    )

    # 2) Intel Core M (M3/M5/M7) -> guardar 3/5/7
    corem = pd.to_numeric(
        s_low.str.extract(r"core\s+m\s*(?:m)?\s*([357])\b", expand=False),
        errors="coerce"
    )

    # 3) Ryzen 3/5/7/9
    ryzen = pd.to_numeric(
        s_low.str.extract(r"ryzen\s*([3579])", expand=False),
        errors="coerce"
    )

    # 4) AMD A-Series A6/A8/A9/A10/A12 -> guardar 6/8/9/10/12
    aseries = pd.to_numeric(
        s_low.str.extract(r"\ba\s*(6|8|9|10|12)\b", expand=False),
        errors="coerce"
    )

    # Convertimos NaN -> 0 (como en tu lógica inicial)
    CPU_Core_Ord    = core.fillna(0).astype(int)
    CPU_CoreM_Ord   = corem.fillna(0).astype(int)
    CPU_Ryzen_Ord   = ryzen.fillna(0).astype(int)
    CPU_ASeries_Ord = aseries.fillna(0).astype(int)

    # ---- flags de otras familias (solo cuando NO hay Core/CoreM/Ryzen/A-Series) ----
    has_main_family = (
        (CPU_Core_Ord > 0) |
        (CPU_CoreM_Ord > 0) |
        (CPU_Ryzen_Ord > 0) |
        (CPU_ASeries_Ord > 0)
    )

    # flags directos
    CPU_ESeries = (
        s_low.str.contains("e-series", regex=False) |
        s_low.str.contains(r"\be\s*-\s*series\b", regex=True)
    ).astype(int)

    CPU_Celeron = s_low.str.contains("celeron", regex=False).astype(int)
    CPU_Pentium = s_low.str.contains("pentium", regex=False).astype(int)
    CPU_Atom    = s_low.str.contains("atom", regex=False).astype(int)

    # En tu loop: si encontraba alguna familia principal, dejaba estos flags en 0.
    # Entonces los anulamos donde haya familia principal:
    CPU_ESeries = np.where(has_main_family.values, 0, CPU_ESeries.values)
    CPU_Celeron = np.where(has_main_family.values, 0, CPU_Celeron.values)
    CPU_Pentium = np.where(has_main_family.values, 0, CPU_Pentium.values)
    CPU_Atom    = np.where(has_main_family.values, 0, CPU_Atom.values)

    # OtherFamily = 1 si NO hay familia principal y tampoco cae en E-Series/Celeron/Pentium/Atom
    any_minor = (CPU_ESeries + CPU_Celeron + CPU_Pentium + CPU_Atom) > 0
    CPU_OtherFamily = np.where(has_main_family.values, 0, np.where(any_minor, 0, 1)).astype(int)


    # ---- GHz ----
    CPU_GHz = pd.to_numeric(
        s_low.str.extract(r"(\d+(?:\.\d+)?)\s*ghz", expand=False),
        errors="coerce"
    )  # deja NaN si no encuentra, como tu código

    out = pd.DataFrame({
        "CPU_Intel": CPU_Intel,
        "CPU_AMD": CPU_AMD,
        "CPU_Other": CPU_Other,

        "CPU_Core_Ord": CPU_Core_Ord,
        "CPU_CoreM_Ord": CPU_CoreM_Ord,
        "CPU_Ryzen_Ord": CPU_Ryzen_Ord,
        "CPU_ASeries_Ord": CPU_ASeries_Ord,

        "CPU_Celeron": CPU_Celeron,
        "CPU_Pentium": CPU_Pentium,
        "CPU_Atom": CPU_Atom,
        "CPU_ESeries": CPU_ESeries,
        "CPU_OtherFamily": CPU_OtherFamily,

        "CPU_GHz": CPU_GHz
    }, index=df.index)

    return out


def transformer_memory(df: pd.DataFrame, col: str = "Memory"):
    """
    Recibe un DataFrame con columna 'Memory' (texto) y devuelve un DataFrame
    con SSD_GB, HDD_GB, Flash_Storage_GB, Hybrid_GB.
    
    Regla: TB -> GB multiplicando por 1000 (igual que tu código).
    Soporta strings con " + " (1 o más partes).
    """
    s = df[col].fillna("").astype(str).str.strip()
    s_low = s.str.lower()

    # Separar en partes (lista por fila)
    parts = s_low.str.split(r"\+", regex=True)

    def parse_row(parts_list):
        ssd = 0.0
        hdd = 0.0
        flash = 0.0
        hybrid = 0.0

        for memoria in parts_list:
            memoria = memoria.strip()
            if not memoria:
                continue

            # extraer número (primer número que aparezca)
            m = pd.Series([memoria]).str.extract(r"(\d+(?:\.\d+)?)", expand=False).iloc[0]
            if pd.isna(m):
                continue

            num = float(m)

            # unidad -> GB (tu regla TB * 1000)
            if "tb" in memoria:
                num *= 1000

            # detectar tipo y sumar
            if "ssd" in memoria:
                ssd += num
            elif "hdd" in memoria:
                hdd += num
            elif "flash storage" in memoria:
                flash += num
            elif "hybrid" in memoria:
                hybrid += num

        return pd.Series([ssd, hdd, flash, hybrid])

    # Aplicar por fila
    parsed = parts.apply(parse_row)
    parsed.columns = ["SSD_GB", "HDD_GB", "Flash_Storage_GB", "Hybrid_GB"]
    parsed.index = df.index

    return parsed


def transformer_gpu(df: pd.DataFrame, col: str = "Gpu"):
    '''
    Toma el texto de la GPU y genera indicadores de marca/tipo/segmento + números de modelo cuando existen.
    '''
    
    s = df[col].fillna("").astype(str).str.strip()
    s_low = s.str.lower()

    # Marca
    gpu_intel  = s_low.str.contains("intel", regex=False).astype(int)
    gpu_nvidia = s_low.str.contains("nvidia", regex=False).astype(int)
    gpu_amd    = s_low.str.contains("amd", regex=False).astype(int)
    gpu_other  = ((gpu_intel == 0) & (gpu_nvidia == 0) & (gpu_amd == 0)).astype(int)

    # Familias / segmentos
    gpu_quadro  = s_low.str.contains("quadro", regex=False).astype(int)
    gpu_gtx     = s_low.str.contains("gtx", regex=False).astype(int)
    gpu_rtx     = s_low.str.contains("rtx", regex=False).astype(int)
    gpu_mx      = s_low.str.contains(r"\bmx\s*\d", regex=True).astype(int)
    gpu_rx      = (s_low.str.contains("radeon rx", regex=False) | s_low.str.contains(r"\brx\s*\d", regex=True)).astype(int)
    gpu_firepro = s_low.str.contains("firepro", regex=False).astype(int)

    # Intel integradas
    gpu_iris = s_low.str.contains("iris", regex=False).astype(int)
    gpu_hd   = s_low.str.contains("hd graphics", regex=False).astype(int)
    gpu_uhd  = s_low.str.contains("uhd graphics", regex=False).astype(int)

    # Dedicada vs integrada (tu regla práctica)
    gpu_dedicada = ((gpu_nvidia == 1) | (gpu_quadro == 1) | (gpu_rx == 1) | (gpu_firepro == 1) | ((gpu_amd == 1) & (s_low.str.contains("radeon", regex=False)))).astype(int)

    # Números por familia (0 si no aplica; 1 si aplica pero no pudo extraer)
    gtx_num_raw = pd.to_numeric(
        s_low.str.extract(r"gtx\s*([0-9]{3,4})", expand=False),
        errors="coerce"
    ).fillna(0).astype(int)
    gtx_num = np.where(gpu_gtx.values == 1, np.where(gtx_num_raw.values > 0, gtx_num_raw.values, 1), 0)

    mx_num_raw = pd.to_numeric(
        s_low.str.extract(r"mx\s*([0-9]{2,4})", expand=False),
        errors="coerce"
    ).fillna(0).astype(int)
    mx_num = np.where(gpu_mx.values == 1, np.where(mx_num_raw.values > 0, mx_num_raw.values, 1), 0)

    quadro_num_raw = pd.to_numeric(
        s_low.str.extract(r"quadro\s*[a-z]*\s*([0-9]{3,4})", expand=False),
        errors="coerce"
    ).fillna(0).astype(int)
    quadro_num = np.where(gpu_quadro.values == 1, np.where(quadro_num_raw.values > 0, quadro_num_raw.values, 1), 0)

    rx_num_raw = pd.to_numeric(
        s_low.str.extract(r"rx\s*([0-9]{3,4})", expand=False),
        errors="coerce"
    ).fillna(0).astype(int)
    rx_num = np.where(gpu_rx.values == 1, np.where(rx_num_raw.values > 0, rx_num_raw.values, 1), 0)

    # Intel num (HD/UHD/Iris + número o fallback 3-4 dígitos)
    intel_num_main = pd.to_numeric(
        s_low.str.extract(r"(?:hd graphics|uhd graphics|iris(?: plus)?)\s*([0-9]{3,4})", expand=False),
        errors="coerce"
    )
    intel_num_fallback = pd.to_numeric(
        s_low.str.extract(r"(\d{3,4})", expand=False),
        errors="coerce"
    )
    intel_num = intel_num_main.fillna(intel_num_fallback).fillna(0).astype(int)
    intel_num = np.where(gpu_intel.values == 1, intel_num.values, 0)

    # AMD Radeon sin RX + número
    radeon_num_raw = pd.to_numeric(
        s_low.str.extract(r"radeon\s*([0-9]{3,4})", expand=False),
        errors="coerce"
    ).fillna(0).astype(int)

    radeon_num = np.where(
        (gpu_amd.values == 1) & (s_low.str.contains("radeon", regex=False).values) & (gpu_rx.values == 0),
        radeon_num_raw.values,
        0
    )

    out = pd.DataFrame({
        "GPU_Intel": gpu_intel,
        "GPU_Nvidia": gpu_nvidia,
        "GPU_AMD": gpu_amd,
        "GPU_Other": gpu_other,

        "GPU_Dedicada": gpu_dedicada,

        "GPU_GTX": gpu_gtx,
        "GPU_RTX": gpu_rtx,
        "GPU_MX": gpu_mx,
        "GPU_Quadro": gpu_quadro,
        "GPU_RX": gpu_rx,
        "GPU_FirePro": gpu_firepro,

        "GPU_GTX_Num": gtx_num,
        "GPU_MX_Num": mx_num,
        "GPU_Quadro_Num": quadro_num,
        "GPU_RX_Num": rx_num,

        "GPU_Iris": gpu_iris,
        "GPU_HD": gpu_hd,
        "GPU_UHD": gpu_uhd,

        "GPU_Intel_Num": intel_num,
        "GPU_Radeon_Num": radeon_num,
    }, index=df.index)

    return out


def transformer_unit(df: pd.DataFrame, col: str, unit: str):
    """
    Convierte una columna string con unidad a numérico.
    Soporta por ahora:
      - unit="GB"  -> devuelve columna: f"{col}_GB" (int)
      - unit="kg"  -> devuelve columna: f"{col}_kg" (float)
    """
    unit_norm = unit.strip().lower()

    s = df[col].fillna("").astype(str).str.strip()

    # normalización general
    s = s.str.replace(",", ".", regex=False)          # por si "1,37kg"
    s_low = s.str.lower()

    if unit_norm == "gb":
        # ejemplo: "8GB" / "16 gb" -> 8 / 16
        num = (s_low
               .str.replace("gb", "", regex=False)
               .str.strip())
        num = pd.to_numeric(num, errors="coerce").fillna(0).astype(int)
        out_col = f"{col}_GB"

    elif unit_norm == "kg":
        # ejemplo: "1.37kg" / "2 kg" -> 1.37 / 2.0
        num = (s_low
               .str.replace("kg", "", regex=False)
               .str.strip())
        num = pd.to_numeric(num, errors="coerce")  # deja NaN si no puede
        out_col = f"{col}_kg"

    else:
        raise ValueError("Unidad no soportada. Usá 'GB' o 'kg'.")

    return pd.DataFrame({out_col: num}, index=df.index)


def transformer_weight(df: pd.DataFrame, col: str = "Weight") -> pd.DataFrame:
    s = df[col].fillna("").astype(str).str.lower().str.strip()
    # extrae float: "1.37kg" / "2.1 kg" (si hay coma, la convertimos)
    s = s.str.replace(",", ".", regex=False)
    w = pd.to_numeric(s.str.extract(r"(\d+(?:\.\d+)?)", expand=False), errors="coerce")
    out = pd.DataFrame({"Weight_kg": w}, index=df.index)  # float con NaN si no hay
    return out


def transformer_ram(df: pd.DataFrame, col: str = "Ram") -> pd.DataFrame:
    s = df[col].fillna("").astype(str).str.lower().str.strip()
    # extrae número, tolera "8gb" o "16 gb"
    ram = pd.to_numeric(s.str.extract(r"(\d+)", expand=False), errors="coerce")
    out = pd.DataFrame({"Ram_GB": ram.fillna(0).astype(int)}, index=df.index)
    return out
