# src/split_time.py
import os
import argparse
import pandas as pd

def main(in_path: str, outdir: str, ts_col: str, p_train: float, p_val: float, min_rows: int):
    assert os.path.exists(in_path), f"No existe {in_path}"
    assert 0 < p_train < 1, "ptrain debe estar en (0,1)"
    assert 0 < p_val < 1,   "pval debe estar en (0,1)"
    assert p_train + p_val < 1, "ptrain + pval debe ser < 1 (deja espacio para test)"

    df = pd.read_parquet(in_path)

    if ts_col not in df.columns:
        raise ValueError(f"Falta la columna de timestamp '{ts_col}'. Cols: {df.columns.tolist()}")

    # Asegurar datetime y remover NaT
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    n_bad = int(df[ts_col].isna().sum())
    if n_bad:
        print(f"[WARN] {n_bad} filas con {ts_col} inválido → descartadas.")
        df = df.dropna(subset=[ts_col]).copy()

    # Orden global por tiempo (si hay station_id también ordenará estable)
    by_cols = [ts_col] if "station_id" not in df.columns else ["station_id", ts_col]
    df = df.sort_values(by_cols)

    # Construir ejes de tiempo únicos y cortes
    ts_sorted = pd.Index(df[ts_col].sort_values().unique())
    n = len(ts_sorted)
    if n < 3:
        raise ValueError("Muy pocos timestamps únicos para dividir en train/val/test.")

    cut1 = ts_sorted[int(n * p_train)]
    cut2 = ts_sorted[int(n * (p_train + p_val))]

    train = df[df[ts_col] <= cut1].copy()
    val   = df[(df[ts_col] > cut1) & (df[ts_col] <= cut2)].copy()
    test  = df[df[ts_col] > cut2].copy()

    for name, part in [("train", train), ("val", val), ("test", test)]:
        if len(part) < min_rows:
            raise ValueError(f"El split '{name}' quedó con muy pocas filas ({len(part)}). "
                             f"Ajustá ptrain/pval o revisá el rango temporal.")

    os.makedirs(outdir, exist_ok=True)
    train.to_parquet(os.path.join(outdir, "train.parquet"), index=False)
    val.to_parquet(  os.path.join(outdir, "val.parquet"),   index=False)
    test.to_parquet( os.path.join(outdir, "test.parquet"),  index=False)

    print(f"[OK] Splits guardados en {outdir}")
    print(f"train: {train[ts_col].min()} → {train[ts_col].max()} | filas={len(train)}")
    print(f"val:   {val[ts_col].min()}   → {val[ts_col].max()}   | filas={len(val)}")
    print(f"test:  {test[ts_col].min()}  → {test[ts_col].max()}  | filas={len(test)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Split temporal Train/Val/Test sin fuga.")
    ap.add_argument("--in",      dest="in_path",  required=True, help="Parquet preparado (ecobici_model_ready.parquet)")
    ap.add_argument("--out",     dest="outdir",   required=True, help="Directorio salida (data/splits)")
    ap.add_argument("--ts",      dest="ts_col",   default="ts_local", help="Columna timestamp (default: ts_local)")
    ap.add_argument("--ptrain",  dest="p_train",  type=float, default=0.70, help="Proporción train (default 0.70)")
    ap.add_argument("--pval",    dest="p_val",    type=float, default=0.15, help="Proporción val (default 0.15)")
    ap.add_argument("--minrows", dest="min_rows", type=int, default=100, help="Mínimo de filas por split (default 100)")
    args = ap.parse_args()
    main(args.in_path, args.outdir, args.ts_col, args.p_train, args.p_val, args.min_rows)