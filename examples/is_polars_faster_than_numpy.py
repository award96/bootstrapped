if __name__ == "__main__":
    import polars as pl
    import numpy as np
    import time

    def create_index_matrix(
        n_rows: int,
        n_bootstraps: int,
        fraction: float=1.0,
        seed: int=10,
        with_replacement: bool=True
    ) -> pl.LazyFrame:
        pl.set_random_seed(seed)
        integer_range = pl.int_range(0, n_rows)
        return (
            pl.LazyFrame(
                pl.Series(pl.select(integer_range)).alias("index")
            )
            .with_columns(
                (
                    integer_range.sample(fraction=fraction, with_replacement=with_replacement)
                    .alias(str(i))
                    for i in range(n_bootstraps)
                )
            )
        )

    # Parameters
    n = 1_000_000
    n_bootstraps = 1_000
    confidence_level = 0.95
    alpha = (1 - confidence_level) / 2

    # Generate data
    rng = np.random.default_rng(42)
    data = rng.standard_normal(n)

    # —— NumPy Bootstrap —— 
    start_np = time.perf_counter()
    rep_means_np = np.empty(n_bootstraps)
    rep_meds_np = np.empty(n_bootstraps)
    for i in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        sample = data[idx]
        rep_means_np[i] = sample.mean()
        rep_meds_np[i] = np.median(sample)
    obs_mean = data.mean()
    obs_median = np.median(data)
    lower_mean = np.percentile(rep_means_np, 100 * alpha)
    upper_mean = np.percentile(rep_means_np, 100 * (1 - alpha))
    lower_med = np.percentile(rep_meds_np, 100 * alpha)
    upper_med = np.percentile(rep_meds_np, 100 * (1 - alpha))
    duration_np = time.perf_counter() - start_np

    # —— Polars Bootstrap —— 
    s = pl.LazyFrame({"data": data})
    idx_df: pl.DataFrame = create_index_matrix(n_rows=n, n_bootstraps=n_bootstraps, seed=42).collect()

    start_pl = time.perf_counter()
    rep_means_pl = pl.Series([None]*n_bootstraps, dtype=pl.Float64)
    rep_meds_pl = pl.Series([None]*n_bootstraps, dtype=pl.Float64)
    for i in range(n_bootstraps):
        sample = s.select(
            pl.all()
            .gather(idx_df[str(i)])
        )
        rep_means_pl[i] = sample.select(pl.col("data").mean()).collect().item()
        rep_meds_pl[i] = sample.select(pl.col("data").median()).collect().item()
    
    obs_mean_pl = s.select(pl.col("data").mean()).collect().item()
    obs_median_pl = s.select(pl.col("data").median()).collect().item()
    lower_mean_pl = rep_means_pl.quantile(alpha)
    upper_mean_pl = rep_means_pl.quantile(1 - alpha)
    lower_med_pl = rep_meds_pl.quantile(alpha)
    upper_med_pl = rep_meds_pl.quantile(1 - alpha)
    duration_pl = time.perf_counter() - start_pl

    # —— Display Results —— 
    print("NumPy Bootstrap Results:")
    print(f" Mean = {obs_mean:.6f}, CI = [{lower_mean:.6f}, {upper_mean:.6f}]")
    print(f" Median = {obs_median:.6f}, CI = [{lower_med:.6f}, {upper_med:.6f}]")
    print(f" Time = {duration_np:.4f} sec\n")

    print("Polars Bootstrap Results:")
    print(f" Mean = {obs_mean_pl:.6f}, CI = [{lower_mean_pl:.6f}, {upper_mean_pl:.6f}]")
    print(f" Median = {obs_median_pl:.6f}, CI = [{lower_med_pl:.6f}, {upper_med_pl:.6f}]")
    print(f" Time = {duration_pl:.4f} sec\n") # 10.570825458999025

    print(f"Numpy takes {duration_np/duration_pl:.2f}x longer than Polars")


    

