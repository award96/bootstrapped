import polars as pl

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

if __name__ == "__main__":
    random_integers_matrix = create_index_matrix(10, 5).collect()
    print(random_integers_matrix)
