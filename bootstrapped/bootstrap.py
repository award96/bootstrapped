from collections.abc import Iterable as IterableABC
from typing import (
    Callable,
    Iterable,
    Optional,
    Union,
    Dict
)
import polars as pl
from . import generate
from . import bootstrap_result

def bootstrap_simulation(
    observed: Iterable[Union[Iterable[Union[int, float]], int, float]],
    statistics: Iterable[Callable],
    confidence_level: float=0.95,
    n_bootstraps: int=100,
    seed: int=10
) -> Dict[str, bootstrap_result.BootstrapResults]:
    """Estimate the mean, upper, and lower bounds of an array of arrays
    for a list of statistics. The result will be an estimate for each
    array of observed samples, for each statistic.

    Args:
        observed: An array of arrays of observed samples, or a singular array.
        statistics: A list of statistics to estimate.
        confidence_level: The confidence level to use for the bootstrap.
        n_bootstraps: The number of bootstraps to use.
        seed: The seed to use for the bootstrap.
    Returns:
        A dictionary of bootstrap results, with the statistic name as the key.
    """
    observed: pl.LazyFrame = arrays_to_pl_lazyframe(observed)
    observed_schema = observed.collect_schema()
    bootstrap_sample_indices: pl.DataFrame = generate.create_index_matrix(
        n_rows=len(observed),
        n_bootstraps=n_bootstraps,
        seed=seed
    )
    statistic_results: Dict[str, Dict[str, pl.Series]] = {
        colname: {
            stat.__name__: pl.Series([None]*n_bootstraps).alias(stat.__name__)
                for stat in statistics
        } for colname in observed_schema.keys()
    }
    for i in range(n_bootstraps):
        bs_sample = observed.select(pl.all().gather(
            bootstrap_sample_indices[str(i)]
        ))
        for colidx, colname in enumerate(observed_schema.keys()):
            for j, stat in enumerate(statistics):
                # TODO
                result = calculate_statistic_on_lazy_series(
                    stat,
                    bs_sample.select(colname)
                )
                statistic_results[colname][stat.__name__][i] = result
    # TODO this is incorrect
    bootstrap_results = {
        colname: {
            stat.__name__: bootstrap_result.BootstrapResults(
                lower_bound=statistic_results[colname][stat.__name__].quantile(alpha),
                value=statistic_results[colname][stat.__name__].quantile(0.5),
                upper_bound=statistic_results[colname][stat.__name__].quantile(1 - alpha)
            )
            for stat in statistics
        } for colname in observed_schema.keys()
    }
    return bootstrap_results
        

def arrays_to_pl_lazyframe(
        observed: Iterable[Union[Iterable[Union[int, float]], int, float]]
    ) -> pl.LazyFrame:
    """For each array in the observed array, create a column in the output lazyframe.
    
    Args:
        observed: An array of arrays of observed samples, or a singular array.
    Returns:
        A lazyframe with the observed samples as columns.
    """
    if isinstance(observed, pl.DataFrame):
        out = observed.lazy()
    elif isinstance(observed, pl.LazyFrame):
        out = observed
    else:
        out = pl.LazyFrame(observed)
    return out
