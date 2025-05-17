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
    bootstrap_sample_indices: pl.DataFrame = generate.create_index_matrix(
        n_rows=len(observed),
        n_bootstraps=n_bootstraps,
        seed=seed
    )
    bootstrap_results: Dict[str, bootstrap_result.BootstrapResults] = {
        stat.__name__: [
            pl.Series([None]*n_bootstraps).alias(stat.__name__) for _ in range(observed.shape[0])
        ] for stat in statistics
    }
    for i in range(n_bootstraps):
        for input_idx, obs in enumerate(observed):
            bs_sample = obs.select(pl.all().gather(
                 bootstrap_sample_indices[str(i)]
            ))
            for j, stat in enumerate(statistics):
                result = stat(bs_sample.collect())
                result = bootstrap_result.BootstrapResults(
                    lower_bound=result.lower_bound,
                    value=result.value,
                    upper_bound=result.upper_bound
                )
                bootstrap_results[stat.__name__][input_idx][i] = result
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
        return observed.lazy()
    elif isinstance(observed, pl.LazyFrame):
        return observed
    return pl.LazyFrame(
        sub_array_pl.alias(str(i)) for i, sub_array_pl in enumerate(arrays_to_iter_of_series(observed))
    )

def arrays_to_iter_of_series(
    observed: Iterable[Union[Iterable[Union[int, float]], int, float]]
) -> Iterable[pl.Series]:
    """Convert an array of arrays (or a single array) to an iterator of Polars Series."""
    # Materialize into a list so we can inspect it multiple times
    obs_list = list(observed)

    # 1) Detect a single one-dimensional array of numbers
    if all(not isinstance(el, IterableABC) for el in obs_list):
        # e.g. [1, 2, 3, 4]
        yield pl.Series(obs_list)
        return

    # 2) Otherwise we assume a “list of lists” – enforce consistency
    lengths = [len(el) for el in obs_list]
    if len(set(lengths)) != 1:
        raise ValueError("All sub‐arrays must have the same length for bootstrapping.")

    # 3) Convert each element into a Series
    for sub_array in obs_list:
        if isinstance(sub_array, pl.Series):
            yield sub_array
        else:
            # Polars will coerce numpy arrays, Arrow arrays, pandas Series, Python lists, etc.
            yield pl.Series(sub_array)
