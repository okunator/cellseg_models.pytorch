import warnings
from typing import Any, Callable, Generator, List, Union

from pathos.pools import ProcessPool, SerialPool, ThreadPool

__all__ = ["set_pool", "run_pool"]


def iter_pool_generator(it: Generator, res: List = None) -> Union[List[Any], None]:
    """Iterate over a pool generator object.

    Parameters
    ----------
        it : Generator
            A Generator object containing results from a concurrent run.
        res : List | None
            An empty list, where the results from the generator will be saved.
            If None, no results will be saved.

    Returns
    -------
        Union[List[Any], None]:
            A list of results or None.
    """
    if res is not None:
        for x in it:
            res.append(x)
    else:
        for _ in it:
            pass

    return res


def set_pool(
    pooltype: str, nodes: int = -1
) -> Union[ThreadPool, ProcessPool, SerialPool]:
    """Set up a pool of workers based on the specified pool type and number of nodes.

    Paramaters:
        pooltype (str):
            The type of pool to create. Must be one of "process", "thread", or "serial".
        nodes (int, default=-1):
            The number of nodes to use. If -1, use all available CPUs.

    Returns:
        Union[ThreadPool, ProcessPool, SerialPool]:
            An instance of the specified pool type.

    Raises:
        ValueError:
            If `pooltype` is not one of the allowed values ("process", "thread", "serial").
    """

    if pooltype != "serial" and nodes == 1:
        warnings.warn(
            "Serial pool is being used with one node although another pooltype was specified."
        )

    if nodes == -1:
        nodes = None  # Use all available CPUs
    elif nodes == 1:
        pooltype = "serial"

    allowed = ("process", "thread", "serial")
    if pooltype not in allowed:
        raise ValueError(f"Illegal `pooltype`. Got {pooltype}. Allowed: {allowed}")

    Pool = None
    if pooltype == "thread":
        Pool = ThreadPool
    elif pooltype == "process":
        Pool = ProcessPool
    else:
        Pool = SerialPool

    return Pool(nodes=nodes)


def run_pool(
    p: Union[ThreadPool, ProcessPool, SerialPool],
    func: Callable,
    args: List[Any],
    ret: bool = True,
    maptype: str = "amap",
) -> Union[List[Any], None]:
    """
    Run a function in parallel using a specified pool type.

    Parameters:
        p (Union[ThreadPool, ProcessPool, SerialPool]):
            The pool object to use for parallel execution.
        func (Callable):
            The function to be executed in parallel.
        args (List[Any]):
            A list of arguments for each of the parallelly executed functions.
        ret (bool, default=True):
            Flag, whether to return a list of results from the pool object.
            Will be set to False e.g. when saving data to disk in parallel etc.
        pooltype (str, default="thread"):
            The pathos pooltype. Allowed: ("process", "thread", "serial").
        maptype (str, optional):
            The map type of the pathos Pool object.
            Allowed: ("map", "amap", "imap", "uimap").

    Raises:
        ValueError: If illegal `pooltype` or `maptype` is given.

    Returns:
        Union[List[Any], None]: A list of results or None.

    Example:
        >>> def myfunc(x):
        ...     return x * x
        >>> args = [1, 2, 3]
        >>> pool = set_pool("thread")
        >>> res_list = run_pool(myfunc, args, pooltype="thread")
    """
    allowed = ("map", "amap", "imap", "uimap")
    if maptype not in allowed:
        raise ValueError(f"Illegal `maptype`. Got {maptype}. Allowed: {allowed}")

    if isinstance(p, SerialPool):
        if maptype not in ("imap", "map"):
            raise ValueError(
                f"`SerialPool` has only `map` & `imap` implemented. Got: {maptype}."
            )

    results = [] if ret else None
    if maptype == "map":
        with p as pool:
            it = pool.map(func, args)
            results = iter_pool_generator(it, results)
    elif maptype == "imap":
        with p as pool:
            it = pool.imap(func, args)
            results = iter_pool_generator(it, results)
    elif maptype == "uimap":
        with p as pool:
            it = pool.uimap(func, args)
            results = iter_pool_generator(it, results)
    elif maptype == "amap":
        with p as pool:
            results = pool.amap(func, args).get()

    return results
