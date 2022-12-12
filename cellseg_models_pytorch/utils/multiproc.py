from typing import Any, Callable, Generator, List, Union

from pathos.pools import ProcessPool, SerialPool, ThreadPool

__all__ = ["run_pool"]


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


def run_pool(
    func: Callable,
    args: List[Any],
    ret: bool = True,
    pooltype: str = "thread",
    maptype: str = "amap",
) -> Union[List[Any], None]:
    """Run a pathos Thread, Process or Serial pool object.

    NOTE: if `ret` is set to True and `func` callable does not return anything. This
          will return a list of None values.

    Parameters
    ----------
        func : Callable
            The function that will be copied to existing cores and run in parallel.
        args : List[Any]
            A list of arguments for each of the parallelly executed functions.
        ret : bool, default=True
            Flag, whether to return a list of results from the pool object. Will be set
            to False e.g. when saving data to disk in parallel etc.
        pooltype : str, default="thread"
            The pathos pooltype. Allowed: ("process", "thread", "serial")
        maptype : str, default="amap"
            The map type of the pathos Pool object.
            Allowed: ("map", "amap", "imap", "uimap")

    Raises
    ------
        ValueError: if illegal `pooltype` or `maptype` is given.

    Returns
    -------
        Union[List[Any], None]:
            A list of results or None.
    """
    allowed = ("process", "thread", "serial")
    if pooltype not in allowed:
        raise ValueError(f"Illegal `pooltype`. Got {pooltype}. Allowed: {allowed}")

    allowed = ("map", "amap", "imap", "uimap")
    if maptype not in allowed:
        raise ValueError(f"Illegal `maptype`. Got {maptype}. Allowed: {allowed}")

    Pool = None
    if pooltype == "thread":
        Pool = ThreadPool
    elif pooltype == "process":
        Pool = ProcessPool
    else:
        if maptype in ("amap", "uimap"):
            raise ValueError(
                f"`SerialPool` has only `map` & `imap` implemented. Got: {maptype}."
            )
        Pool = SerialPool

    results = [] if ret else None
    if maptype == "map":
        with Pool() as pool:
            it = pool.map(func, args)
            results = iter_pool_generator(it, results)
    elif maptype == "amap":
        with Pool() as pool:
            results = pool.amap(func, args).get()
    elif maptype == "imap":
        with Pool() as pool:
            it = pool.imap(func, args)
            results = iter_pool_generator(it, results)
    elif maptype == "uimap":
        with Pool() as pool:
            it = pool.uimap(func, args)
            results = iter_pool_generator(it, results)

    return results
