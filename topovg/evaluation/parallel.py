"""Bounded multiprocessing; yield samples as their work completes."""
from .worker import evaluate_and_save

def parallel_results(jobs, workers):
    """Return completed jobs promptly; bound queued work to twice the workers."""
    import multiprocessing as mp
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    iterator = iter(jobs)
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'))
    pending = set()
    try:
        for _ in range(workers * 2):
            job = next(iterator, None)
            if job is None:
                break
            pending.add(pool.submit(evaluate_and_save, job))
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
                job = next(iterator, None)
                if job is not None:
                    pending.add(pool.submit(evaluate_and_save, job))
    finally:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
