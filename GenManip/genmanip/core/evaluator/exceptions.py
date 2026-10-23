"""Custom exceptions for evaluation infrastructure."""


class EvalError(Exception):
    """Base exception for evaluation errors."""

    pass


class WorkerNotFoundError(EvalError):
    """Raised when a worker is not found in the pool."""

    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        super().__init__(f"Worker {worker_id} not found in pool")


class TaskTimeoutError(EvalError):
    """Raised when an operation times out."""

    def __init__(self, operation: str, timeout: float):
        self.operation = operation
        self.timeout = timeout
        super().__init__(f"{operation} timed out after {timeout}s")


class InvalidConfigError(EvalError):
    """Raised when configuration is invalid."""

    def __init__(self, message: str):
        super().__init__(f"Invalid configuration: {message}")


class InsufficientResourcesError(EvalError):
    """Raised when there are not enough resources to allocate workers."""

    def __init__(self, requested: int, available: int):
        self.requested = requested
        self.available = available
        super().__init__(
            f"Insufficient resources: requested {requested} workers, "
            f"but only {available} slots available"
        )


class JobInProgressError(EvalError):
    """Raised when trying to start a new job while one is already in progress."""

    def __init__(self, benchmark_id: str, run_id: str, active_workers: list[str]):
        self.benchmark_id = benchmark_id
        self.run_id = run_id
        self.active_workers = active_workers
        super().__init__(
            f"Job already in progress: benchmark_id={benchmark_id}, run_id={run_id}, "
            f"active_workers={active_workers}. Use --overwrite to force restart."
        )
