import os

_exit_status = 0


def pytest_sessionfinish(session, exitstatus):
    global _exit_status
    _exit_status = exitstatus


def pytest_unconfigure(config):
    """Force-exit after pytest completes to avoid hanging on JAX background threads."""
    os._exit(_exit_status)
