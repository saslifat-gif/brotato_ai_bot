"""Deterministic same-trace controller comparisons."""


def compare_recording(*args, **kwargs):
    from .backtest import compare_recording as implementation

    return implementation(*args, **kwargs)


__all__ = ["compare_recording"]
