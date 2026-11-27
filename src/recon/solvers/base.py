"""Solver interface. A solver sees only ``instance["input"]`` and returns ranked candidates."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from typing import Protocol

SOLVERS: dict[str, Callable[..., Solver]] = {}


class Solver(Protocol):
    name: str

    def solve(self, inp: dict) -> dict:
        """Return ``{"candidates": [...], "stats": {...}}`` (see recon.bench.metrics)."""
        ...


def register(name: str):
    def deco(factory):
        SOLVERS[name] = factory
        return factory

    return deco


def run(solver: Solver, instances: Iterable[dict]) -> Iterator[dict]:
    for inst in instances:
        t0 = time.perf_counter()
        out = solver.solve(inst["input"])
        out.setdefault("stats", {})["seconds"] = time.perf_counter() - t0
        yield {"id": inst["id"], **out}
