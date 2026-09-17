"""Shared board that tools write to and the app reads from.

This is the bridge between the crew's tool calls and the marimo app's map /
stats-table rendering: a model tool that computes a raster or a number calls
`current().add_layer(...)` / `current().add_stat(...)` as a side effect, and
after (or during) a run the app reads `current()` to build the map and table.
No parsing of the crew's final-answer text is needed.

One global board is used — this sandbox runs one crew invocation at a time in
one notebook process. Call reset() before each crew.run().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Legend:
    label: str
    min: float
    max: float
    palette: list[str]
    unit: str = ""


@dataclass
class Layer:
    name: str
    tile_url: str  # XYZ tile template, e.g. from Image.getMapId()['tile_fetcher'].url_format
    attribution: str = "Google Earth Engine"
    legend: Legend | None = None  # lets the map draw a colorbar matching this layer's vis params


@dataclass
class Board:
    layers: list[Layer] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)  # one dict per stats-table row
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)  # named summary tables
    notes: list[str] = field(default_factory=list)  # free-text asides (e.g. tool warnings)

    def add_layer(
        self, name: str, tile_url: str, attribution: str = "Google Earth Engine",
        legend: dict[str, Any] | None = None,
    ) -> None:
        self.layers.append(Layer(
            name=name, tile_url=tile_url, attribution=attribution,
            legend=Legend(**legend) if legend else None,
        ))

    def add_stat(self, **row: Any) -> None:
        self.rows.append(row)

    def add_table(self, name: str, rows: list[dict[str, Any]]) -> None:
        """Publish a named summary table (e.g. a multi-row/multi-column
        breakdown a model wants to show as its own table, distinct from the
        single flat stats table `add_stat` rows build up). The app renders
        one block per table, alongside the stats table."""
        self.tables[name] = rows

    def add_note(self, text: str) -> None:
        self.notes.append(text)


_board = Board()


def current() -> Board:
    return _board


def reset() -> Board:
    global _board
    _board = Board()
    return _board
