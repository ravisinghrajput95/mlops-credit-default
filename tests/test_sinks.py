"""Monitoring must never be able to take down serving."""

from __future__ import annotations

from credit_default.api.sinks import GCSParquetSink, NullSink, build_sink
from credit_default.config import Settings


def test_default_is_a_null_sink():
    assert isinstance(build_sink(Settings(prediction_sink="none")), NullSink)


def test_unreachable_postgres_degrades_to_null_sink():
    """A dead database must not stop the API from serving predictions."""
    settings = Settings(
        prediction_sink="postgres",
        postgres_dsn="postgresql://nobody@127.0.0.1:1/nothing",
    )
    assert isinstance(build_sink(settings), NullSink)


def test_gcs_sink_without_a_prefix_degrades_to_null_sink():
    assert isinstance(build_sink(Settings(prediction_sink="gcs")), NullSink)


def test_gcs_sink_requires_a_prefix():
    try:
        GCSParquetSink("")
    except ValueError as exc:
        assert "gcs_prediction_prefix" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected a ValueError for an empty prefix")


def test_null_sink_accepts_records_silently():
    NullSink().write([{"anything": 1}])
