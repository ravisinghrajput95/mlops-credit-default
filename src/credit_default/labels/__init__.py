"""The delayed-label pipeline.

In credit risk the label is not merely late, it is late *by definition*: a
"default next month" outcome does not exist until next month has happened, and
it does not reach whoever is measuring until the servicing and collections
process has run. Everything in this package exists to make that delay explicit
rather than to pretend it away.

* ``store``       -- where late-arriving outcomes land (Parquet or Postgres).
* ``arrival``     -- the synthetic arrival-lag model used for the demo.
* ``backfill``    -- the point-in-time-correct join of outcomes to predictions.
* ``performance`` -- retrospective scoring on labels that have actually matured.
"""
