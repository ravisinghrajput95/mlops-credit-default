"""A feature store with point-in-time correctness, built on the dataset's real time axis.

The UCI file looks like a flat table of 23 columns, but eleven of those columns
are not attributes -- they are six monthly observations of three quantities,
flattened sideways. ``PAY_0``/``BILL_AMT1``/``PAY_AMT1`` describe September 2005;
``PAY_6``/``BILL_AMT6``/``PAY_AMT6`` describe April. The time dimension was there
the whole time, wearing a disguise that makes leakage effortless.

* ``events``  -- unfolds the wide frame into an event table, losslessly.
* ``views``   -- feature definitions computed over a window that ends at as-of.
* ``pit``     -- the as-of join, and the one everybody writes instead.
"""
