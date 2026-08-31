"""Load test for the prediction API.

Answers a question the unit tests cannot: what happens under concurrency? A model
that scores in 5ms single-threaded can still miss its latency target once
requests queue, and batch size matters more than request count because inference
cost scales with rows, not calls.

Run against the local stack:

    make load-test                     # 20 users, 60s
    make load-test USERS=100 TIME=5m   # heavier

The p95 target below is the same 1s the Grafana alert rule uses, so a failing
load test and a firing alert mean the same thing.
"""

from __future__ import annotations

import random

from locust import HttpUser, between, events, task

P95_TARGET_MS = 1000
ERROR_RATE_TARGET = 0.01


def _application(rng: random.Random) -> dict[str, int]:
    """A plausible applicant. Values stay inside the API's validation bounds."""
    delinquency = rng.choice([-1, 0, 0, 0, 1, 2])
    limit = rng.choice([20_000, 50_000, 100_000, 200_000, 500_000])
    payload = {
        "LIMIT_BAL": limit,
        "SEX": rng.choice([1, 2]),
        "EDUCATION": rng.choice([1, 2, 3, 4]),
        "MARRIAGE": rng.choice([1, 2, 3]),
        "AGE": rng.randint(21, 70),
    }
    for column in ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"):
        payload[column] = max(-2, min(9, delinquency + rng.choice([-1, 0, 0, 1])))
    for index in range(1, 7):
        payload[f"BILL_AMT{index}"] = rng.randint(0, limit)
        payload[f"PAY_AMT{index}"] = rng.randint(0, limit // 10)
    return payload


class PredictionUser(HttpUser):
    wait_time = between(0.1, 0.5)

    def on_start(self) -> None:
        self.rng = random.Random()

    @task(10)
    def predict_single(self) -> None:
        body = {"applications": [_application(self.rng)]}
        self.client.post("/predict", json=body, name="/predict [1 row]")

    @task(3)
    def predict_batch(self) -> None:
        """Batches are where latency actually lives -- inference cost is per row."""
        size = self.rng.randint(10, 50)
        body = {"applications": [_application(self.rng) for _ in range(size)]}
        self.client.post("/predict", json=body, name="/predict [10-50 rows]")

    @task(2)
    def health(self) -> None:
        self.client.get("/health", name="/health")

    @task(1)
    def rejects_bad_input(self) -> None:
        """422s are correct behaviour, so this must not count as a failure."""
        bad = {**_application(self.rng), "AGE": 5}
        with self.client.post(
            "/predict",
            json={"applications": [bad]},
            name="/predict [invalid]",
            catch_response=True,
        ) as response:
            if response.status_code == 422:
                response.success()
            else:
                response.failure(f"expected 422, got {response.status_code}")


@events.quitting.add_listener
def _assert_targets(environment, **_kwargs) -> None:
    """Fail the process if the run missed its targets, so CI can use this."""
    stats = environment.stats.total
    p95 = stats.get_response_time_percentile(0.95)
    failure_ratio = stats.fail_ratio

    print("\n" + "=" * 62)
    print(f"  requests        {stats.num_requests}")
    print(f"  failures        {stats.num_failures} ({failure_ratio:.2%})")
    print(f"  median          {stats.median_response_time} ms")
    print(f"  p95             {p95} ms   (target < {P95_TARGET_MS} ms)")
    print(f"  throughput      {stats.total_rps:.1f} req/s")
    print("=" * 62)

    environment.process_exit_code = 0
    if failure_ratio > ERROR_RATE_TARGET:
        print(f"FAIL: error rate {failure_ratio:.2%} exceeds {ERROR_RATE_TARGET:.0%}")
        environment.process_exit_code = 1
    if p95 and p95 > P95_TARGET_MS:
        print(f"FAIL: p95 {p95} ms exceeds {P95_TARGET_MS} ms")
        environment.process_exit_code = 1
