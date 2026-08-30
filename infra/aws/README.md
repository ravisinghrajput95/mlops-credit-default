# AWS deployment — planned, not built

This directory is a placeholder. The GCP deployment in `../gcp` is the one that
works; nothing here has been applied.

It is documented rather than half-implemented because a broken second cloud
target is worse than an honest roadmap.

## Why it is not built yet

**Cost.** GCP was chosen first because Cloud Run scales to zero, so an idle
deployment is genuinely free. AWS has no equivalent for App Runner, and its free
tier is a 12-month trial rather than an always-free allowance. Standing this up
would cost real money every month it stays running.

## The intended mapping

| GCP (built) | AWS (planned) |
| --- | --- |
| Cloud Run | App Runner, or ECS Fargate behind an ALB |
| Artifact Registry | ECR |
| Cloud Storage | S3 |
| Workload Identity Federation | IAM OIDC provider for GitHub Actions |
| Cloud Billing budget | AWS Budgets |

The application needs no changes: `MODEL_SOURCE` and `PREDICTION_SINK` are
already abstractions, so an `S3ParquetSink` alongside the existing
`GCSParquetSink` is the only application-level work.

## Before starting

1. Create an AWS account and configure credentials (`aws configure`).
2. Create the AWS Budgets alarm *first*, as `../gcp/budget.tf` does.
3. Expect this to cost money. Do not start it on a small GCP credit balance.
