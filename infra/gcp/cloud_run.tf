resource "google_cloud_run_v2_service" "api" {
  name     = var.service_name
  location = var.region

  # Traffic is served only from a revision that passed its startup probe.
  deletion_protection = false

  template {
    service_account = google_service_account.runtime.email

    scaling {
      # Scale to zero is what makes this free: an idle service costs nothing.
      min_instance_count = 0
      max_instance_count = 2
    }

    containers {
      image = var.image

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
        # CPU is allocated only while a request is in flight, which is the
        # billing mode the free tier assumes.
        cpu_idle = true

        # ...but "only while a request is in flight" includes container startup,
        # and this container spends that window importing sklearn, xgboost and
        # mlflow. Throttled, that took 5m07s to reach uvicorn's first log line --
        # before any model loading at all -- so the instance could never pass its
        # own startup probe and every deploy failed with ERROR_CONNECTION_FAILED.
        # The boost is free: it applies only to the startup window, which is
        # exactly the window scale-to-zero guarantees will happen often.
        startup_cpu_boost = true
      }

      env {
        name  = "MODEL_SOURCE"
        value = "gcs"
      }
      env {
        name  = "GCS_MODEL_URI"
        value = "gs://${google_storage_bucket.artifacts.name}/models/champion"
      }
      env {
        name  = "PREDICTION_SINK"
        value = "gcs"
      }
      env {
        name  = "GCS_PREDICTION_PREFIX"
        value = "gs://${google_storage_bucket.predictions.name}/served"
      }

      # Marking api_keys sensitive makes Terraform redact *every* env block on
      # this container in plan output, not just this one -- sensitivity spreads
      # across the block set. That is a real loss of reviewability, and it is
      # still the right trade: the alternative prints the key into plan output
      # and into any CI log that captures it.
      #
      # On exactly when keys are supplied. Deriving it rather than exposing a
      # second flag removes the one combination that bricks a deploy: demanding
      # authentication with no keys, which the API refuses to start under.
      env {
        name  = "REQUIRE_AUTH"
        value = var.api_keys == "" ? "false" : "true"
      }
      env {
        name  = "API_KEYS"
        value = var.api_keys
      }

      ports {
        container_port = 8000
      }

      # The two probes ask different questions and must therefore hit different
      # endpoints. Pointing both at /health -- which this did -- means the only
      # answer available is "the process is up", so Cloud Run cannot tell an
      # instance that is still warming from one that will never serve.
      #
      # /ready is 503 until the model is actually loaded, so no traffic reaches an
      # instance that would only 503 it, and a revision whose model cannot load
      # never takes traffic from the revision already serving.
      #
      # The budget is large because the real cold start was measured, not guessed:
      # 1m40s to uvicorn's first log line and 5m32s to load the model, 7m12s in
      # total. The old 70s budget was never survivable by this container. Note
      # this is a ceiling, not a wait -- a probe that answers sooner proceeds
      # sooner, and the cost of setting it too low is a deploy that fails.
      startup_probe {
        http_get {
          path = "/ready"
        }
        initial_delay_seconds = 10
        period_seconds        = 10
        failure_threshold     = 60
      }

      # Liveness stays on /health, which reports 200 whenever the process can
      # answer at all. Restarting a live process because its model is missing
      # replaces a degraded service with a crash loop.
      liveness_probe {
        http_get {
          path = "/health"
        }
        period_seconds = 30
      }
    }
  }

  depends_on = [google_project_service.required, google_billing_budget.guardrail]
}

# Reachable without a Google identity, which is what makes the demo URL
# shareable. That is not the same as unprotected: with api_keys set, an
# unauthenticated request reaches the container and gets a 401 from the
# application. The two layers answer different questions -- Cloud Run IAM asks
# "may this principal invoke the service", the API asks "which caller is this,
# and record it against the decision". A deployment serving real applicants
# would use both, dropping allUsers in favour of named invokers.
resource "google_cloud_run_v2_service_iam_member" "public" {
  location = google_cloud_run_v2_service.api.location
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
