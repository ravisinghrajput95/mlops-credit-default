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

      startup_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 10
        period_seconds        = 5
        failure_threshold     = 12
      }

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
