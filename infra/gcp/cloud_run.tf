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

# Public read access. This is a portfolio demo with no sensitive data; a real
# credit-scoring endpoint would sit behind authentication.
resource "google_cloud_run_v2_service_iam_member" "public" {
  location = google_cloud_run_v2_service.api.location
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
