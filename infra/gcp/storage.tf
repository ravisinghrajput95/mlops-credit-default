# All buckets are regional in a US region, which is what the always-free 5 GB
# allowance requires. Total project usage is a few tens of MB.

resource "google_storage_bucket" "artifacts" {
  name     = "${var.project_id}-mlops-artifacts"
  location = var.region

  # Cheaper and safer than object versioning for a demo: nothing here is
  # irreplaceable, and force_destroy keeps `terraform destroy` from failing on a
  # non-empty bucket, which is the usual reason teardown gets abandoned halfway.
  force_destroy               = true
  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [google_project_service.required]
}

resource "google_storage_bucket" "predictions" {
  name     = "${var.project_id}-mlops-predictions"
  location = var.region

  force_destroy               = true
  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [google_project_service.required]
}

resource "google_artifact_registry_repository" "images" {
  location      = var.region
  repository_id = "credit-default"
  format        = "DOCKER"
  description   = "API container images"

  # The always-free allowance is 0.5 GB, and the API image is ~289 MB, so old
  # revisions are pruned aggressively rather than accumulating into a bill.
  cleanup_policies {
    id     = "keep-recent"
    action = "KEEP"
    most_recent_versions {
      keep_count = 2
    }
  }

  cleanup_policies {
    id     = "delete-old"
    action = "DELETE"
    condition {
      older_than = "604800s" # 7 days
    }
  }

  depends_on = [google_project_service.required]
}
