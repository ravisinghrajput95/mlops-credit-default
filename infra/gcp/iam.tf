# Workload Identity Federation: GitHub Actions presents a short-lived OIDC token
# which GCP exchanges for credentials. No service-account key is ever created,
# so there is no long-lived secret to leak, rotate, or accidentally commit.

resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "github-pool"
  display_name              = "GitHub Actions"

  depends_on = [google_project_service.required]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-provider"
  display_name                       = "GitHub OIDC"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
  }

  # Without this condition, any GitHub repository in the world could exchange a
  # token for credentials in this project.
  attribute_condition = "assertion.repository == '${var.github_repository}'"

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

resource "google_service_account" "deployer" {
  depends_on   = [google_project_service.required]
  account_id   = "github-deployer"
  display_name = "GitHub Actions deployer"
}

resource "google_service_account_iam_member" "github_impersonation" {
  service_account_id = google_service_account.deployer.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repository}"
}

# Least privilege: push images, deploy revisions, act as the runtime account.
# Deliberately not roles/editor, which is the usual shortcut.
resource "google_project_iam_member" "deployer_roles" {
  for_each = toset([
    "roles/artifactregistry.writer",
    "roles/run.developer",
    "roles/storage.objectViewer",
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.deployer.email}"
}

resource "google_service_account" "runtime" {
  depends_on   = [google_project_service.required]
  account_id   = "credit-default-api"
  display_name = "Cloud Run runtime identity"
}

resource "google_service_account_iam_member" "deployer_acts_as_runtime" {
  service_account_id = google_service_account.runtime.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.deployer.email}"
}

# The runtime reads its model and writes prediction logs. Scoped to the two
# buckets rather than granted project-wide.
resource "google_storage_bucket_iam_member" "runtime_reads_model" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_storage_bucket_iam_member" "runtime_writes_predictions" {
  bucket = google_storage_bucket.predictions.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.runtime.email}"
}

# Cloud Run writes a container's stdout and stderr to Cloud Logging as the
# *runtime* identity, so a service account without this role produces a revision
# that logs absolutely nothing -- not an error, not a stack trace, not uvicorn's
# startup banner. The failure is silent in both directions: the deploy fails on a
# startup probe, and the logs that would explain why were dropped on the floor.
#
# This does not bite the default compute service account, which carries
# roles/editor and therefore this permission by inheritance. It bites exactly the
# least-privilege custom account this file is careful to create, which is what
# made it invisible until a real apply.
resource "google_project_iam_member" "runtime_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.runtime.email}"
}
