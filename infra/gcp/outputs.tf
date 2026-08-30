output "service_url" {
  description = "Public URL of the deployed API."
  value       = google_cloud_run_v2_service.api.uri
}

output "artifact_registry" {
  description = "Docker repository to push images to."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}

output "model_uri" {
  description = "Where the API expects to find its model artifact."
  value       = "gs://${google_storage_bucket.artifacts.name}/models/champion"
}

output "prediction_prefix" {
  description = "Where served predictions are written for drift monitoring."
  value       = "gs://${google_storage_bucket.predictions.name}/served"
}

# The three values below go into the repository's GitHub Actions variables so
# that CD can authenticate without a stored key.
output "github_actions_variables" {
  description = "Set these as repository variables (Settings > Secrets and variables > Actions)."
  value = {
    GCP_PROJECT_ID        = var.project_id
    GCP_REGION            = var.region
    GCP_WIF_PROVIDER      = google_iam_workload_identity_pool_provider.github.name
    GCP_DEPLOY_SA         = google_service_account.deployer.email
    GCS_MODEL_URI         = "gs://${google_storage_bucket.artifacts.name}/models/champion"
    GCS_PREDICTION_PREFIX = "gs://${google_storage_bucket.predictions.name}/served"
  }
}
