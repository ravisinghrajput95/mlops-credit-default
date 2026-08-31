# Services must be enabled before anything can use them. disable_on_destroy is
# false so that `terraform destroy` does not disable APIs that other work in the
# project might depend on.
resource "google_project_service" "required" {
  for_each = toset([
    "run.googleapis.com",
    "iam.googleapis.com",
    "serviceusage.googleapis.com",
    "artifactregistry.googleapis.com",
    "storage.googleapis.com",
    "iamcredentials.googleapis.com",
    "monitoring.googleapis.com",
    "billingbudgets.googleapis.com",
    "cloudresourcemanager.googleapis.com",
  ])

  service            = each.value
  disable_on_destroy = false
}
