variable "project_id" {
  description = "Target GCP project. Use a dedicated project so teardown and cost attribution are clean."
  type        = string
}

variable "region" {
  description = <<-EOT
    Deployment region. Must stay a US region: Cloud Storage's always-free 5 GB
    allowance applies only to us-east1, us-west1 and us-central1, and Cloud Run's
    free tier covers North America. A region outside this list silently turns a
    zero-cost deployment into a billed one.
  EOT
  type        = string
  default     = "us-central1"

  validation {
    condition     = contains(["us-central1", "us-east1", "us-west1"], var.region)
    error_message = "Region must be us-central1, us-east1 or us-west1 to stay inside the always-free tier."
  }
}

variable "billing_account" {
  description = "Billing account ID, used for the budget alert."
  type        = string
}

variable "budget_amount_inr" {
  description = "Budget ceiling in INR. Alerts fire at 50/90/100% of this."
  type        = number
  default     = 100
}

variable "budget_alert_email" {
  description = "Address that receives budget alerts."
  type        = string
}

variable "github_repository" {
  description = "owner/repo allowed to deploy via Workload Identity Federation."
  type        = string
}

variable "service_name" {
  type    = string
  default = "credit-default-api"
}

variable "image" {
  description = <<-EOT
    Container image to deploy. Defaults to a public placeholder so the first
    `terraform apply` succeeds before any image has been pushed; CD then replaces
    it with the real one.
  EOT
  type        = string
  default     = "us-docker.pkg.dev/cloudrun/container/hello"
}
