terraform {
  required_version = ">= 1.9"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region

  # The Billing Budgets API refuses requests from user Application Default
  # Credentials unless a quota project is attached. Without these two settings
  # the budget fails to create while everything billable succeeds -- the exact
  # inverse of what this configuration is supposed to guarantee.
  billing_project       = var.project_id
  user_project_override = true
}
