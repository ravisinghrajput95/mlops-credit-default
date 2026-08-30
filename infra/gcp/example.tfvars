# Copy to terraform.tfvars and fill in. terraform.tfvars is gitignored.

project_id         = "mlops-credit-default"
region             = "us-central1"
billing_account    = "011C94-83A787-B23026"
budget_alert_email = "you@example.com"
github_repository  = "your-username/mlops-credit-default"

# Budget ceiling in INR. Everything here targets the always-free tier, so this
# is a tripwire, not an expected spend.
budget_amount_inr = 100
