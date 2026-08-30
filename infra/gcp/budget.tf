# The budget is defined first and deliberately has no dependency on anything
# else, so `terraform apply` creates the cost guardrail before it creates
# anything that can spend money.
#
# Everything else in this configuration is designed to sit inside GCP's
# always-free tier, so the expected bill is zero. The budget exists to catch the
# case where that assumption is wrong -- a misconfigured region, a runaway
# retry loop, a resource left running after a demo.

resource "google_billing_budget" "guardrail" {
  billing_account = var.billing_account
  display_name    = "credit-default-mlops guardrail"

  budget_filter {
    projects = ["projects/${data.google_project.this.number}"]
  }

  amount {
    specified_amount {
      currency_code = "INR"
      units         = tostring(var.budget_amount_inr)
    }
  }

  # Warn early, not just once the money is gone.
  threshold_rules {
    threshold_percent = 0.5
  }
  threshold_rules {
    threshold_percent = 0.9
  }
  threshold_rules {
    threshold_percent = 1.0
  }

  all_updates_rule {
    monitoring_notification_channels = [google_monitoring_notification_channel.budget_email.id]
    disable_default_iam_recipients   = false
  }
}

resource "google_monitoring_notification_channel" "budget_email" {
  display_name = "Budget alerts"
  type         = "email"

  labels = {
    email_address = var.budget_alert_email
  }
}

data "google_project" "this" {
  project_id = var.project_id
}
