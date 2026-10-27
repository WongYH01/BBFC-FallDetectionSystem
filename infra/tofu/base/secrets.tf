resource "google_secret_manager_secret" "secret" {
  for_each  = toset(["telegram-bot-token", "detection-stream-password", "tailscale-operator-oauth"])
  secret_id = each.key

  replication {
    auto {}
  }
}
