data "terraform_remote_state" "base" {
  backend = "gcs"

  config = {
    bucket = "fyp-elderly-detection-tofu-state"
    prefix = "bbfc/base"
  }
}

locals {
  base = data.terraform_remote_state.base.outputs
}
