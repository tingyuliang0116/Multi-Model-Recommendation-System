# terraform/versions.tf

terraform {
  required_version = ">= 1.2.0" # Adjust as needed, check latest stable
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.52.0" # Use a compatible version, e.g., current stable is around 5.x
    }
  }
}