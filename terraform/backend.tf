# terraform/backend.tf

terraform {
  backend "s3" {
    bucket         = "mlops-recommendation-engine-terraform-state-bucket" # Replace with a globally unique S3 bucket name
    key            = "mlops-recommendation-engine/terraform.tfstate"
    region         = "us-east-1" # Choose a region for your state bucket. us-east-1 is common.
    encrypt        = true
    dynamodb_table = "mlops-recommendation-engine-terraform-state-lock-table"
  }
}