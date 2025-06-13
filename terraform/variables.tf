# terraform/variables.tf

variable "aws_region" {
  description = "The AWS region where resources will be deployed."
  type        = string
  default     = "us-east-1" # Recommended region for Free Tier availability and general purpose.
                           # Check https://aws.amazon.com/free/ for regional availability of services.
}

variable "project_name" {
  description = "A unique prefix for all resources created."
  type        = string
  default     = "mlops-reco-engine"
}

variable "mlflow_db_username" {
  description = "Username for the MLflow RDS database."
  type        = string
  default     = "mlflowadmin"
}

variable "mlflow_db_password" {
  description = "Password for the MLflow RDS database."
  type        = string
  sensitive   = true # Mark as sensitive to prevent output in logs
}

variable "mlflow_db_instance_class" {
  description = "DB instance class for MLflow RDS. Use t2.micro for Free Tier."
  type        = string
  default     = "db.t3.micro" # Free Tier eligible: 750 hours/month
}

variable "mlflow_db_allocated_storage" {
  description = "Allocated storage for the MLflow RDS database in GB."
  type        = number
  default     = 20 # Free Tier eligible: 20 GB of General Purpose (SSD) Storage
}

variable "ec2_instance_type" {
  description = "EC2 instance type for Jenkins/Airflow/MLflow server. Use t2.micro for Free Tier."
  type        = string
  default     = "t2.micro" # Free Tier eligible: 750 hours/month
}

variable "ec2_ami_id" {
  description = "AMI ID for the EC2 instance (e.g., Amazon Linux 2 AMI)."
  type        = string
  default     = "ami-0dc3a08bd93f84a35" # Example: Amazon Linux 2 AMI (HVM) - Kernel 5.10, us-east-1.
                                       # Always verify the latest AMI for your chosen region!
                                       # Search in AWS Console -> EC2 -> AMIs for "Amazon Linux 2 AMI"
}

variable "public_key_path" {
  description = "Path to your SSH public key for EC2 access (e.g., ~/.ssh/id_rsa.pub)."
  type        = string
  default     = "~/.ssh/id_rsa.pub" # Adjust this to your actual public key path
}