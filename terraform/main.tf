# terraform/main.tf

terraform {
  required_version = ">= 1.2.0" # Adjust as needed, check latest stable
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.52.0" # Use a compatible version, e.g., current stable is around 5.x
    }
  }
}

# Configure the AWS Provider
provider "aws" {
  region = var.aws_region
}

# --- S3 Bucket for Data & MLflow Artifacts ---
resource "aws_s3_bucket" "data_artifacts_bucket" {
  bucket = "${var.project_name}-data-artifacts" # Globally unique bucket name

  tags = {
    Name    = "${var.project_name}-data-artifacts-bucket"
    Project = var.project_name
  }
}


resource "aws_s3_bucket_versioning" "data_artifacts_bucket_versioning" {
  bucket = aws_s3_bucket.data_artifacts_bucket.id
  versioning_configuration {
    status = "Enabled" # Recommended for data and artifacts
  }
}

# --- RDS Instance for MLflow Tracking Server Backend ---
resource "aws_db_instance" "mlflow_db" {
  allocated_storage    = var.mlflow_db_allocated_storage
  engine               = "postgres"
  engine_version       = "17.4" # Choose a Free Tier compatible version (e.g., 13.x, 14.x)
  instance_class       = var.mlflow_db_instance_class
  identifier           = "${var.project_name}-mlflow-db"
  db_name              = "mlflow_db" # Database name for MLflow
  username             = var.mlflow_db_username
  password             = var.mlflow_db_password
  parameter_group_name = "default.postgres17" # Or "default.postgres14" based on engine_version
  skip_final_snapshot  = true # Set to false in production!
  publicly_accessible  = true # Set to false in production! For this learning project, allows easier access.
  multi_az             = false # Keep false for Free Tier
  storage_type         = "gp2" # General Purpose SSD (Free Tier eligible)
  port                 = 5432 # Default PostgreSQL port

  # Security Group to allow inbound traffic (from your IP for testing)
  vpc_security_group_ids = [aws_security_group.mlflow_db_sg.id]

  tags = {
    Name    = "${var.project_name}-mlflow-db"
    Project = var.project_name
  }
}

# --- EC2 Instance for Jenkins, Airflow, MLflow Server ---
resource "aws_key_pair" "ec2_key_pair" {
  key_name   = "${var.project_name}-ec2-key"
  public_key = file(var.public_key_path) # Reads your public key
}

resource "aws_instance" "mlops_server" {
  ami           = var.ec2_ami_id
  instance_type = var.ec2_instance_type
  key_name      = aws_key_pair.ec2_key_pair.key_name
  associate_public_ip_address = true # Assign a public IP for easy access

  vpc_security_group_ids = [aws_security_group.mlops_server_sg.id]

  tags = {
    Name    = "${var.project_name}-mlops-server"
    Project = var.project_name
  }
}

# --- Security Groups ---
# Security Group for MLflow RDS DB
resource "aws_security_group" "mlflow_db_sg" {
  name        = "${var.project_name}-mlflow-db-sg"
  description = "Allow inbound traffic to MLflow RDS DB"

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    # For simplicity in testing, allow from anywhere.
    # In production, restrict to your EC2 instance's security group or specific IPs.
    cidr_blocks = ["0.0.0.0/0"] # WARNING: Highly insecure for production!
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1" # Allow all outbound
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "${var.project_name}-mlflow-db-sg"
    Project = var.project_name
  }
}

# Security Group for MLops Server (EC2)
resource "aws_security_group" "mlops_server_sg" {
  name        = "${var.project_name}-mlops-server-sg"
  description = "Allow SSH, HTTP, and custom ports to MLops Server"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    # Allow SSH from your current public IP for security
    # You can get your public IP using `curl ifconfig.me`
    cidr_blocks = ["0.0.0.0/0"] # Replace with your actual public IP/32 for production.
  }

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # For general HTTP access
  }

  ingress {
    from_port   = 8080 # Example for Jenkins
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 8081 # For Airflow UI
    to_port     = 8081
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    }

  ingress {
    from_port   = 5000 # Example for MLflow UI
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "${var.project_name}-mlops-server-sg"
    Project = var.project_name
  }
}