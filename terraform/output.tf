# terraform/outputs.tf

output "s3_data_artifacts_bucket_name" {
  description = "Name of the S3 bucket for data and MLflow artifacts."
  value       = aws_s3_bucket.data_artifacts_bucket.bucket
}

output "mlflow_db_endpoint" {
  description = "Endpoint of the MLflow RDS database."
  value       = aws_db_instance.mlflow_db.address
}

output "mlflow_db_port" {
  description = "Port of the MLflow RDS database."
  value       = aws_db_instance.mlflow_db.port
}

output "mlflow_db_username" {
  description = "Username for the MLflow RDS database."
  value       = aws_db_instance.mlflow_db.username
}

output "mlops_server_public_ip" {
  description = "Public IP address of the MLops server EC2 instance."
  value       = aws_instance.mlops_server.public_ip
}

output "mlops_server_public_dns" {
  description = "Public DNS of the MLops server EC2 instance."
  value       = aws_instance.mlops_server.public_dns
}

output "ssh_command" {
  description = "SSH command to connect to the MLops server."
  value = "ssh -i ${var.public_key_path} ec2-user@${aws_instance.mlops_server.public_ip}"
}