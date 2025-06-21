# Multi-Model Recommendation System

A scalable recommendation system built on Amazon review data, featuring multiple recommendation algorithms with MLOps best practices for production deployment.

## 🏗️ Architecture Overview

This system implements three recommendation models (Collaborative Filtering, Content-Based, and Popularity-Based) with a complete MLOps pipeline including data processing, model training, serving, and monitoring.

### Key Components
- **Data Pipeline**: Apache Airflow for orchestrating ETL processes
- **Storage**: S3 (MinIO for local development) for data and model artifacts
- **ML Tracking**: MLflow for experiment tracking and model registry
- **CI/CD**: Jenkins for continuous integration and deployment
- **Model Serving**: Kubernetes-based microservices on EC2
- **Message Queue**: RabbitMQ for real-time inference requests
- **Monitoring**: Prometheus + Grafana for system and model monitoring

## 🛠️ Tech Stack

### Data & ML Infrastructure
- **Apache Airflow**: Workflow orchestration
- **MLflow**: ML lifecycle management
- **Amazon S3 / MinIO**: Object storage
- **PostgreSQL (RDS) / MinIO**: Metadata and MLflow backend store

### CI/CD & Deployment
- **Jenkins**: Continuous integration and deployment
- **Kubernetes**: Container orchestration
- **Docker**: Containerization
- **EC2**: Cloud compute instances
- **RabbitMQ**: Message broker

### Monitoring & Observability
- **Prometheus**: Metrics collection
- **Grafana**: Visualization and alerting
- **Custom metrics**: Model performance tracking
