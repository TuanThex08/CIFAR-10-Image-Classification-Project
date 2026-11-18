variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "ap-southeast-1"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.micro"
}

variable "key_pair_name" {
  description = "Name of an existing EC2 key pair for SSH access"
  type        = string
}

variable "allowed_ssh_cidr" {
  description = "CIDR blocks allowed SSH access (restrict to your IP for security)"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "app_port" {
  description = "Port for Streamlit application"
  type        = number
  default     = 8501
}

variable "github_repo_url" {
  description = "GitHub repository URL to clone"
  type        = string
  default     = "https://github.com/TuanThex08/CIFAR-10-Image-Classification-Project"
}

variable "project_name" {
  description = "Name prefix for AWS resources"
  type        = string
  default     = "cifar10-classifier"
}
