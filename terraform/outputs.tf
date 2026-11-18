output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.app.id
}

output "public_ip" {
  description = "Elastic IP address"
  value       = aws_eip.app.public_ip
}

output "app_url" {
  description = "Streamlit application URL"
  value       = "http://${aws_eip.app.public_ip}:${var.app_port}"
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i <your-key>.pem ubuntu@${aws_eip.app.public_ip}"
}

output "setup_status_command" {
  description = "Command to monitor setup progress (run after SSH)"
  value       = "tail -f /var/log/cloud-init-output.log"
}
