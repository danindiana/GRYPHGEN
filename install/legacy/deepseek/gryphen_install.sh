#!/bin/bash

# Update package list and upgrade all packages to their latest versions
sudo apt-get update
sudo apt-get upgrade -y

# Install essential build tools
sudo apt-get install -y build-essential curl wget git

# Install Python and pip
sudo apt-get install -y python3 python3-pip

# Install Docker
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add the repository to Apt sources:
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

# Install Docker Engine, containerd, and Docker Compose.
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Install Kubernetes (kubectl)
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-archive-keyring.gpg
echo "deb [signed-by=/etc/apt/keyrings/kubernetes-archive-keyring.gpg] https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee /etc/apt/sources.list.d/kubernetes.list
sudo apt-get update
sudo apt-get install -y kubectl

# Install ZeroMQ and Python bindings
sudo apt-get install -y libzmq3-dev
pip3 install pyzmq

# Install additional Python packages
pip3 install requests

# Install monitoring and logging tools
sudo apt-get install -y prometheus grafana

# Install security and compliance tools
sudo apt-get install -y vault openssl

# Install database tools
sudo apt-get install -y postgresql mongodb

# Install web servers
sudo apt-get install -y nginx apache2

# Ensure Docker service is running
sudo systemctl start docker
sudo systemctl enable docker

# Output installation completion message
echo "Installation of GRYPHGEN dependencies is complete."
