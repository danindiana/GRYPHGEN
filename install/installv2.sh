#!/bin/bash

# Install script for GRYPHGEN framework on Ubuntu 22.04

# Exit immediately if a command exits with a non-zero status.
set -e

# Update package list and upgrade all packages to their latest versions
sudo apt-get update
sudo apt-get upgrade -y

# Install necessary dependencies
sudo apt-get install -y build-essential python3-pip python3-venv libzmq3-dev curl

# Install ZeroMQ and Python binding
pip3 install pyzmq

# Install Docker
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker

# Add current user to docker group to run Docker without sudo
sudo usermod -aG docker $USER

# Create and activate a Python virtual environment
python3 -m venv gryphgen-env
source gryphgen-env/bin/activate

# Install required Python packages within the virtual environment
pip install requests

# Verify installations
docker --version
python3 -m zmq --version

# Print completion message
echo "GRYPHGEN framework installation script completed successfully."
echo "Please log out and log back in for Docker group changes to take effect."
