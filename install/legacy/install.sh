#!/bin/bash

# Install script for GRYPHGEN framework on Ubuntu 22.04

# Update package list and upgrade all packages to their latest versions
sudo apt-get update
sudo apt-get upgrade -y

# Install necessary dependencies
sudo apt-get install -y build-essential python3-pip libzmq3-dev curl

# Install ZeroMQ
sudo apt-get install -y libzmq3-dev
pip3 install pyzmq

# Install Docker
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker

# Install Python packages
pip3 install requests

# Verify installations
docker --version
python3 -m zmq --version

# Print completion message
echo "GRYPHGEN framework installation script completed successfully."
