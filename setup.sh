#!/bin/bash
########################################################

## Shell Script to setup Docker Image on client machine

########################################################

# Assuming linux enviornment
# Updating package, or run in WSL 2 if using windows, follow setup guide
# installing docker
if [[ $(which docker) && $(docker --version) ]]; then
    echo "Update docker"
    sudo apt-get update
    sudo apt-get upgrade
      
  else
    echo "Install docker"
    # safe keeping: uninstall any docker remenant
    sudo apt-get remove docker docker-engine docker.io containerd runc

    # setting up the repository
    sudo apt-get update
    sudo apt-get upgrade
    sudo apt-get install \
        ca-certificates \
        curl \
        gnupg \
        lsb-release
    
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    
    echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt-get update
    sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
fi

# Build the docker image
DOCKER_IMAGE="w210-pyspark-env"
echo "Initiating the docker image: ${DOCKER_IMAGE}"
cd ./
sudo docker-compose up