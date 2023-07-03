#!/bin/bash

# Install Zsh
sudo apt update
sudo apt install zsh -y

# Install Oh My Zsh
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# Install Tmux
sudo apt install tmux -y

# Set Zsh as the default shell
chsh -s $(which zsh)

# Set up a default tmux configuration
echo "set-option -g default-shell /bin/zsh" >> ~/.tmux.conf

# Print installation completed message
echo "Zsh, Oh My Zsh, and Tmux installation completed."

# Reload the terminal
source ~/.zshrc

# Install Python
sudo apt install python3.11 -y
sudo apt install python3.11-venv -y

# Make a virtual environment
python3.11 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip and install the requirements
pip install --upgrade pip
pip install -r requirements.txt

# Exit the script
exit 0
