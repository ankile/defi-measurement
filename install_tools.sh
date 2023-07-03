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

# Exit the script
exit 0
