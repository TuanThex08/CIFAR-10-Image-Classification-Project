#!/bin/bash
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive

# --- 1. System packages ---
apt-get update
apt-get upgrade -y

# Add deadsnakes PPA for Python 3.12 (Ubuntu 22.04 ships 3.10)
apt-get install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y python3.12 python3.12-venv python3.12-dev git

# --- 2. Clone repository ---
APP_DIR="/opt/cifar10-app"
git clone --depth 1 ${github_repo_url} "$APP_DIR"

# --- 3. Python virtual environment & dependencies ---
cd "$APP_DIR"
python3.12 -m venv venv
source venv/bin/activate

# CPU-only PyTorch (~200MB instead of ~2GB CUDA version)
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Remaining dependencies (skip torch/torchvision already installed)
pip install numpy matplotlib seaborn scikit-learn tqdm streamlit

# --- 4. Create dedicated system user ---
useradd --system --no-create-home --shell /bin/false streamlit-app
chown -R streamlit-app:streamlit-app "$APP_DIR"

# --- 5. Create systemd service ---
cat > /etc/systemd/system/streamlit.service << 'UNIT'
[Unit]
Description=CIFAR-10 Streamlit Classifier
After=network.target

[Service]
Type=simple
User=streamlit-app
Group=streamlit-app
WorkingDirectory=/opt/cifar10-app
ExecStart=/opt/cifar10-app/venv/bin/streamlit run streamlit_app.py \
    --server.port=${app_port} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
Restart=always
RestartSec=5
Environment=HOME=/opt/cifar10-app

[Install]
WantedBy=multi-user.target
UNIT

# --- 6. Enable and start service ---
systemctl daemon-reload
systemctl enable streamlit.service
systemctl start streamlit.service

# --- 7. Completion marker ---
echo "Setup complete at $(date)" > /opt/cifar10-app/setup-complete.txt
