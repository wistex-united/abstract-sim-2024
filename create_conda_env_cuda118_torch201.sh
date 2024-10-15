#!/bin/bash

# Check if conda command exists (Conda is installed)
if ! command -v conda &>/dev/null; then
    echo "Conda is not installed. Installing Conda..."
    mkdir -p ~/miniconda3
    # Fetch install script, currently Linux and MacOS are supported
    if [[ "$(uname -s)" == "Linux" ]]; then
        echo "Detected Linux OS."
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    elif [[ "$(uname -s)" == "Darwin" ]]; then
        echo "Detected macOS."
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda3/miniconda.sh
    else
        echo "Unsupported OS."
        exit 1
    fi
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh
    echo -e "\033[32;1mMiniconda installed.\033[0m"
else
    echo "Conda is already installed."
fi

# Check if Conda is initialized for bash
if [ -f ~/.bashrc ]; then
    if ! grep -q 'conda initialize' ~/.bashrc; then
        echo "Initializing Conda for bash..."
        ~/miniconda3/bin/conda init bash
        echo -e "\033[32;1mConda initialized in bash.\033[0m"
    else
        echo "Conda is already initialized for bash."
    fi
else
    echo "~/.bashrc does not exist. Skipping conda initialization for bash."
fi

# Check if zsh is installed before attempting to initialize Conda for zsh
if command -v zsh &>/dev/null && [ -f ~/.zshrc ]; then
    if ! grep -q 'conda initialize' ~/.zshrc; then
        echo "Initializing Conda for zsh..."
        ~/miniconda3/bin/conda init zsh
        echo -e "\033[32;1mConda initialized in zsh.\033[0m"
    else
        echo "Conda is already initialized for zsh."
    fi
else
    echo "zsh is not installed or ~/.zshrc does not exist. Skipping conda initialization for zsh."
fi

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

ENV_NAME="AbstractSim_cuda118_torch201"
# Check if the environment already exists
if conda env list | grep -q "$ENV_NAME"; then
    echo "The $ENV_NAME environment already exists. Skipping creation."
    echo -e "If you want to create a new one, run \033[34;1mconda env remove -n $ENV_NAME\033[0m"
else
    echo "Creating the $ENV_NAME environment..."
cat <<EOT > "$HOME/$ENV_NAME.yml"
name: $ENV_NAME
channels:
  - nvidia
  - pytorch
  - defaults
dependencies:
  - python=3.9
  - pip
  - cudatoolkit=11.8
  - cudnn=8.9.2.26=cuda11_0
  - pytorch==2.0.1
  - torchvision==0.15.2
  - torchaudio==2.0.2
  - pytorch-cuda=11.8
  - pip:
    - absl-py==2.0.0
    - appdirs==1.4.4
    - cachetools==5.3.2
    - certifi==2023.7.22
    - charset-normalizer==3.2.0
    - click==8.1.7
    - cloudpickle==2.2.1
    - coloredlogs==15.0.1
    - contourpy==1.1.1
    - cycler==0.11.0
    - docker-pycreds==0.4.0
    - Farama-Notifications==0.0.4
    - filelock==3.12.4
    - flatbuffers==23.5.26
    - fonttools==4.42.1
    - gitdb==4.0.10
    - GitPython==3.1.36
    - google-auth==2.23.4
    - google-auth-oauthlib==1.1.0
    - grpcio==1.59.3
    - gymnasium==0.29.1
    - humanfriendly==10.0
    - idna==3.4
    - importlib-metadata==6.8.0
    - importlib-resources==6.1.0
    - Jinja2==3.1.2
    - kiwisolver==1.4.5
    - llvmlite==0.41.1
    - Markdown==3.5.1
    - MarkupSafe==2.1.3
    - matplotlib==3.8.0
    - mpmath==1.3.0
    - networkx==3.1
    - numba==0.58.1
    - numpy==1.26.0
    - oauthlib==3.2.2
    - onnx==1.15.0
    - onnxruntime==1.17.0
    - overrides==7.7.0
    - packaging==23.1
    - pandas==2.1.1
    - pathtools==0.1.2
    - pettingzoo==1.24.1
    - Pillow==10.0.1
    - protobuf==4.23.4
    - psutil==5.9.5
    - pyasn1==0.5.1
    - pyasn1-modules==0.3.0
    - pygame==2.5.2
    - pyparsing==3.1.1
    - python-dateutil==2.8.2
    - pytz==2023.3.post1
    - PyYAML==6.0.1
    - requests==2.31.0
    - requests-oauthlib==1.3.1
    - rsa==4.9
    - sb3-contrib==2.1.0
    - scipy==1.12.0
    - sentry-sdk==1.31.0
    - setproctitle==1.3.2
    - six==1.16.0
    - smmap==5.0.1
    - stable-baselines3==2.1.0
    - SuperSuit==3.9.0
    - sympy==1.12
    - tensorboard==2.15.1
    - tensorboard-data-server==0.7.2
    - tinyscaler==1.2.7
    - typing_extensions==4.8.0
    - tzdata==2023.3
    - urllib3==2.0.5
    - wandb==0.16.0
    - Werkzeug==3.0.1
    - zipp==3.17.0
    - chardet==5.2.0
EOT

    # Create the Conda environment
    conda env create -n $ENV_NAME -f "$HOME/$ENV_NAME.yml"
    rm -f "$HOME/$ENV_NAME.yml"
    if [ $? -ne 0 ]; then
        echo "Failed to create conda environment."
        exit 1
    fi
    echo -e "\033[32;1mThe $ENV_NAME environment has been created.\033[0m"
fi

# Activate the newly created environment
echo -e "To activate the $ENV_NAME environment:"
echo -e ">> \033[34;1mconda activate $ENV_NAME\033[0m"
