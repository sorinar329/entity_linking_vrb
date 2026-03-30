# Dockerfile of the base image: https://github.com/IntEL4CoRo/jupyter-ros2/blob/main/Dockerfile
FROM intel4coro/jupyter-ros2:jazzy-py3.12

USER root
ENV PATH=$HOME/.local/bin:$PATH

# Set Working Directory
ENV REPO_DIR=/home/repo
RUN mkdir -p ${REPO_DIR}
WORKDIR ${REPO_DIR}

#============= Install extra software packages =============#
# Install dependencies manually
# Download Mujoco offical assets
RUN git clone --depth=1 https://github.com/google-deepmind/mujoco_menagerie.git

# Upgrade vscode-server
RUN mamba update -y conda-forge::code-server

# Install python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
#===========================================================#

# Copy content to the image
COPY . ${REPO_DIR}/
RUN git config --global --add safe.directory ${REPO_DIR}

# Set JupyterLab default directory
WORKDIR ${REPO_DIR}
# Set VSCODE default directory
ENV CODE_WORKING_DIRECTORY=${REPO_DIR}

# The entrypoint of the docker image
COPY binder/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
