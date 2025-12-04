FROM quay.io/jupyter/minimal-notebook:latest

# Switch to root to install uv and build tools
USER root

# Install build tools as fallback for packages without wheels
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package installation
# uv will prefer prebuilt wheels and avoid compilation
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /home/jovyan/.local/bin/uv /usr/local/bin/uv && \
    mv /home/jovyan/.local/bin/uvx /usr/local/bin/uvx && \
    chmod +x /usr/local/bin/uv /usr/local/bin/uvx

# Switch back to jovyan user
USER ${NB_UID}

# Clone the repository from github
RUN git clone https://gitlab.com/ShuxiangCao/software.git /home/jovyan/packages/QubiC/software
RUN git clone https://gitlab.com/ShuxiangCao/distributed_processor.git /home/jovyan/packages/QubiC/distributed_processor
RUN git clone https://gitlab.com/LBL-QubiC/experiments/qubitconfig.git /home/jovyan/packages/QubiC/qubitconfig
RUN git clone https://gitlab.com/LBL-QubiC/experiments/tutorial.git /home/jovyan/packages/QubiC/tutorial
RUN git clone https://gitlab.com/LBL-QubiC/experiments/chipcalibration.git /home/jovyan/packages/QubiC/chipcalibration
RUN git clone https://github.com/ShuxiangCao/LabChronicle.git /home/jovyan/packages/LabChronicle

# Copy the content of the local src directory to the packagesing directory
COPY --chown=${NB_UID}:${NB_GID} . /home/jovyan/packages/LeeQ
COPY --chown=${NB_UID}:${NB_GID} ./notebooks/* /home/jovyan/notebook_examples

# Temporarily switch to root to fix permissions
USER root
RUN mkdir -p /home/jovyan/.local/share/jupyter/runtime && \
    mkdir -p /home/jovyan/.config/matplotlib && \
    mkdir -p /home/jovyan/.cache && \
    chown -R ${NB_UID}:${NB_GID} /home/jovyan/.local && \
    chown -R ${NB_UID}:${NB_GID} /home/jovyan/.config && \
    chown -R ${NB_UID}:${NB_GID} /home/jovyan/.cache && \
    chmod -R 755 /home/jovyan/.local && \
    chmod -R 755 /home/jovyan/.config && \
    chmod -R 755 /home/jovyan/.cache

# Switch back to jovyan user
USER ${NB_UID}

# Install base requirements using uv (much faster and uses prebuilt wheels)
RUN uv pip install --system ipdb ipywidgets

# Install the QubiC packages using uv
RUN uv pip install --system -e /home/jovyan/packages/QubiC/software
RUN uv pip install --system -e /home/jovyan/packages/QubiC/distributed_processor/python
RUN uv pip install --system -e /home/jovyan/packages/QubiC/qubitconfig
#RUN uv pip install --system -e /home/jovyan/packages/QubiC/chipcalibration

# Install the LabChronicle package using uv
RUN uv pip install --system -e /home/jovyan/packages/LabChronicle

# Copy Docker-specific requirements file
COPY --chown=${NB_UID}:${NB_GID} requirements-docker.txt /home/jovyan/packages/LeeQ/

# Install the requirements using uv (will use prebuilt wheels when available)
# Using requirements-docker.txt to avoid conflicts with packages that don't support NumPy 2.x
RUN uv pip install --system -r /home/jovyan/packages/LeeQ/requirements-docker.txt

# Install the package using uv
RUN uv pip install --system -e /home/jovyan/packages/LeeQ

# Skip pytest in Docker build due to k_agents dependency not available for NumPy 2.x
# Tests should be run locally with proper dependencies
# RUN pytest /home/jovyan/packages/LeeQ

# Add entrypoint script for mode switching (daemon vs jupyter)
COPY --chown=${NB_UID}:${NB_GID} scripts/docker/entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
