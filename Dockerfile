ARG JUPYTER_BASE_IMAGE=quay.io/jupyter/minimal-notebook:python-3.10
FROM ${JUPYTER_BASE_IMAGE}

ARG UV_VERSION=0.11.10
ARG QUBIC_SOFTWARE_REF=c402200c79696e2f5f5fd877a39e5ec80db57eff
ARG QUBIC_DISTRIBUTED_PROCESSOR_REF=b3b856f771dd5624f6079a3b9ffa66970033a5ac
ARG QUBIC_QUBITCONFIG_REF=c766ddbcae76e08ff7dbd770fa1203b4aee224b9
ARG QUBIC_TUTORIAL_REF=578524de38f2beb158c07efb1bb38cb57e6f2c2c
ARG QUBIC_CHIPCALIBRATION_REF=855663adc0e659f765145c4b74cdfd28e345fbbc
ARG LABCHRONICLE_REF=d6d1d280355afedd47295a1f5c0aaf92c83a1c12
ARG RUN_DOCKER_TESTS=false

# Switch to root to install uv and build tools
USER root

# Install build tools as fallback for packages without wheels
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install a pinned uv release for fast Python package installation.
RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"

# Switch back to jovyan user
USER ${NB_UID}

# Clone external repositories and pin them to known revisions.
RUN git clone https://gitlab.com/ShuxiangCao/software.git /home/jovyan/packages/QubiC/software && \
    git -C /home/jovyan/packages/QubiC/software checkout --detach "${QUBIC_SOFTWARE_REF}"
RUN git clone https://gitlab.com/ShuxiangCao/distributed_processor.git /home/jovyan/packages/QubiC/distributed_processor && \
    git -C /home/jovyan/packages/QubiC/distributed_processor checkout --detach "${QUBIC_DISTRIBUTED_PROCESSOR_REF}"
RUN git clone https://gitlab.com/LBL-QubiC/experiments/qubitconfig.git /home/jovyan/packages/QubiC/qubitconfig && \
    git -C /home/jovyan/packages/QubiC/qubitconfig checkout --detach "${QUBIC_QUBITCONFIG_REF}"
RUN git clone https://gitlab.com/LBL-QubiC/experiments/tutorial.git /home/jovyan/packages/QubiC/tutorial && \
    git -C /home/jovyan/packages/QubiC/tutorial checkout --detach "${QUBIC_TUTORIAL_REF}"
RUN git clone https://gitlab.com/LBL-QubiC/experiments/chipcalibration.git /home/jovyan/packages/QubiC/chipcalibration && \
    git -C /home/jovyan/packages/QubiC/chipcalibration checkout --detach "${QUBIC_CHIPCALIBRATION_REF}"
RUN git clone https://github.com/ShuxiangCao/LabChronicle.git /home/jovyan/packages/LabChronicle && \
    git -C /home/jovyan/packages/LabChronicle checkout --detach "${LABCHRONICLE_REF}"

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

# Always verify the installed source compiles; full tests are opt-in for slower builds.
RUN python -m compileall -q /home/jovyan/packages/LeeQ/leeq && \
    if [ "${RUN_DOCKER_TESTS}" = "true" ]; then \
        python -m pytest /home/jovyan/packages/LeeQ/tests; \
    fi

# Add entrypoint script for Jupyter startup
COPY --chown=${NB_UID}:${NB_GID} scripts/docker/entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
