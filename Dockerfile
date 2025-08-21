FROM quay.io/jupyter/minimal-notebook:latest

# Switch to root to install system packages
USER root

# Install build dependencies for Python packages
# Required for packages like NumPy that may need to compile from source
# especially when pre-built wheels are not available for the Python version
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

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
COPY . /home/jovyan/packages/LeeQ
COPY ./notebooks/* /home/jovyan/notebook_examples

# Install the requirements
RUN pip install ipdb ipywidgets

# Install the QubiC package
RUN pip install -e /home/jovyan/packages/QubiC/software
RUN pip install -e /home/jovyan/packages/QubiC/distributed_processor/python
RUN pip install -e /home/jovyan/packages/QubiC/qubitconfig
#RUN pip install -e /home/jovyan/packages/QubiC/chipcalibration

# Install the LabChronicle package
RUN pip install -e /home/jovyan/packages/LabChronicle

# Install the requirements
RUN pip install -r /home/jovyan/packages/LeeQ/requirements.txt

# Install the package
RUN pip install -e /home/jovyan/packages/LeeQ

# Run pytest to test the package
RUN pytest /home/jovyan/packages/LeeQ
