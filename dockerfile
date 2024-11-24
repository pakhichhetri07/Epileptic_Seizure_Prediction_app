# Use the official Python image as a base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install curl, Java JDK 22, required packages, and Nextflow
RUN set -e && \
    apt-get update && \
    apt-get install -y curl wget && \
    wget https://download.java.net/java/GA/jdk22.0.2/c9ecb94cd31b495da20a27d4581645e8/9/GPL/openjdk-22.0.2_linux-x64_bin.tar.gz && \
    tar -xzf openjdk-22.0.2_linux-x64_bin.tar.gz -C /opt && \
    update-alternatives --install /usr/bin/java java /opt/jdk-22.0.2/bin/java 1 && \
    update-alternatives --install /usr/bin/javac javac /opt/jdk-22.0.2/bin/javac 1 && \
    rm openjdk-22.0.2_linux-x64_bin.tar.gz && \
    pip install --no-cache-dir -r requirements.txt && \
    curl -s https://get.nextflow.io | bash && \
    mv nextflow /usr/local/bin/ && \
    chmod a+rx /usr/local/bin/nextflow

# Set environment variables for Java
ENV JAVA_HOME=/opt/jdk-22.0.2
ENV PATH=$JAVA_HOME/bin:$PATH

# Copy the workflow files
COPY . .

# Expose the port Streamlit will run on
EXPOSE 8080

# Create a startup script
RUN echo '#!/bin/bash\nnextflow run workflow.nf\nstreamlit run app.py --server.port=8080 --server.address=0.0.0.0' > start.sh && \
    chmod +x start.sh

# Set the entrypoint to the startup script
ENTRYPOINT ["./start.sh"]
