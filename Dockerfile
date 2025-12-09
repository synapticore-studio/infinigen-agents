# Multi-stage build: Builder stage for compilation + Runtime stage for execution
FROM python:3.11-slim AS builder

# Python environment setup for UV
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_CACHE_DIR=/tmp/.uv-cache

# System dependencies installation for build stage
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        tzdata \
        build-essential \
        git \
        cmake \
        ninja-build \
        libgtk-3-dev \
        libpng-dev \
        libjpeg-dev \
        libwebp-dev \
        libtiff5-dev \
        libopenexr-dev \
        libopenblas-dev \
        libx11-dev \
        libavutil-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libswresample-dev \
        libssl-dev \
        libva-dev \
        libmfx-dev \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        opencl-headers \
        ocl-icd-opencl-dev \
        xvfb \
        xauth \
        g++ \
        gcc \
        libomp-dev \
        libgomp1 \
        coreutils \
        findutils \
        bash \
        procps \
        # Additional dependencies for terrain compilation
        libc6-dev \
        pkg-config \
        make \
        dos2unix \
        # CUDA dependencies - install NVIDIA CUDA toolkit
        wget \
        software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA toolkit for terrain compilation
RUN wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run -O cuda_installer.run && \
    chmod +x cuda_installer.run && \
    ./cuda_installer.run --no-opengl-libs --no-man-page --no-opengl-libs --override --silent --toolkit && \
    rm cuda_installer.run && \
    echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> /etc/profile && \
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> /etc/profile

# Create user for build process
RUN useradd -ms /bin/bash infinigen

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

# Create virtual environment
RUN uv venv /app/.venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Set up working directory
WORKDIR /app

# Copy project files for build
COPY pyproject.toml uv.lock setup.py ./
COPY infinigen/ ./infinigen/
COPY infinigen_examples/ ./infinigen_examples/
COPY tests/ ./tests/
COPY scripts/ ./scripts/

# Install Python dependencies
RUN sh -c "ulimit -n 4096 && uv sync --frozen --extra terrain --extra vis"

# Compile terrain libraries with CUDA support
RUN chmod +x scripts/install/compile_terrain.sh && \
    dos2unix scripts/install/compile_terrain.sh && \
    dos2unix infinigen/OcMesher/install.sh && \
    bash -c "set -e; cd /app && bash scripts/install/compile_terrain.sh" && \
    echo "Terrain compilation completed successfully with CUDA support"

# Compile Cython terrain libraries
RUN bash -c "source .venv/bin/activate && python setup.py build_ext --inplace" && \
    echo "Cython terrain compilation completed successfully"

# Install additional runtime dependencies
RUN bash -c "source .venv/bin/activate && pip install PyOpenGL-accelerate"

# Install OpenVINO and MCP
RUN sh -c "ulimit -n 4096 && uv add openvino==2025.2.0.0 openvino-tokenizers==2025.2.0.0 openvino-genai==2025.2.0.0" && \
    sh -c "ulimit -n 4096 && uv add mcp"

# Set ownership
RUN chown -R infinigen:infinigen /app/.venv && \
    chown -R infinigen:infinigen /root/.local

# ============================================
# Runtime stage - minimal image for execution
# ============================================
FROM python:3.11-slim AS runtime

# Runtime environment setup
ENV UV_COMPILE_BYTECODE=1
ENV UV_CACHE_DIR=/tmp/.uv-cache
ENV DISPLAY=:99
ENV BLENDER_HEADLESS=1

# Minimal runtime dependencies with NVIDIA GPU support
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        tzdata \
        libx11-6 \
        libglib2.0-0 \
        libgtk-3-0 \
        libgomp1 \
        libomp5 \
        libopenblas0 \
        libpng16-16 \
        libjpeg62-turbo \
        libtiff6 \
        libwebp7 \
        libopenexr-3-1-30 \
        libssl3 \
        libgstreamer1.0-0 \
        libgstreamer-plugins-base1.0-0 \
        opencl-headers \
        ocl-icd-opencl-dev \
        xvfb \
        xauth \
        bash \
        procps \
    && rm -rf /var/lib/apt/lists/*

# Create infinigen user
RUN useradd -ms /bin/bash infinigen

# Copy built artifacts from builder stage
COPY --from=builder /app /app
COPY --from=builder /root/.local /root/.local

# Set up environment
ENV PATH="/app/.venv/bin:/root/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
ENV VIRTUAL_ENV=/app/.venv

# Set up working directory
WORKDIR /app

# Create startup scripts
RUN echo '#!/bin/bash\nXvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &\nexport DISPLAY=:99\nexec "$@"' > /usr/local/bin/start-headless.sh && \
    chmod +x /usr/local/bin/start-headless.sh

RUN echo '#!/bin/bash\ncd /app/infinigen_examples\nexport DISPLAY=:99\nexport PATH="/app/.venv/bin:/usr/local/bin:/usr/bin:/bin:$PATH"\nexec /app/.venv/bin/python mcp_server.py --transport streamable-http --host 0.0.0.0 --port 8080' > /usr/local/bin/start-mcp-server.sh && \
    chmod +x /usr/local/bin/start-mcp-server.sh && \
    chown infinigen:infinigen /usr/local/bin/start-mcp-server.sh

# Configure bashrc for infinigen user
RUN echo 'source /app/.venv/bin/activate' >> /home/infinigen/.bashrc && \
    echo 'export PATH="/app/.venv/bin:/usr/local/bin:/usr/bin:/bin:$PATH"' >> /home/infinigen/.bashrc && \
    echo 'export UV_CACHE_DIR=/tmp/.uv-cache' >> /home/infinigen/.bashrc && \
    echo 'export DISPLAY=:99' >> /home/infinigen/.bashrc

# Create logs directory
RUN mkdir -p /app/logs && chown infinigen:infinigen /app/logs

# Switch to infinigen user
USER infinigen

# Default command
CMD ["/usr/local/bin/start-mcp-server.sh"]