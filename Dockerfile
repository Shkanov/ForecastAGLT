FROM python:3.11-slim

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps for building gmdh + common scientific deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    git-lfs \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    # Boost (including Boost.JSON on Debian via this package set)
    libboost-all-dev \
    # sometimes required by native builds
    pkg-config \
 && rm -rf /var/lib/apt/lists/* \
 && git lfs install

# Install Python deps first (better caching)
COPY src/requirements.txt ./requirements.txt
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# Copy the rest
COPY . .

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "src/Portfolio_optimization.py", "--server.port=8501", "--server.address=0.0.0.0"]
