FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

# --- OS deps for PyRep/RLBench + X11/GL runtime ---
RUN apt-get update && apt-get install -y \
    python3.8 python3.8-dev python3-pip python3-venv \
    git wget ca-certificates \
    build-essential cmake pkg-config \
    libffi-dev \
    libglib2.0-0 \
    libsm6 libxrender1 libxext6 libxi6 libxrandr2 libxinerama1 libxcursor1 \
    libxkbcommon-x11-0 \
    nano python-is-python3 \
    libx11-xcb1 libxcb1 libxcb-randr0 libxcb-xinerama0 libxcb-icccm4 libxcb-image0 \
    libxcb-keysyms1 libxcb-render-util0 libxcb-xkb1 libxcb-cursor0 \
    libfontconfig1 libfreetype6 libdbus-1-3 \
    libgl1 libglvnd0 libegl1 libglx0 \
    libavcodec-dev libavformat-dev libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# --- CoppeliaSim 4.1.0 ---
ENV COPPELIASIM_ROOT=/opt/CoppeliaSim
RUN mkdir -p ${COPPELIASIM_ROOT} && \
    wget -q https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -O /tmp/cs.tar.xz && \
    tar -xf /tmp/cs.tar.xz -C ${COPPELIASIM_ROOT} --strip-components 1 && \
    rm -f /tmp/cs.tar.xz

# --- Persistent env vars for PyRep/CoppeliaSim ---
ENV LD_LIBRARY_PATH=${COPPELIASIM_ROOT}
ENV QT_QPA_PLATFORM_PLUGIN_PATH=${COPPELIASIM_ROOT}
ENV QT_QPA_PLATFORM=xcb
ENV XDG_RUNTIME_DIR=/tmp/runtime-root
RUN mkdir -p /tmp/runtime-root && chmod 700 /tmp/runtime-root

# --- Python deps + PyRep + RLBench + SB3 (pinned to avoid conflicts) ---
RUN python3.8 -m pip install --upgrade pip setuptools wheel && \
    python3.8 -m pip install --no-cache-dir \
        "gymnasium==0.29.1" \
        "shimmy>=1.3.0" && \
    python3.8 -m pip install --no-cache-dir "cffi==1.14.2" && \
    python3.8 -m pip install --no-cache-dir --no-build-isolation \
        "pyrep @ git+https://github.com/stepjam/PyRep.git" && \
    python3.8 -m pip install --no-cache-dir \
        "rlbench @ git+https://github.com/stepjam/RLBench.git" && \
    # ---- SB3 stack (avoid stable-baselines3[extra] because it drags opencv-python Qt) ----
    python3.8 -m pip install --no-cache-dir \
        "torch==2.2.2" \
        "stable-baselines3==2.4.0" && \
    # If you need SB3 "extras", install selectively (safe for RLBench GUI)
    python3.8 -m pip install --no-cache-dir \
        "tensorboard>=2.10" \
        "tqdm" \
        "matplotlib" \
        "pandas" && \
    # Ensure OpenCV is headless (no Qt plugins)
    python3.8 -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless || true && \
    python3.8 -m pip install --no-cache-dir "opencv-python-headless<5"

WORKDIR /workspace
CMD ["bash"]
