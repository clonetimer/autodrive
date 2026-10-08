# Optional reproducible CPU deployment recipe. Docker itself was not run in the sandbox.
FROM python:3.11-slim
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir numpy PyYAML Pillow opencv-python-headless pytest matplotlib
COPY . .
ENV OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
CMD ["python", "-m", "pytest", "-q"]
