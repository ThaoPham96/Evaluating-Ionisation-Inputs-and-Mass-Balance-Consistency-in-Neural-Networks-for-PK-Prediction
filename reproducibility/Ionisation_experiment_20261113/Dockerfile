FROM docker.io/pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755
ENV OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
WORKDIR /opt/ionisation
COPY requirements-training.txt requirements-scoring.txt requirements-aws.txt ./
RUN python -m pip install --no-cache-dir -r requirements-training.txt -r requirements-aws.txt
RUN apt-get update && apt-get install -y --no-install-recommends libxrender1 libxext6 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY analysis ./analysis
COPY aws/ionisation_fleet.py ./aws/ionisation_fleet.py
COPY config ./config
ENV PYTHONPATH=/opt/ionisation/analysis
ENTRYPOINT ["python", "aws/ionisation_fleet.py"]
