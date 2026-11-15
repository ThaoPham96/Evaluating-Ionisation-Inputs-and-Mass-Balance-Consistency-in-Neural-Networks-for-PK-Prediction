# Public upstream image only; no AWS account, credential, worker or entrypoint.
FROM docker.io/pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755
ENV OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /opt/pipk-runtime
COPY base_inventory.json target-pip-list.json conda-explicit.txt pip-overlay.lock apt-runtime.lock verify_environment.py ./
RUN python verify_environment.py --base
# --no-deps prevents a resolver from silently changing the captured base packages.
RUN python -m pip install --no-cache-dir --no-deps --only-binary=:all: --require-hashes --report pip-install-report.json -r pip-overlay.lock
# Same runtime-library families as the historical image; exact OS equivalence is
# not asserted. Retain upstream notices under /usr/share/doc and site-packages.
RUN apt-get update && xargs -r apt-get install -y --no-install-recommends < apt-runtime.lock && rm -rf /var/lib/apt/lists/*
RUN python verify_environment.py --output /opt/pipk-runtime/environment-verification.json
WORKDIR /release
ENTRYPOINT ["python", "-B"]
CMD ["/opt/pipk-runtime/verify_environment.py"]
