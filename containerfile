# Build with: anyscale image build -n pytorch-to-ray-train -f containerfile --ray-version 2.58.0
# Base images: https://docs.anyscale.com/reference/base-images
FROM anyscale/ray:2.58.0-py312-cu128

# Keep these pins identical to requirements.txt.
RUN pip install --no-cache-dir torch==2.14.0 torchvision==0.29.0 tensorboard==2.21.0 mlflow==2.22.5
