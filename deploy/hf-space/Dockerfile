# Hugging Face Space image for the AegisMem demo API.
# Installs the aegismem package (core deps only) straight from the public repo,
# then serves the seeded demo app. The seeded retrieval path is pure Python, so
# no ML/torch dependencies are needed — the image is small and builds fast.
FROM python:3.12-slim

# Non-root user (Hugging Face Spaces convention).
RUN useradd --create-home --uid 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

# Core package from GitHub. Pin to a tag/commit for a reproducible Space build.
RUN pip install --no-cache-dir --user \
    "git+https://github.com/sobarslap/Redoubt.git@main"

COPY --chown=user app.py .

EXPOSE 7860
CMD ["python", "app.py"]
