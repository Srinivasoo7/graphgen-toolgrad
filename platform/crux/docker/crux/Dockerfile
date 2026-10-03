FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY crux ./crux
COPY docs ./docs
RUN pip install --no-cache-dir .
VOLUME /data/crux
EXPOSE 8788
CMD ["python", "-m", "crux", "entry"]
