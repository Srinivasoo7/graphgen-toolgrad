FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir "headroom-ai[proxy]" \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8787
# Compress-only worker. CCR off. Not published to the host.
CMD ["headroom", "proxy", "--host", "0.0.0.0", "--port", "8787", "--no-ccr", "--disable-kompress"]
