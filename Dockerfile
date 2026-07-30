# AgentCore Runtime requires linux/arm64 images; build with:
#   docker build --platform linux/arm64 -t strictcall .
# The same image runs locally on any architecture (drop the flag).
FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Deterministic demo warehouse baked into the image; nothing checked into git.
RUN python -m strictcall.dataset generate

# Model credentials come in at run time, e.g.
#   docker run -p 8080:8080 -e OPENROUTER_API_KEY -e STRICTCALL_MODEL=openrouter:... strictcall
EXPOSE 8080
CMD ["uvicorn", "strictcall.runtime:app", "--host", "0.0.0.0", "--port", "8080"]
