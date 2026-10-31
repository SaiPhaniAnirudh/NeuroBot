FROM python:3.12-slim

# Set up user with UID 1000 (Hugging Face standard)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Install dependencies as user
COPY --chown=user:user backend/requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend and ui
COPY --chown=user:user backend/ ./backend/
COPY --chown=user:user ui/ ./ui/

EXPOSE 7860

# Run FastAPI serving backend and mounted static UI on port 7860
CMD ["uvicorn", "main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "7860"]
