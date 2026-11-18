FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir .
# Mount a persistent disk at /data (books/, passages.jsonl, aitutor.db).
# Set AITUTOR_ADMIN_PASSWORD, AITUTOR_STUDENT_CODE, AITUTOR_SYNC_KEY and AITUTOR_SECRET. Teachers upload books at /teacher/library.
VOLUME /data
EXPOSE 8000
# $PORT is set by most hosts (Render, Railway, ...); 8000 otherwise
CMD aitutor serve --host 0.0.0.0 --port ${PORT:-8000} --db /data/aitutor.db --passages /data/passages.jsonl --books /data/books
