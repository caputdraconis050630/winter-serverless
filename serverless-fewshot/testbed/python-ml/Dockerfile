FROM python:3.11-slim
RUN pip install --no-cache-dir numpy pandas
WORKDIR /app
COPY app.py .
ENV PORT=8080
EXPOSE 8080
CMD ["python", "app.py"]
