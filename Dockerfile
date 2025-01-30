# Etapa 1: Construcción con Python en ARM64 (para Mac M3)
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt --target /dependencies

# Etapa 2: Imagen de Azure Functions (amd64)
FROM --platform=linux/amd64 mcr.microsoft.com/azure-functions/python:4-python3.12
WORKDIR /home/site/wwwroot

# Copiar dependencias pre-instaladas desde el builder
COPY --from=builder /dependencies /home/site/wwwroot/dependencies

COPY . .

# Ajustar permisos
RUN chmod +x /home/site/wwwroot/run_pipeline.sh

EXPOSE 7071

CMD ["func", "start", "--verbose"]
