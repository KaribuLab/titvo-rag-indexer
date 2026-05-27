# titvo-rag-indexer

Servicio de indexación RAG (Retrieval-Augmented Generation) de repositorios de código.

Se ejecuta como tarea de AWS ECS a demanda, recibiendo la URL del repositorio (GitHub o Bitbucket, HTTPS o SSH) a través de variables de entorno.

## Variables de entorno

| Variable | Descripción | Requerida |
|---|---|---|
| `TITVO_REPO_URL` | URL del repositorio a indexar (https/ssh) | Sí |
| `TITVO_DYNAMO_CONFIGURATION_TABLE_NAME` | Tabla DynamoDB de configuración | Sí |
| `TITVO_ENCRYPTION_KEY_NAME` | Nombre del secreto en Secrets Manager | Sí |
| `TITVO_LOG_LEVEL` | Nivel de log (default: `INFO`) | No |
| `AWS_ENDPOINT` | Endpoint de LocalStack para desarrollo local | No |

## Desarrollo local

```bash
uv sync
uv run python src/main.py
```

## Infraestructura

La infraestructura está definida en el directorio `aws/` usando Terragrunt:

- `aws/ecr/` — Repositorio ECR
- `aws/ecs/` — Definición de tarea ECS
- `aws/ssm/lookup/` — Lookup de parámetros SSM compartidos
- `aws/ssm/upsert/` — Publicación de ARNs en SSM
