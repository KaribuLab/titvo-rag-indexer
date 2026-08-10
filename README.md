# titvo-rag-indexer

Servicio de indexación RAG (Retrieval-Augmented Generation) de repositorios de código.

Se ejecuta como tarea de AWS Batch a demanda, recibiendo la URL del repositorio (GitHub o Bitbucket, HTTPS o SSH) a través de variables de entorno. Las fuentes se obtienen exclusivamente mediante Git sobre SSH; no existe fallback por API.

## Variables de entorno

| Variable | Descripción | Requerida |
|---|---|---|
| `TITVO_REPO_URL` | URL del repositorio a indexar (https/ssh) | Sí |
| `TITVO_DYNAMO_CONFIGURATION_TABLE_NAME` | Tabla DynamoDB de configuración | Sí |
| `TITVO_ENCRYPTION_KEY_NAME` | Nombre del secreto en Secrets Manager | Sí |
| `TITVO_LOG_LEVEL` | Nivel de log (default: `INFO`) | No |
| `AWS_ENDPOINT` | Endpoint de LocalStack para desarrollo local | No |

## Configuración SSH

La tabla de configuración debe contener, encriptada, la llave del proveedor que se utilizará:

| Parámetro | Proveedor |
|---|---|
| `github_ssh_private_key` | GitHub |
| `bitbucket_ssh_private_key` | Bitbucket |

El servicio desencripta solo la llave correspondiente a la URL recibida. Debe ser una llave de solo
lectura, sin passphrase y con formato PEM válido. La imagen incluye `git`, `openssh-client` y host keys
fijadas para `github.com` y `bitbucket.org`.

## Desarrollo local

```bash
uv sync
uv run python src/main.py
```

Para ejecutar una indexación local, configure primero la llave cifrada en la tabla de configuración y
use una URL admitida, por ejemplo `https://github.com/org/repo` o
`git@bitbucket.org:workspace/repo.git`.

## Infraestructura

La infraestructura está definida en el directorio `aws/` usando Terragrunt:

- `aws/ecr/` — Repositorio ECR
- `aws/ecs/` — Definición de tarea ECS
- `aws/ssm/lookup/` — Lookup de parámetros SSM compartidos
- `aws/ssm/upsert/` — Publicación de ARNs en SSM
