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
- `aws/batch/` — Definición de job AWS Batch
- `aws/ssm/lookup/` — Lookup de parámetros SSM compartidos
- `aws/ssm/upsert/` — Publicación de ARNs en SSM

## Despliegue a AWS

El despliegue a AWS se automatiza con el workflow de GitHub Actions
`.github/workflows/deploy-to-aws.yml`, que corre sobre pushes a `main`. Flujo:

1. Quality gates: lint (Ruff) y tests (pytest).
2. Autenticación por **OIDC** (`configure-aws-credentials` con `role-to-assume`).
3. `terragrunt apply` sobre `aws/ecr`.
4. Build y push de la imagen a ECR con tags `:<sha>` y `:latest` (la job definition de AWS Batch
   referencía `${ecr_repository_url}:latest`).
5. `terragrunt run-all apply` sobre `aws/`.

### Secretos requeridos (repo GitHub)

| Secret | Descripción |
|---|---|
| `AWS_TITVO_BATCH_ROLE_TO_ASSUME` | ARN del rol IAM OIDC con permisos de deploy sobre `aws/*` (ECR, Batch, SSM, IAM) |
| `AWS_TITVO_ACCOUNT_ID` | ID de la cuenta AWS (p. ej. `895649849416`) |

### Variables del workflow

| Variable | Valor |
|---|---|
| `AWS_REGION` | `us-east-2` |
| `AWS_STAGE` | `prod` |
| `ECR_REPOSITORY` | `tvo-rag-indexer-ecr-prod` |
| `IMAGE_TAG` | `${{ github.sha }}` |

`serverless.hcl` monta `AWS_REGION`, `AWS_STAGE` y `AWS_ACCOUNT_ID` vía `get_env`, por lo que el apply
de Terragrunt no requiere variables adicionales. Para reproducir el despliegue localmente:
`AWS_REGION=us-east-2 AWS_STAGE=prod AWS_ACCOUNT_ID=<id> terragrunt run-all apply` dentro de `aws/`.
