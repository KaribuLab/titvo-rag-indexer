# titvo-rag-indexer

Servicio de indexación RAG (Retrieval-Augmented Generation) de repositorios de código.

Se ejecuta como tarea de AWS Batch a demanda, recibiendo la URL del repositorio (GitHub o Bitbucket, HTTPS o SSH) a través de variables de entorno. Las fuentes se obtienen exclusivamente mediante Git sobre SSH; no existe fallback por API.

## Variables de entorno

| Variable | Descripción | Requerida | Default |
|---|---|---|---|
| `TITVO_REPO_URL` | URL del repositorio a indexar (https/ssh) | Sí | — |
| `TITVO_DYNAMO_CONFIGURATION_TABLE_NAME` | Tabla DynamoDB de configuración | Sí | — |
| `TITVO_ENCRYPTION_KEY_NAME` | Nombre del secreto en Secrets Manager | Sí | — |
| `TITVO_LOG_LEVEL` | Nivel de log (default: `INFO`) | No | `INFO` |
| `AWS_ENDPOINT` | Endpoint de LocalStack para desarrollo local | No | — |
| `TITVO_CHECKPOINT_EVERY_N_FILES` | Cada cuántos archivos procesados se sube un checkpoint DB a S3 | No | `100` |
| `TITVO_CHECKPOINT_KEY` | Template S3 key para checkpoint; placeholders `{repo_host}`, `{owner}`, `{repo}`, `{branch}`, `{commit_sha}` | No | (default template) |
| `TITVO_MAX_SNAPSHOT_MB` | Tamaño máximo permitido para el tarball del snapshot del `.git` antes de abortar | No | `200` |
| `TITVO_LOCK_TTL_MINUTES` | TTL inicial del lock distribuido por branch | No | `360` |
| `TITVO_LOCK_RENEW_INTERVAL_MINUTES` | Cada cuántos minutos se renueva el lock durante runs largos | No | `30` |
| `TITVO_EMBEDDING_BATCH_SIZE` | Tamaño de batch (chunks) por request a OpenAI | No | `1000` |

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

## Resiliencia: checkpointing, source snapshot, lock distribuido

Desde el change `add-rag-indexer-resume-checkpointing`, el indexer puede sobrevivir a interrupciones del job (OOM, timeout, evict) y coordinar runs concurrentes:

- **Checkpoint DB**: cada `TITVO_CHECKPOINT_EVERY_N_FILES` archivos procesados, se sube el sqlite-vec completo a `branches/{branch}/checkpoints/{commit_sha}/index.db`. En el siguiente run, se detecta el checkpoint, se descarga y se reanuda.
- **Tabla `indexed_files`**: tracking persistente de qué archivos ya se procesaron en el commit actual. Permite idempotencia intra-commit.
- **Source snapshot del `.git`**: tarball del local clone en `branches/{branch}/checkpoints/{commit_sha}/repo.tar.gz`. En el resume, se restaura y se skipea `git fetch` + `cat-file blob` para los archivos ya indexados.
- **Lock distribuido en S3**: el primer run adquiere `locks/{branch}.json` con `IfNoneMatch="*"` (atomic). Otros jobs del mismo `(repo, branch)` fallan rápido con `RuntimeError` (cero costo OpenAI). Renovación automática cada 30 min. El body incluye `aws_batch_job_id` para que el agent pueda hacer polling.
- **Batching explícito + streaming**: `embed_iter()` itera chunks uno a uno, los inserta a sqlite-vec de a uno. Memoria pico del bloque: ~10 KB (1 chunk + 1 embedding) vs ~6 MB en el diseño previo.

Ver detalles completos en `docs/rag-indexer.md`.
