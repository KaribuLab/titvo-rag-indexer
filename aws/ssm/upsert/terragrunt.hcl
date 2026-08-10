terraform {
  source = "git::https://github.com/KaribuLab/terraform-aws-parameter-upsert.git?ref=v0.7.12"
}

locals {
  serverless  = read_terragrunt_config(find_in_parent_folders("serverless.hcl"))
  base_path   = "${local.serverless.locals.parameter_path}/${local.serverless.locals.stage}/infra"
  common_tags = local.serverless.locals.common_tags
}

dependency "batch" {
  config_path = "${get_parent_terragrunt_dir()}/aws/batch"
  mock_outputs = {
    job_definition_arn  = "arn:aws:batch:us-east-1:012345678901:job-definition/tvo-rag-indexer-batch-test:1"
    job_definition_name = "tvo-rag-indexer-batch-test"
    job_queue_arn       = "arn:aws:batch:us-east-1:012345678901:job-queue/tvo-rag-indexer-batch-test"
    job_queue_name      = "tvo-rag-indexer-batch-test"
    security_group_id   = "sg-123131231321"
  }
}

dependency "ecr" {
  config_path = "${get_parent_terragrunt_dir()}/aws/ecr"
  mock_outputs = {
    ecr_repository_url = "012345678901.dkr.ecr.us-east-1.amazonaws.com/tvo-rag-indexer-ecr-test"
    ecr_repository_arn = "arn:aws:ecr:us-east-1:012345678901:repository/tvo-rag-indexer-ecr-test"
  }
}

include {
  path = find_in_parent_folders()
}

inputs = {
  base_path      = local.base_path
  binary_version = "v0.7.12"
  tags           = local.common_tags
  parameters = [
    {
      path        = "batch/rag-indexer/job_definition_arn"
      type        = "String"
      tier        = "Standard"
      description = "RAG Indexer Batch Job Definition ARN"
      value       = join(":", slice(split(":", dependency.batch.outputs.job_definition_arn), 0, length(split(":", dependency.batch.outputs.job_definition_arn)) - 1))
    },
    {
      path        = "batch/rag-indexer/job_definition_name"
      type        = "String"
      tier        = "Standard"
      description = "RAG Indexer Batch Job Definition Name"
      value       = dependency.batch.outputs.job_definition_name
    },
    {
      path        = "batch/rag-indexer/job_queue_arn"
      type        = "String"
      tier        = "Standard"
      description = "RAG Indexer Batch Job Queue ARN"
      value       = dependency.batch.outputs.job_queue_arn
    },
    {
      path        = "batch/rag-indexer/job_queue_name"
      type        = "String"
      tier        = "Standard"
      description = "RAG Indexer Batch Job Queue Name"
      value       = dependency.batch.outputs.job_queue_name
    },
    {
      path        = "batch/rag-indexer/security_group_id"
      type        = "String"
      tier        = "Standard"
      description = "RAG Indexer Batch Security Group ID"
      value       = dependency.batch.outputs.security_group_id
    },
    {
      path        = "ecr-registry-url"
      type        = "String"
      tier        = "Standard"
      description = "ECR Registry URL"
      value       = dependency.ecr.outputs.ecr_repository_url
    },
    {
      path        = "ecr-registry-arn"
      type        = "String"
      tier        = "Standard"
      description = "ECR Repository ARN"
      value       = dependency.ecr.outputs.ecr_repository_arn
    },
  ]
}
