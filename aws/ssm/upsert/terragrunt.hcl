terraform {
  source = "git::https://github.com/KaribuLab/terraform-aws-parameter-upsert.git?ref=v0.7.2"
}

locals {
  serverless  = read_terragrunt_config(find_in_parent_folders("serverless.hcl"))
  base_path   = "${local.serverless.locals.parameter_path}/${local.serverless.locals.stage}/infra"
  common_tags = local.serverless.locals.common_tags
}

dependency "ecs" {
  config_path = "${get_parent_terragrunt_dir()}/aws/ecs"
  mock_outputs = {
    task_definition_arn    = "arn:aws:ecs:us-east-1:012345678901:task-definition/tvo-rag-indexer-ecs-test:1"
    task_definition_family = "tvo-rag-indexer-ecs-test"
    cluster_arn            = "arn:aws:ecs:us-east-1:012345678901:cluster/tvo-rag-indexer-cluster-test"
    cluster_name           = "tvo-rag-indexer-cluster-test"
    security_group_id      = "sg-123131231321"
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
  binary_version = "v0.7.2"
  tags           = local.common_tags
  parameters = [
    {
      path        = "ecs/rag-indexer/task_definition_arn"
      type        = "String"
      tier        = "Standard"
      description = "RAG Indexer ECS Task Definition ARN"
      value       = join(":", slice(split(":", dependency.ecs.outputs.task_definition_arn), 0, length(split(":", dependency.ecs.outputs.task_definition_arn)) - 1))
    },
    {
      path        = "ecs/rag-indexer/task_definition_family"
      type        = "String"
      tier        = "Standard"
      description = "RAG Indexer ECS Task Definition Family"
      value       = dependency.ecs.outputs.task_definition_family
    },
    {
      path        = "ecs/rag-indexer/cluster_arn"
      type        = "String"
      tier        = "Standard"
      description = "RAG Indexer ECS Cluster ARN"
      value       = dependency.ecs.outputs.cluster_arn
    },
    {
      path        = "ecs/rag-indexer/cluster_name"
      type        = "String"
      tier        = "Standard"
      description = "RAG Indexer ECS Cluster Name"
      value       = dependency.ecs.outputs.cluster_name
    },
    {
      path        = "ecs/rag-indexer/security_group_id"
      type        = "String"
      tier        = "Standard"
      description = "RAG Indexer ECS Security Group ID"
      value       = dependency.ecs.outputs.security_group_id
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
    {
      path        = "s3/rag-index-bucket"
      type        = "String"
      tier        = "Standard"
      description = "S3 Bucket for RAG index databases"
      value       = "tvo-rag-index-${local.serverless.locals.stage}-${local.serverless.locals.region}"
    }
  ]
}
