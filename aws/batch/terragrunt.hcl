terraform {
  source = "git::https://github.com/KaribuLab/terraform-aws-batch.git?ref=v0.3.0"
}

locals {
  serverless  = read_terragrunt_config(find_in_parent_folders("serverless.hcl"))
  batch_name  = "${local.serverless.locals.service_name}-batch-${local.serverless.locals.stage}"
  common_tags = local.serverless.locals.common_tags
  base_path   = "${local.serverless.locals.parameter_path}/${local.serverless.locals.stage}"
}

include {
  path = find_in_parent_folders()
}

dependency ecr {
  config_path = "${get_parent_terragrunt_dir()}/aws/ecr"
  mock_outputs = {
    ecr_repository_url = "00000000000.dkr.ecr.us-east-1.amazonaws.com/tvo-rag-indexer-test"
  }
}

dependency parameters {
  config_path = "${get_parent_terragrunt_dir()}/aws/ssm/lookup"
  mock_outputs = {
    parameters = {
      "/tvo/security-scan/test/infra/vpc/vpc_id"                   = "vpc-000000000000000"
      "/tvo/security-scan/test/infra/vpc/subnets/private"          = "[\"subnet-0c4b3b6b1b7b3b3b3\"]"
      "/tvo/security-scan/test/infra/dynamo/parameter-table-name"  = "tvo-configuration-table"
      "/tvo/security-scan/test/infra/dynamo/parameter-table-arn"   = "arn:aws:dynamodb:us-east-1:000000000000:table/tvo-configuration-table-test"
      "/tvo/security-scan/test/infra/kms/encryption-key-name"      = "/tvo/security-scan/test/infra/encryption-key"
      "/tvo/security-scan/test/infra/secret/manager/arn"           = "arn:aws:secretsmanager:us-east-1:000000000000:secret:/tvo/security-scan/test"
      "/tvo/security-scan/prod/infra/vpc/vpc_id"                   = "vpc-000000000000000"
      "/tvo/security-scan/prod/infra/vpc/subnets/private"          = "[\"subnet-0c4b3b6b1b7b3b3b3\"]"
      "/tvo/security-scan/prod/infra/dynamo/parameter-table-name"  = "tvo-configuration-table"
      "/tvo/security-scan/prod/infra/dynamo/parameter-table-arn"   = "arn:aws:dynamodb:us-east-1:000000000000:table/tvo-configuration-table-prod"
      "/tvo/security-scan/prod/infra/kms/encryption-key-name"      = "/tvo/security-scan/prod/infra/encryption-key"
      "/tvo/security-scan/prod/infra/secret/manager/arn"           = "arn:aws:secretsmanager:us-east-1:000000000000:secret:/tvo/security-scan/prod"
      "/tvo/security-scan/test/infra/s3/rag-code/bucket_arn"       = "arn:aws:s3:::tvo-rag-index-test-us-east-1"
      "/tvo/security-scan/prod/infra/s3/rag-code/bucket_arn"       = "arn:aws:s3:::tvo-rag-index-prod-us-east-1"
    }
  }
}

inputs = {
  subnet_ids         = jsondecode(dependency.parameters.outputs.parameters["${local.base_path}/infra/vpc/subnets/private"])
  name               = local.batch_name
  common_tags        = local.common_tags
  ecr_repository_url = dependency.ecr.outputs.ecr_repository_url
  max_vcpus          = 4
  job_vcpu           = 1
  job_memory         = 2048
  job_command        = ["python", "main.py"]
  vpc_id             = dependency.parameters.outputs.parameters["${local.base_path}/infra/vpc/vpc_id"]
  job_environment = [
    {
      name  = "TITVO_DYNAMO_CONFIGURATION_TABLE_NAME"
      value = dependency.parameters.outputs.parameters["${local.base_path}/infra/dynamo/parameter-table-name"]
    },
    {
      name  = "TITVO_ENCRYPTION_KEY_NAME"
      value = dependency.parameters.outputs.parameters["${local.base_path}/infra/kms/encryption-key-name"]
    },
    {
      name  = "TITVO_LOG_LEVEL"
      value = "INFO"
    }
  ]
  job_policy = jsonencode({
    "Version" : "2012-10-17",
    "Statement" : [
      {
        "Effect" : "Allow",
        "Action" : [
          "dynamodb:GetItem",
        ],
        "Resource" : [
          dependency.parameters.outputs.parameters["${local.base_path}/infra/dynamo/parameter-table-arn"],
        ]
      },
      {
        "Effect" : "Allow",
        "Action" : [
          "secretsmanager:GetSecretValue"
        ],
        "Resource" : [
          dependency.parameters.outputs.parameters["${local.base_path}/infra/secret/manager/arn"]
        ]
      },
      {
        "Effect" : "Allow",
        "Action" : [
          "s3:GetObject",
          "s3:PutObject",
          "s3:CopyObject",
          "s3:ListBucket"
        ],
        "Resource" : [
          dependency.parameters.outputs.parameters["${local.base_path}/infra/s3/rag-code/bucket_arn"],
          "${dependency.parameters.outputs.parameters["${local.base_path}/infra/s3/rag-code/bucket_arn"]}/*"
        ]
      }
    ]
  })
}
