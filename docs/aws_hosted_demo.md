# AWS hosted demo

## Problem and scope

The repository's stated goal is to demonstrate ticket classification, generated reply drafting, and a required human decision. The local workflow already implements that path. Hosting the same path on AWS requires a durable review store, a public HTTPS entry point, and server-side secrets. This deployment remains an **educational demo with fictional data**. It does not establish model quality, authenticated reviewer identity, a real ticket-system connection, or production approval.

Success criteria for this branch:

- One HTTPS URL serves the browser client and API.
- Submission creates a pending draft; only approval writes a response.
- A second decision conflicts, and records survive service replacement.
- No API key is committed to Git or baked into the container image.
- The existing SQLite local workflow and tests still run.

## Architecture

```text
Browser (HTTPS)
    → Amazon ECS Express Mode (static client + FastAPI + local demo model artifacts)
        → DynamoDB drafts and responses tables
        → Secrets Manager submission and reviewer keys
Container image → private Amazon ECR repository
```

ECS tasks can be replaced, so the hosted workflow selects DynamoDB instead of container-local SQLite. The DynamoDB adapter uses one atomic transaction to update a draft and write its approved response. [ECS Express Mode architecture](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/express-service-overview.html); [DynamoDB transaction semantics](https://docs.aws.amazon.com/amazondynamodb/latest/APIReference/API_TransactWriteItems.html).

CloudFormation provisions the tables, generated secrets, ECS execution and task roles, infrastructure role, and Express Mode service. The service references Secrets Manager ARNs as environment variables and uses `/health` for HTTP health checks. [Express Mode CloudFormation resource](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-ecs-expressgatewayservice.html); [container secrets](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-properties-ecs-expressgatewayservice-secret.html); [IAM roles](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/express-service-getting-started.html#express-service-getting-started-roles).

## Deploy

Prerequisites: an AWS account with permission to use ECR, CloudFormation, IAM, Secrets Manager, DynamoDB, and ECS Express Mode; AWS CLI authentication; Docker daemon; a chosen AWS Region; and a default VPC with public subnets. Review AWS pricing and service availability in that Region before deploying. The account and Region are intentionally not hardcoded. [Express Mode prerequisites](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/express-service-getting-started.html#express-service-getting-started-prerequisites).

From a clean commit on this branch:

```bash
export AWS_REGION=YOUR_REGION
bash deploy/aws-demo.sh
```

The script determines the authenticated account, creates the private ECR repository if needed, builds and pushes an image tagged with the Git commit, deploys `deploy/aws-demo.yaml`, and prints the Express Mode HTTPS URL. This follows the [ECR push flow](https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html) and [CloudFormation deploy command](https://docs.aws.amazon.com/cli/latest/reference/cloudformation/deploy.html). To use a different stack name, set `AWS_STACK_NAME` before running the script.

Retrieve the two generated keys from the CloudFormation outputs and Secrets Manager. Keep their values out of shell history, logs, and Git. The account operator can read each secret in the AWS console or with [`get-secret-value`](https://docs.aws.amazon.com/cli/latest/reference/secretsmanager/get-secret-value.html). The submission key and reviewer key should go to different demo participants.

Open the printed HTTPS URL. The browser client selects that same origin for API requests automatically. Submit a fictional ticket, load its draft with the reviewer key, approve an edited reply, and refresh saved responses. Rejection should leave responses unchanged. A repeat decision should return HTTP 409.

## Local testing

The existing README local workflow is unchanged. Its default `TICKET_REVIEW_BACKEND=sqlite` keeps the SQLite review store. The hosted service explicitly sets `TICKET_REVIEW_BACKEND=dynamodb` and the two table names. Run the full test suite with:

```bash
python -m unittest discover -s tests -v
```

## Limits and rollback

The ECS Express Mode service is public and uses shared demo keys, not named accounts. The models and sample tickets are synthetic. Use only fictional data. The service runs `TICKET_MODE=demo`; the repository's production-mode guard remains in force. No live AWS deployment or image build has been verified in this development environment.

For a faulty update, deploy a known-good commit from this branch; the deployment script tags each image with its commit. To stop the demo and charges, delete the CloudFormation stack and then the ECR repository and images. Stack deletion also deletes the demo DynamoDB tables and secrets, so export anything you intend to retain first.
