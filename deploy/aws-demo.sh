#!/usr/bin/env bash
set -euo pipefail

region="${AWS_REGION:?Set AWS_REGION to the chosen AWS Region}"
stack="${AWS_STACK_NAME:-enterprise-llm-transformer-demo}"
repository="enterprise-llm-transformer-demo"

if [[ -n "$(git status --porcelain)" ]]; then
  echo 'Commit the reviewed changes before deploying an image.' >&2
  exit 1
fi

account="$(aws sts get-caller-identity --query Account --output text --region "$region")"
if ! aws ecr describe-repositories --repository-names "$repository" --region "$region" >/dev/null 2>&1; then
  aws ecr create-repository --repository-name "$repository" --region "$region" >/dev/null
fi

image_tag="$(git rev-parse --short=12 HEAD)"
image_uri="$account.dkr.ecr.$region.amazonaws.com/$repository:$image_tag"
aws ecr get-login-password --region "$region" | docker login --username AWS --password-stdin \
  "$account.dkr.ecr.$region.amazonaws.com"
docker build --platform linux/amd64 -t "$image_uri" .
docker push "$image_uri"

aws cloudformation deploy --stack-name "$stack" --template-file deploy/aws-demo.yaml \
  --parameter-overrides "ImageUri=$image_uri" --capabilities CAPABILITY_IAM --region "$region"

endpoint="$(aws cloudformation describe-stacks --stack-name "$stack" --region "$region" \
  --query 'Stacks[0].Outputs[?OutputKey==`AppUrl`].OutputValue|[0]' --output text)"
case "$endpoint" in
  https://*) printf '%s\n' "$endpoint" ;;
  *) printf 'https://%s\n' "$endpoint" ;;
esac
