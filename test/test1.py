
import os
import boto3
from pathlib import Path
from dotenv import load_dotenv

# Locate .env in NEXORA_AI root folder
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

print("Looking for .env at:", ENV_PATH)
print(".env exists:", ENV_PATH.exists())

# Load environment variables
load_dotenv(dotenv_path=ENV_PATH, override=True)

# Read AWS credentials
access_key = os.getenv("AWS_ACCESS_KEY_ID")
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
region = os.getenv("AWS_REGION", "us-east-1")

# Check without exposing credentials
print("AWS Access Key loaded:", bool(access_key))
print("AWS Secret Key loaded:", bool(secret_key))
print("AWS Region:", region)

if not access_key or not secret_key:
    raise ValueError("AWS credentials missing in .env")

# Create AWS Bedrock client
client = boto3.client(
    "bedrock-runtime",
    region_name=region,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
)

print("Bedrock client created successfully!")

# Test Amazon Nova Lite
response = client.converse(
    modelId=os.getenv(
        "BEDROCK_CHAT_MODEL",
        "amazon.nova-lite-v1:0"
    ),
    messages=[
        {
            "role": "user",
            "content": [
                {"text": "Explain Agentic AI in one sentence."}
            ],
        }
    ],
    inferenceConfig={
        "maxTokens": 150,
        "temperature": 0.3,
    },
)

answer = response["output"]["message"]["content"][0]["text"]

print("\nNova Lite Response:")
print(answer)
