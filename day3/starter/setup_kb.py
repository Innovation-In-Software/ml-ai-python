"""
setup_kb.py — Run once before the lab to upload manuals to S3.
After running this, create the Bedrock Knowledge Base in the AWS console
following the steps in Lab 3.
"""
import boto3
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

if len(sys.argv) < 2:
    print("Usage: python setup_kb.py <your-s3-bucket-name>")
    sys.exit(1)

BUCKET = sys.argv[1]
REGION = "us-east-1"

s3 = boto3.client("s3", region_name=REGION)

print(f"Uploading manuals to s3://{BUCKET}/manuals/\n")

for manual_path in Path("data/manuals").glob("*.txt"):
    product_id = manual_path.stem

    s3.upload_file(str(manual_path), BUCKET, f"manuals/{manual_path.name}")
    print(f"  uploaded {manual_path.name}")

    metadata = {"metadataAttributes": {"product_id": product_id}}
    s3.put_object(
        Bucket=BUCKET,
        Key=f"manuals/{manual_path.name}.metadata.json",
        Body=json.dumps(metadata),
        ContentType="application/json"
    )
    print(f"  uploaded {manual_path.name}.metadata.json  (product_id={product_id})")

print(f"\nDone. Now create your Bedrock Knowledge Base in the AWS console.")
print(f"Point it at:  s3://{BUCKET}/manuals/")
print(f"Embedding model: Amazon Titan Embeddings V2")
