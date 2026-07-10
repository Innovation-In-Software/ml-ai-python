import boto3
import os
from dotenv import load_dotenv

load_dotenv()

client = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

response = client.retrieve(
    knowledgeBaseId=os.getenv("KB_ID", "YOUR_KB_ID_HERE"),
    retrievalQuery={"text": "How do I configure a VLAN on the QR-5000?"},
    retrievalConfiguration={
        "managedSearchConfiguration": {"numberOfResults": 4}
    }
)

for r in response["retrievalResults"]:
    print(f"[score {r['score']:.3f}] {r['location']['s3Location']['uri']}")
    print(f"  {r['content']['text'][:150]}...")
    print()
