"""
setup_dynamo.py - Run once to create DynamoDB tables and seed product data.
"""
import boto3
from dotenv import load_dotenv

load_dotenv()

dynamodb = boto3.resource("dynamodb", region_name="us-east-1")


def create_table(name, key):
    try:
        table = dynamodb.create_table(
            TableName=name,
            KeySchema=[{"AttributeName": key, "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": key, "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST"
        )
        table.wait_until_exists()
        print(f"  {name} created.")
        return table
    except dynamodb.meta.client.exceptions.ResourceInExistsException:
        print(f"  {name} already exists, skipping.")
        return dynamodb.Table(name)


print("Creating tables...")
products_table = create_table("quantumnet-products", "product_id")
create_table("quantumnet-tickets", "ticket_id")

print("Seeding product data...")
products = [
    {"product_id": "qr5000", "name": "QuantumRouter QR-5000", "stock": 47, "price": "1299.99", "currency": "USD"},
    {"product_id": "qr500",  "name": "QuantumRouter QR-500",  "stock": 3,  "price": "149.99",  "currency": "USD"},
    {"product_id": "qs24",   "name": "QuantumSwitch QS-24",   "stock": 12, "price": "499.99",  "currency": "USD"},
]
for p in products:
    products_table.put_item(Item=p)
    print(f"  {p['product_id']}: stock={p['stock']}, price=${p['price']}")

print("\nDone. Run python agent.py to test.")
