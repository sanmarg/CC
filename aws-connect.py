import boto3

# # Initialize S3 client
s3 = boto3.client("s3")

# # List S3 buckets
response = s3.list_buckets()
print("S3 Buckets:", [bucket["Name"] for bucket in response["Buckets"]])

