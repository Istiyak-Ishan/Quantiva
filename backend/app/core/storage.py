import boto3
import uuid
from botocore.exceptions import ClientError
from botocore.config import Config
from app.config import settings

# Since S3 upload via boto3 is synchronous by default, we wrap it.
# Note: For fully async API we could use aioboto3, but for simple file uploads
# passing it to a Celery worker or using threadpools is also fine.
# We'll use standard boto3 since S3 operations here are fast or happen in Celery.

s3_client = boto3.client(
    "s3",
    endpoint_url=settings.S3_ENDPOINT_URL,
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_DEFAULT_REGION,
    config=Config(connect_timeout=2, read_timeout=3, retries={"max_attempts": 1}),
)

BUCKET_NAME = "quantiva-documents"

def ensure_bucket_exists():
    """Ensure the target S3 bucket exists, if not, create it."""
    try:
        s3_client.head_bucket(Bucket=BUCKET_NAME)
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            s3_client.create_bucket(Bucket=BUCKET_NAME)
        else:
            raise

def upload_file(file_bytes: bytes, filename: str, content_type: str) -> str:
    """
    Uploads a file to S3/s3mock and returns the generated object key.
    """
    ensure_bucket_exists()
    
    # Create a unique object key to prevent filename collisions
    ext = filename.split('.')[-1] if '.' in filename else ''
    object_key = f"{uuid.uuid4()}.{ext}"
    
    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=object_key,
        Body=file_bytes,
        ContentType=content_type
    )
    return object_key

def get_file(object_key: str) -> bytes:
    """Retrieves a file from S3."""
    response = s3_client.get_object(Bucket=BUCKET_NAME, Key=object_key)
    return response['Body'].read()

def delete_file(object_key: str):
    """Deletes a file from S3."""
    s3_client.delete_object(Bucket=BUCKET_NAME, Key=object_key)


def presigned_url(object_key: str, expires_in: int = 3600) -> str:
    """Sign a fresh browser-reachable GET URL for an existing object.

    A new signer is created because the public/browser endpoint can differ
    from the internal one used for uploads. The caller never persists this URL.
    """
    public_endpoint = settings.S3_PUBLIC_ENDPOINT_URL or settings.S3_ENDPOINT_URL
    signer = boto3.client(
        "s3",
        endpoint_url=public_endpoint,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_DEFAULT_REGION,
        config=Config(connect_timeout=2, read_timeout=3, retries={"max_attempts": 1}),
    )
    try:
        return signer.generate_presigned_url(
            "get_object",
            Params={"Bucket": BUCKET_NAME, "Key": object_key},
            ExpiresIn=expires_in,
        )
    finally:
        signer.close()

class StorageService:
    @staticmethod
    def upload_file(file_bytes: bytes, filename: str, content_type: str) -> str:
        return upload_file(file_bytes, filename, content_type)

    @staticmethod
    def get_file(object_key: str) -> bytes:
        return get_file(object_key)

    @staticmethod
    def delete_file(object_key: str):
        return delete_file(object_key)
