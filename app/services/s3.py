import uuid
import boto3
from fastapi import UploadFile, HTTPException
from typing import Optional, Union
import io

from app.config import settings


class S3Service:
    def __init__(
            self,
            aws_access_key_id: str = settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key: str = settings.AWS_SECRET_ACCESS_KEY,
            region_name: str = settings.AWS_REGION,
            bucket_name: str = settings.S3_BUCKET_PUBLIC
    ):

        if not aws_access_key_id or not aws_secret_access_key:
            self.s3_client = boto3.client(
                "s3",
                region_name=region_name
            )
        else:
            self.s3_client = boto3.client(
                "s3",
                region_name=region_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )

        self.bucket_name = bucket_name
        self.region_name = region_name

    async def upload_file(self, file: UploadFile,  prefix: str = "images/") -> str:
        # Determine extension
        filename = file.filename if file.filename else "file.jpg"
        if "." in filename:
            extension = filename.split(".")[-1]
        else:
            extension = "jpg"
            
        unique_filename = f"{uuid.uuid4()}.{extension}"
        object_key = f"{prefix.rstrip('/')}/{unique_filename}"

        file_content = await file.read()
        # Reset cursor if needed, though usually read() consumes it one-off for UploadFile
        # If it's a spooled file, we might need seek(0) if it was read before? 
        # But we assume it's fresh or passed correctly.
        
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=file_content,
                ContentType=file.content_type if file.content_type else "image/jpeg"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to upload to S3: {str(e)}")

        return f"https://{self.bucket_name}.s3.{self.region_name}.amazonaws.com/{object_key}"


# Instantiate services for Public and Private buckets
s3_service_public = S3Service(bucket_name=settings.S3_BUCKET_PUBLIC)
s3_service_private = S3Service(bucket_name=settings.S3_BUCKET_PRIVATE)

__all__ = ["S3Service", "s3_service_public", "s3_service_private"]
