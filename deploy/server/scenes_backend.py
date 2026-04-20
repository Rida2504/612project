"""
scenes_backend.py — pluggable scene storage (local filesystem OR S3-compatible).

Env vars select the backend:
  SCENES_BACKEND=local|s3
  local:  SCENES_DIR=/scenes
  s3:     S3_BUCKET, S3_ENDPOINT_URL (e.g. https://storage.eu-north1.nebius.cloud),
          S3_PREFIX (optional, default "scenes/"),
          AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (standard boto3 auth)
          S3_PUBLIC_BASE_URL (optional, for unsigned viewer access)
          S3_PRESIGN_EXPIRES_S (default 86400)

The API layer calls `backend.put(local_path, scene_name)` after training and
`backend.list()` / `backend.url_for(scene_name)` for viewer discovery.
"""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

log = logging.getLogger("twvr.scenes")


class ScenesBackend:
    def put(self, local_path: str, scene_name: str) -> str: raise NotImplementedError
    def list(self) -> list[str]: raise NotImplementedError
    def url_for(self, scene_name: str) -> str: raise NotImplementedError
    def exists(self, scene_name: str) -> bool: raise NotImplementedError


@dataclass
class LocalScenesBackend(ScenesBackend):
    root: Path

    def __post_init__(self):
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, local_path: str, scene_name: str) -> str:
        dst = self.root / scene_name
        if Path(local_path).resolve() != dst.resolve():
            shutil.move(local_path, dst)
        return str(dst)

    def list(self) -> list[str]:
        return sorted(p.name for p in self.root.glob("*.ply"))

    def url_for(self, scene_name: str) -> str:
        return f"/splats/{scene_name}"

    def exists(self, scene_name: str) -> bool:
        return (self.root / scene_name).exists()


class S3ScenesBackend(ScenesBackend):
    """S3-compatible backend (works with Nebius Object Storage, AWS S3, MinIO, etc.)."""

    def __init__(self, bucket: str, endpoint_url: Optional[str] = None,
                 prefix: str = "scenes/", public_base_url: Optional[str] = None,
                 presign_expires_s: int = 86_400):
        import boto3
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/"
        self.public_base_url = public_base_url.rstrip("/") if public_base_url else None
        self.presign_expires_s = presign_expires_s
        self.s3 = boto3.client("s3", endpoint_url=endpoint_url)
        # Fail fast if bucket is wrong / creds are wrong.
        self.s3.head_bucket(Bucket=bucket)
        log.info("S3 backend ready: bucket=%s endpoint=%s prefix=%s", bucket, endpoint_url, self.prefix)

    def _key(self, scene_name: str) -> str:
        return f"{self.prefix}{scene_name}"

    def put(self, local_path: str, scene_name: str) -> str:
        key = self._key(scene_name)
        self.s3.upload_file(local_path, self.bucket, key,
                            ExtraArgs={"ContentType": "model/ply"})
        Path(local_path).unlink(missing_ok=True)
        log.info("S3 put %s/%s (from %s)", self.bucket, key, local_path)
        return f"s3://{self.bucket}/{key}"

    def list(self) -> list[str]:
        pager = self.s3.get_paginator("list_objects_v2")
        names = []
        for page in pager.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                name = obj["Key"][len(self.prefix):]
                if name.endswith(".ply"):
                    names.append(name)
        return sorted(names)

    def url_for(self, scene_name: str) -> str:
        if self.public_base_url:
            return f"{self.public_base_url}/{self._key(scene_name)}"
        return self.s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": self._key(scene_name)},
            ExpiresIn=self.presign_expires_s,
        )

    def exists(self, scene_name: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self._key(scene_name))
            return True
        except Exception:
            return False


def from_env() -> ScenesBackend:
    """Pick and construct backend from env vars."""
    kind = os.environ.get("SCENES_BACKEND", "local").lower()
    if kind == "s3":
        return S3ScenesBackend(
            bucket=os.environ["S3_BUCKET"],
            endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
            prefix=os.environ.get("S3_PREFIX", "scenes/"),
            public_base_url=os.environ.get("S3_PUBLIC_BASE_URL"),
            presign_expires_s=int(os.environ.get("S3_PRESIGN_EXPIRES_S", "86400")),
        )
    return LocalScenesBackend(root=os.environ.get("SCENES_DIR", "/scenes"))
