"""Checksum-verified, bounded HTTPS downloads; no automatic license bypass."""
from __future__ import annotations
from pathlib import Path, PurePosixPath
import hashlib
import os
import stat
import urllib.parse
import urllib.request
import zipfile


def download_file(url: str, destination: str | Path, sha256: str,
                  accept_license: bool = False, max_bytes: int = 4 * 1024 ** 3) -> Path:
    if not accept_license:
        raise PermissionError('Dataset download requires explicit license acceptance')
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != 'https' or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError('Only public HTTPS URLs without embedded credentials are accepted')
    if len(sha256) != 64 or any(char not in '0123456789abcdefABCDEF' for char in sha256):
        raise ValueError('An expected SHA-256 digest is mandatory')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing_digest = hashlib.sha256()
        with destination.open('rb') as existing_stream:
            for chunk in iter(lambda: existing_stream.read(1024 * 1024), b''):
                existing_digest.update(chunk)
        existing = existing_digest.hexdigest()
        if existing.lower() == sha256.lower():
            return destination
        raise FileExistsError('Destination exists with a different checksum; it will not be overwritten')
    temporary = destination.with_suffix(destination.suffix + '.part')
    digest = hashlib.sha256()
    total = 0
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'rl-autodrive-agent/2.0'})
        with urllib.request.urlopen(request, timeout=30) as response, temporary.open('wb') as stream:
            if urllib.parse.urlparse(response.geturl()).scheme != 'https':
                raise ValueError('Redirect to non-HTTPS transport rejected')
            length = response.headers.get('Content-Length')
            if length and int(length) > max_bytes:
                raise ValueError('Dataset exceeds configured byte limit')
            for chunk in iter(lambda: response.read(1024 * 1024), b''):
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError('Download exceeded configured byte limit')
                digest.update(chunk)
                stream.write(chunk)
        if digest.hexdigest().lower() != sha256.lower():
            raise ValueError('Dataset SHA-256 mismatch')
        temporary.replace(destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def safe_extract_zip(archive: str | Path, target: str | Path, max_bytes: int = 5 * 1024 ** 3,
                     exclude_git: bool = False) -> None:
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if sum(member.file_size for member in members) > max_bytes:
            raise ValueError('Uncompressed archive exceeds byte limit')
        for member in members:
            path = PurePosixPath(member.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in member.filename or ':' in member.filename:
                raise ValueError('Unsafe archive path')
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError('Archive symlinks are not supported')
            resolved = (target / str(path)).resolve()
            if target.resolve() not in [resolved, *resolved.parents]:
                raise ValueError('Archive path escapes target')
        for member in members:
            if exclude_git and any(part in {'.git', '__pycache__'} for part in PurePosixPath(member.filename).parts):
                continue
            bundle.extract(member, target)
