"""Lightweight provenance checks for prepared ALOHA data (no model imports here)."""
import hashlib
import json
from pathlib import Path

ENCODING_VERSION = 3


def file_identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def model_identity(root):
    root = Path(root)
    return {component: [file_identity(p) for p in sorted((root/component).iterdir()) if p.is_file()]
            for component in ('vae', 'text_encoder', 'tokenizer')}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def job_identity(job, model):
    return digest({'job': job, 'source': file_identity(job['video']),
                   'model': model, 'encoding_version': ENCODING_VERSION})


def checked_cache(job, model):
    output = Path(job['output'])
    sidecar = output.with_suffix('.provenance.json')
    if not output.is_file() or not sidecar.is_file():
        # A killed writer may leave a complete tensor without its commit record.
        # Recompute it; legacy output directories lack encoding_contract.json and
        # are rejected by the caller before reaching this resumable path.
        return False
    saved = json.loads(sidecar.read_text())
    if saved != {'identity': job_identity(job, model), 'size': output.stat().st_size}:
        raise ValueError(f'Stale or modified encoding cache: {output}; rebuild in a new output directory')
    return True
