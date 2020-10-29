#!/home/mark/phd/venv/bin/python
# coding: utf-8
"""Function to persist experiment results."""
from typing import Dict
from typing import Any
from typing import Optional

import torch as th

from hashlib import sha256

from base64 import b64encode
from os import mkdir
from os.path import isdir
from os.path import basename
from os.path import getctime
from json import dumps
from glob import glob


Params = Dict[str, Any]
HASH_LEN: int = 10


def save_tensor(t: th.Tensor, path: str, params: Optional[Params]) -> None:
    path = get_path_with_hash(path, params)
    file_name: int = 0

    if isdir(path):
        last_file = get_last_file(path)
        file_name = int(basename(last_file)) + 1
    else:
        mkdir(path)

    th.save(t, f'{path}/{file_name}')


def load_tensor(path: str, params: Optional[Params]) -> th.Tensor:
    """Load all the tensors into a stack."""
    path = get_path_with_hash(path, params)
    files = glob(f'{path}/*')
    return th.stack([th.tensor(th.load(f)) for f in files])

def load_last_tensor(path: str, params: Optional[Params]) -> th.Tensor:
    """Load only the last saved tensor."""
    path = get_path_with_hash(path, params)
    f = get_last_file(path)
    return th.load(f)


def get_last_file(path: str) -> str:
    return max(glob(f'{path}/*'), key=getctime)


def get_path_with_hash(path: str, params: Optional[Params]) -> str:
    if params is not None:
        h = hash_p(params)
    return f'{path}_{h}' if params else path


def hash_p(params: Params) -> str:
    s: str = dumps(params)
    h: str = sha256(s.encode()).hexdigest()
    b64: bytes = b64encode(h.encode())
    return b64.decode('utf-8')[:HASH_LEN]
