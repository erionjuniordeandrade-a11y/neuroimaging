import hashlib

from neuro_core.hashing import sha256_bytes, sha256_file


def test_file_hash_matches_hashlib_across_blocks(tmp_path):
    data = bytes(range(256)) * 9000  # larger than one 1 MiB block
    path = tmp_path / "blob"
    path.write_bytes(data)
    assert sha256_file(path) == hashlib.sha256(data).hexdigest() == sha256_bytes(data)
