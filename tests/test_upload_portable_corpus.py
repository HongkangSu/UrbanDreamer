import hashlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools.upload_portable_corpus import file_hash, verify_remote, upload_one


def test_remote_git_hash_and_size(tmp_path):
    path = tmp_path / 'sample'
    path.write_bytes(b'cache')
    record = {'bytes': 5, 'sha256': file_hash(path)}
    api = Mock()
    api.get_paths_info.return_value = [SimpleNamespace(size=5, lfs=None, blob_id=file_hash(path, git=True))]
    verify_remote(api, 'owner/repo', 'sample', path, record)
    api.get_paths_info.return_value[0].blob_id = 'wrong'
    with pytest.raises(ValueError, match='remote_git_hash_mismatch'):
        verify_remote(api, 'owner/repo', 'sample', path, record)


def test_remote_lfs_verification(tmp_path):
    path = tmp_path / 'sample'
    path.write_bytes(b'cache')
    record = {'bytes': 5, 'sha256': file_hash(path)}
    api = Mock()
    api.get_paths_info.return_value = [SimpleNamespace(size=5, lfs={'sha256': record['sha256']})]
    verify_remote(api, 'owner/repo', 'sample', path, record)
    api.get_paths_info.return_value[0].lfs['sha256'] = 'wrong'
    with pytest.raises(ValueError, match='remote_lfs_hash_mismatch'):
        verify_remote(api, 'owner/repo', 'sample', path, record)


def test_modified_local_file_rejected_before_upload(tmp_path):
    path = tmp_path / 'sample'
    path.write_bytes(b'wrong')
    with pytest.raises(ValueError, match='local_artifact_hash_mismatch'):
        upload_one(tmp_path, tmp_path, 'owner/repo', dict(path='sample', bytes=5,
            sha256=hashlib.sha256(b'cache').hexdigest()))


def test_symlink_escape_rejected(tmp_path):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (tmp_path / 'outside').write_bytes(b'cache')
    (bundle / 'sample').symlink_to(tmp_path / 'outside')
    with pytest.raises(ValueError, match='unsafe_upload_path'):
        upload_one(bundle, tmp_path, 'owner/repo', dict(path='sample', bytes=5, sha256='unused'))
