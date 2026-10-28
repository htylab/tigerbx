import hashlib
import os
import re
from os.path import dirname, join

import pytest

from tigerbx import lib_tool
from tigerbx.core import models

GOOD = b'release model'
BAD = b'old model'


def test_every_default_model_has_a_sha256():
    names = [v for k, v in vars(models).items() if k.isupper() and isinstance(v, str)]
    assert names and all(n in models.SHA256 for n in names)
    assert all(re.fullmatch('[0-9a-f]{64}', h) for h in models.SHA256.values())


def test_executable_models_are_release_models():
    text = open(join(dirname(dirname(__file__)), '.github', 'workflows', 'exe.yml')).read()
    embedded = re.search(r'MODELS: "([^"]+)"', text).group(1).split()
    assert embedded and all(m + '.onnx' in models.SHA256 for m in embedded)


def test_python_and_executable_versions_agree():
    import tigerbx
    root = join(dirname(dirname(__file__)), 'rust')
    cargo = re.search(r'^version = "([^"]+)"', open(join(root, 'Cargo.toml')).read(), re.M).group(1)
    lock = re.search(r'name = "tigerbx"\nversion = "([^"]+)"', open(join(root, 'Cargo.lock')).read()).group(1)
    assert tigerbx.__version__ == cargo == lock


def test_check_reports_the_model_cache(tmp_path, monkeypatch):
    import tigerbx
    monkeypatch.setenv(lib_tool.MODEL_DIR_ENV, str(tmp_path))
    open(join(str(tmp_path), models.TBX), 'wb').write(BAD)
    r = tigerbx.check(verbose=False)
    assert r['tigerbx'] == tigerbx.__version__ and r['ort_threads'] >= 1
    assert r['models'][models.TBX].startswith('sha256 mismatch')
    assert r['models'][models.INSPECT] == 'not downloaded'


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    monkeypatch.setenv(lib_tool.MODEL_DIR_ENV, str(tmp_path))
    monkeypatch.setitem(models.SHA256, 'fake_model.onnx', hashlib.sha256(GOOD).hexdigest())
    served = {'data': GOOD}

    def download(url, file_name, timeout_s=None):
        with open(file_name, 'wb') as f:
            f.write(served['data'])
    monkeypatch.setattr(lib_tool, 'download', download)
    return tmp_path, served


def test_verify_model(tmp_path, fake_hub):
    f = join(str(tmp_path), 'fake_model.onnx')
    open(f, 'wb').write(BAD)
    assert not lib_tool.verify_model(f)
    open(f, 'wb').write(GOOD)
    assert lib_tool.verify_model(f)
    other = join(str(tmp_path), 'my_own.onnx')
    open(other, 'wb').write(BAD)
    assert lib_tool.verify_model(other)        # not a release name: not checked


def test_stale_cached_model_is_set_aside_and_downloaded(fake_hub):
    d, _ = fake_hub
    f = join(str(d), 'fake_model.onnx')
    open(f, 'wb').write(BAD)
    with pytest.warns(RuntimeWarning, match='does not match'):
        assert lib_tool.get_model('fake_model') == f
    assert open(f, 'rb').read() == GOOD
    assert open(f + '.bad', 'rb').read() == BAD


def test_download_with_a_wrong_sha256_fails(fake_hub):
    d, served = fake_hub
    served['data'] = BAD
    with pytest.raises(ValueError, match='sha256 mismatch'):
        lib_tool.get_model('fake_model')
    assert not os.path.exists(join(str(d), 'fake_model.onnx'))
