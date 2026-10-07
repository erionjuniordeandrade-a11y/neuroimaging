import numpy as np
import pytest

from neuro_core.tck import decode_tck, encode_tck, read_tck, write_tck


def _lines():
    return [np.array([[0, 0, 0], [1, 2, 3]], np.float32), np.array([[5, 5, 5], [6, 6, 6], [7, 7, 7]], np.float32)]


def test_round_trip_and_minimal_header(tmp_path):
    path = tmp_path / "t.tck"
    write_tck(path, _lines())
    data = path.read_bytes()
    header = data[:data.index(b"END\n")].decode()
    assert header.splitlines()[1:] == ["datatype: Float32LE", "count: 2", f"file: . {len(header) + 4}"]
    out = read_tck(path)
    assert len(out) == 2 and all(np.array_equal(a, b) for a, b in zip(out, _lines()))


def test_write_never_overwrites(tmp_path):
    path = tmp_path / "t.tck"
    path.write_bytes(b"x")
    with pytest.raises(FileExistsError):
        write_tck(path, _lines())


def test_rejects_non_tck_and_non_finite():
    with pytest.raises(ValueError):
        decode_tck(b"not a tck")
    with pytest.raises(ValueError):
        encode_tck([np.array([[0, 0, np.nan], [1, 1, 1]])])


def test_big_endian_decodes():
    body = np.array([[1, 2, 3], [np.nan] * 3, [np.inf] * 3], ">f4").tobytes()
    header = b"mrtrix tracks\ndatatype: Float32BE\nfile: . 52\nEND\n"
    header = header.ljust(52, b"\n")
    assert np.array_equal(decode_tck(header + body)[0], [[1, 2, 3]])
