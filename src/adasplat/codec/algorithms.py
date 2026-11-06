"""Preserved V6/V2 geometry, container and RAHT helper algorithms."""
from __future__ import annotations
import hashlib,importlib.util,json,lzma,math,struct,sys,types,zlib
from pathlib import Path
import numpy as np
import torch
MAGIC=b"ADAGC01\0"
HERE=Path(__file__).resolve().parent
UPSTREAM=HERE/'_upstream'
RAHT_SOURCE=UPSTREAM/'raht_torch.py'
LAPLACE_SOURCE=UPSTREAM/'laplace_codec.py'


def _read_leb128_prefix(data, count):
    result = np.empty(count, dtype=np.int64)
    value = shift = index = 0
    for offset, byte in enumerate(data):
        value |= (byte & 127) << shift
        if byte & 128:
            shift += 7
            if shift > 35:
                raise ValueError('position varint exceeds 35 bits')
        else:
            result[index] = value
            index += 1
            if index == count:
                return result, offset + 1
            value = shift = 0
    raise ValueError('truncated position varint stream')

def _rgb_basis():
    """Fixed orthonormal luminance/chroma basis; no trained parameters."""
    return np.asarray([[1/math.sqrt(3), 1/math.sqrt(3), 1/math.sqrt(3)],
                       [1/math.sqrt(2), 0, -1/math.sqrt(2)],
                       [1/math.sqrt(6), -2/math.sqrt(6), 1/math.sqrt(6)]], dtype=np.float64)

def _unpack_field(header, sections, name, count, bits):
    packing = header.get('field_packing', {}).get(name)
    if packing is None:
        return unpack_unsigned(sections[name], count, bits)
    if packing == 'u16_byte_shuffle':
        payload = sections[name]
        if len(payload) != count * 2:
            raise ValueError(f'{name}: u16 byte-shuffle length mismatch')
        raw = np.frombuffer(payload, dtype=np.uint8).reshape(2, count).T.copy().tobytes()
        words = np.frombuffer(raw, dtype='<u2')
        if np.any(words >= (1 << bits)):
            raise ValueError(f'{name}: unpacked code exceeds {bits} bits')
        return words
    raise ValueError(f'{name}: unsupported field packing {packing}')

def decode(stream_path):
    raht, laplace = load_official()
    h, encoded = read_stream(stream_path)
    if h['schema'] != 'adasplat_mesongs_plus_raht_laplace_v1':
        raise ValueError('wrong codec schema')
    if hashlib.sha256(RAHT_SOURCE.read_bytes()).hexdigest() != h['raht_sha256']:
        raise ValueError('upstream RAHT implementation changed')
    if hashlib.sha256(LAPLACE_SOURCE.read_bytes()).hexdigest() != h['laplace_sha256']:
        raise ValueError('upstream Laplace implementation changed')
    sec = {}
    for name, data in encoded.items():
        codec = h.get('section_lossless_codecs', {}).get(name, 'mesongs_plus_laplace' if name == 'rgb_raht_laplace' else 'zlib9')
        if codec == 'mesongs_plus_laplace':
            sec[name] = data
        elif codec == 'zlib9':
            sec[name] = zlib.decompress(data)
        elif codec == 'lzma6':
            sec[name] = lzma.decompress(data)
        else:
            raise ValueError('unsupported lossless section codec')
    for name, value in sec.items():
        if len(value) != h['uncompressed_section_bytes'][name]:
            raise ValueError('section length mismatch')
        if hashlib.sha256(value).hexdigest() != h['uncompressed_section_sha256'][name]:
            raise ValueError('section hash mismatch')
    if h.get('position_predictor') == 'delta_u16_row':
        sec['position_f16'] = position_delta_decode(sec['position_f16'])
    elif h.get('position_predictor') is not None:
        raise ValueError('unsupported position predictor')
    if h.get('position_byte_shuffled'):
        if h.get('position_encoding') != 'fixed_delta_zigzag_u32':
            raise ValueError('byte shuffled position requires fixed-position encoding')
        sec['position_fixed_delta_u32'] = position_byte_unshuffle(sec['position_fixed_delta_u32'])
    fields = decode_geometry(h, sec)
    keys, depth = point_keys(fields['xy'], h['width'], h['height'])
    if depth != h['raht_depth']:
        raise ValueError('RAHT point key depth mismatch')
    codes, lengths, channels = laplace.laplace_decode_blocks(sec['rgb_raht_laplace'])
    # Explicit V2 quality-guard extension: original Laplace coder on high16/low8
    # symbols, recombined into the original 24-bit RAHT coefficient indices.
    if h.get('V2_RGB_lsb_bits', 0):
        if h['V2_RGB_lsb_bits'] != 8 or h['rgb_bits'] != 24:
            raise ValueError('unsupported split RAHT precision')
        low, low_lengths, low_channels = laplace.laplace_decode_blocks(sec['rgb_raht_laplace_low8'])
        if low_lengths != lengths or low_channels != channels or np.any(low > 255):
            raise ValueError('invalid RAHT low8 stream')
        codes = codes.astype(np.int32) * 256 + low.astype(np.int32)
    if channels != 3 or sum(lengths) != h['N'] - 1 or len(lengths) != h['rgb_blocks']:
        raise ValueError('invalid RAHT coefficient stream')
    lo = np.frombuffer(sec['rgb_raht_lo_f32'], dtype='<f4').reshape(3, h['rgb_blocks'])
    step = np.frombuffer(sec['rgb_raht_step_f32'], dtype='<f4').reshape(3, h['rgb_blocks'])
    dc = np.frombuffer(sec['rgb_raht_dc_f32'], dtype='<f4')
    coeff = dequantize_ac(codes, lo, step, lengths, dc)
    transformed_rgb = raht.inv_haar3D(keys, coeff, depth)
    transform = h.get('rgb_transform', 'identity')
    if transform in {'orthonormal_lcc', 'pca_raht'}:
        basis = (_rgb_basis() if transform == 'orthonormal_lcc' else
                 np.asarray(h['rgb_basis'], dtype=np.float64))
        if basis.shape != (3, 3) or not np.allclose(basis @ basis.T, np.eye(3), atol=1e-6):
            raise ValueError('invalid RGB basis')
        transformed_rgb = transformed_rgb @ basis
    elif transform != 'identity':
        raise ValueError('unsupported RGB transform')
    fields['rgb'] = transformed_rgb.astype('<f4')
    if not all(np.isfinite(v).all() for v in fields.values()):
        raise ValueError('non-finite decoded parameter')
    return h, fields

def decode_geometry(header, sections):
    h = header
    n, width, height = h['N'], h['width'], h['height']
    if h.get('position_encoding') == 'fixed_delta_zigzag_u32':
        xy = fixed_position_decode(sections['position_fixed_delta_u32'], n, h['position_fixed_factor'])
    elif h.get('position_encoding') in {'fixed_reset_u16_shuffle', 'fixed_reset_leb128_split'}:
        xy = fixed_reset_decode(sections['position_fixed_delta_u32'], n,
                                h['position_fixed_factor'], h['position_encoding'])
    else:
        xy_raw = np.frombuffer(sections['position_f16'], dtype='<f2').astype('<f4').reshape(n, 2)
        xy = np.tanh(xy_raw)
        xy[:, 0] = (xy[:, 0] + 1) * (width - 1) / 2
        xy[:, 1] = (xy[:, 1] + 1) * (height - 1) / 2
    scale_bits = int(h.get('scale_bits', 10))
    rotation_bits = int(h.get('rotation_bits', 8))
    scale = _unpack_field(header, sections, f'scale_u{scale_bits}', n * 2, scale_bits).astype('<f4').reshape(n, 2)
    scale = scale * np.asarray(h['scale_step'], dtype='<f4') + np.asarray(h['scale_lo'], dtype='<f4')
    angle = _unpack_field(header, sections, f'rotation_u{rotation_bits}', n, rotation_bits).astype('<f4')
    angle = angle * np.float32(h['rotation_step']) + np.float32(h['rotation_lo'])
    p = _unpack_field(header, sections, 'p_sidecar', n, h['p_bits']).astype('<f4')
    p = p / ((1 << h['p_bits']) - 1) * np.float32(h['p_max'] - h['p_min']) + np.float32(h['p_min'])
    return dict(xy=xy, scale=scale, angle=angle, p=p)

def dequantize_ac(codes, lo, step, lengths, dc):
    n = codes.shape[1] + 1
    coeff = np.empty((n, 3), dtype=np.float64)
    coeff[0] = dc
    start = 0
    for b, length in enumerate(lengths):
        stop = start + length
        coeff[1 + start:1 + stop] = codes[:, start:stop].T * step[:, b] + lo[:, b]
        start = stop
    return coeff

def fixed_position_decode(data, n, factor):
    zigzag = np.frombuffer(data, dtype='<u4')
    if len(zigzag) != n * 2:
        raise ValueError('fixed position section length mismatch')
    zigzag = zigzag.reshape(n, 2).astype(np.int64)
    delta = (zigzag >> 1) ^ -(zigzag & 1)
    quantized = np.cumsum(delta, axis=0, dtype=np.int64)
    return quantized.astype('<f4') / factor

def fixed_position_encode(xy, factor, width, height):
    if factor <= 0:
        raise ValueError('position factor must be positive')
    if len(xy) == 0 or not np.isfinite(xy).all():
        raise ValueError('invalid source positions')
    if np.any(xy < 0) or np.any(xy[:, 0] > width - 1) or np.any(xy[:, 1] > height - 1):
        raise ValueError('source positions outside image bounds')
    quantized = np.rint(xy * factor).astype(np.int32)
    delta = np.diff(quantized, axis=0, prepend=np.zeros((1, 2), dtype=np.int32))
    zigzag = ((delta << 1) ^ (delta >> 31)).astype('<u4')
    data = zigzag.tobytes()
    restored = fixed_position_decode(data, len(xy), factor)
    if not np.array_equal(restored, quantized.astype('<f4') / factor):
        raise AssertionError('fixed-point position round trip failed')
    return data, restored

def fixed_reset_decode(data, n, factor, encoding):
    if encoding == 'fixed_reset_u16_shuffle':
        if len(data) != n * 4:
            raise ValueError('reset-u16 position section length mismatch')
        raw = np.frombuffer(data, dtype=np.uint8).reshape(2, n * 2).T.copy().tobytes()
        delta = np.frombuffer(raw, dtype='<u2').astype(np.int64).reshape(n, 2)
    elif encoding == 'fixed_reset_leb128_split':
        dx, split = _read_leb128_prefix(data, n)
        dy, consumed = _read_leb128_prefix(data[split:], n)
        if split + consumed != len(data):
            raise ValueError('position varint trailing bytes')
        delta = np.stack((dx, dy), axis=1)
    else:
        raise ValueError('unsupported reset position encoding')
    fixed = np.empty((n, 2), dtype=np.int64)
    fixed[:, 0] = np.cumsum(delta[:, 0], dtype=np.int64)
    if n:
        fixed[0, 1] = delta[0, 1]
    for i in range(1, n):
        fixed[i, 1] = delta[i, 1] + (fixed[i - 1, 1] if delta[i, 0] == 0 else 0)
    return fixed.astype('<f4') / factor

def load_official():
    # The RAHT file imports PlyData for unrelated PLY utilities. It is not
    # called by haar3D/inv_haar3D and is absent from this inference runtime.
    if 'plyfile' not in sys.modules and importlib.util.find_spec('plyfile') is None:
        sys.modules['plyfile'] = types.SimpleNamespace(PlyData=object)
    out = []
    for label, path in [('official_mesongs_plus_raht', RAHT_SOURCE),
                        ('official_mesongs_plus_laplace', LAPLACE_SOURCE)]:
        spec = importlib.util.spec_from_file_location(label, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out.append(mod)
    return out

def point_keys(xy, width, height):
    x = np.rint(xy[:, 0]).clip(0, width - 1).astype(np.int32)
    y = np.rint(xy[:, 1]).clip(0, height - 1).astype(np.int32)
    n = len(x)
    order = np.lexsort((np.arange(n), y, x))
    sx, sy = x[order], y[order]
    first = np.empty(n, dtype=bool)
    first[0] = True
    first[1:] = (sx[1:] != sx[:-1]) | (sy[1:] != sy[:-1])
    group_start = np.maximum.accumulate(np.where(first, np.arange(n), 0))
    z = np.empty(n, dtype=np.int32)
    z[order] = np.arange(n, dtype=np.int32) - group_start
    keys = np.stack((x, y, z), axis=1)
    depth = max(width, height, int(z.max()) + 1).bit_length()
    if depth > 14:
        raise ValueError('RAHT upstream Morton helper supports <=14 bits/axis')
    return keys, depth

def position_byte_unshuffle(data):
    if len(data) % 4:
        raise ValueError('shuffled position payload is not u32-aligned')
    return np.frombuffer(data, dtype=np.uint8).reshape(4, -1).T.copy().tobytes()

def position_delta_decode(encoded):
    delta = np.frombuffer(encoded, dtype='<u2').reshape(-1, 2)
    words = np.cumsum(delta, axis=0, dtype=np.uint32).astype('<u2')
    return words.tobytes()

def quantize_ac(coefficients, bits, blocks):
    ac = coefficients[1:]
    lengths = split_lengths(len(ac), blocks)
    qcodes = np.empty((3, len(ac)), dtype=np.int32)
    lo = np.empty((3, blocks), dtype='<f4')
    step = np.empty((3, blocks), dtype='<f4')
    start = 0
    for b, length in enumerate(lengths):
        stop = start + length
        values = ac[start:stop]
        low = values.min(axis=0).astype('<f4')
        high = values.max(axis=0).astype('<f4')
        lo[:, b] = low
        for c in range(3):
            bit_depth = int(bits[c, b]) if isinstance(bits, np.ndarray) else int(bits)
            levels = (1 << bit_depth) - 1
            spacing = np.float32((high[c] - low[c]) / levels)
            if spacing == 0:
                spacing = np.float32(1)
            step[c, b] = spacing
            qcodes[c, start:stop] = np.rint((values[:, c] - low[c]) / spacing).clip(0, levels).astype(np.int32)
        start = stop
    return qcodes, lo, step, lengths

def read_stream(path: Path):
    with path.open("rb") as file:
        if file.read(len(MAGIC)) != MAGIC:
            raise ValueError("invalid codec magic")
        header = json.loads(file.read(struct.unpack("<I", file.read(4))[0]))
        sections = {entry["name"]: file.read(entry["bytes"]) for entry in header["sections"]}
        if file.read(1):
            raise ValueError("trailing bytes")
    return header, sections

def shuffle(values):
    raw = values.astype('<u2').tobytes()
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, 2).T.copy().tobytes()

def split_lengths(n, blocks):
    q, r = divmod(n, blocks)
    return [q + (i < r) for i in range(blocks)]

def uniform_encode(values: torch.Tensor, bits: int):
    qmax = (1 << bits) - 1
    lo = values.amin(dim=0)
    hi = values.amax(dim=0)
    step = ((hi - lo) / qmax).clamp_min(torch.finfo(values.dtype).eps)
    code = ((values - lo) / step).round().clamp(0, qmax).to(torch.int64)
    return code, lo, step

def unpack_unsigned(payload: bytes, count: int, bits: int) -> np.ndarray:
    raw = np.frombuffer(payload, dtype=np.uint8)
    bit_values = np.unpackbits(raw, bitorder="little")[: count * bits]
    bit_matrix = bit_values.reshape(count, bits).astype(np.uint64)
    weights = (1 << np.arange(bits, dtype=np.uint64))[None, :]
    return (bit_matrix * weights).sum(axis=1).astype(np.uint16)

def write_stream(path: Path, header: dict, sections: list[tuple[str, bytes]]):
    header = dict(header)
    header["sections"] = [{"name": name, "bytes": len(data)} for name, data in sections]
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.write(MAGIC)
        file.write(struct.pack("<I", len(encoded)))
        file.write(encoded)
        for _, data in sections:
            file.write(data)
