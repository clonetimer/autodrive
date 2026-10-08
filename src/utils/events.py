"""Small TensorBoard-compatible scalar event writer (protobuf + TFRecord).

Implements public Event/Summary wire formats without requiring TensorFlow or
TensorBoard. TensorBoard UI is optional. File header and each record CRC are
unit tested; full TensorBoard UI is not installed in the offline sandbox.
"""
from pathlib import Path
import os
import math
import socket
import struct
import time


def _varint(value: int) -> bytes:
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def _message(field: int, value: bytes) -> bytes:
    return _varint(field << 3 | 2) + _varint(len(value)) + value


def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    return crc ^ 0xFFFFFFFF


def _masked_crc(data: bytes) -> bytes:
    crc = crc32c(data)
    return struct.pack('<I', (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF)


class ScalarWriter:
    def __init__(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f'events.out.tfevents.{int(time.time())}.{socket.gethostname()}.{os.getpid()}.{time.time_ns()}'
        self.stream = self.path.open('wb')
        self._record(b'\x09' + struct.pack('<d', time.time()) + _message(3, b'brain.Event:2'))

    def _record(self, data):
        length = struct.pack('<Q', len(data))
        self.stream.write(length + _masked_crc(length) + data + _masked_crc(data))
        self.stream.flush()

    def add_scalar(self, tag: str, value: float, step: int):
        if step < 0 or not math.isfinite(float(value)):
            raise ValueError('Scalar step must be non-negative and value finite')
        summary_value = _message(1, tag.encode()) + b'\x15' + struct.pack('<f', float(value))
        summary = _message(1, summary_value)
        event = b'\x09' + struct.pack('<d', time.time()) + b'\x10' + _varint(step) + _message(5, summary)
        self._record(event)

    def close(self):
        self.stream.close()
