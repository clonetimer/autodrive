"""Parse emitted bytes through protobuf, independent of the writer's encoder."""
import struct
import pytest
from src.utils.events import ScalarWriter


def test_public_tensorboard_scalar_schema(tmp_path):
    descriptor_pb2 = pytest.importorskip('google.protobuf.descriptor_pb2')
    from google.protobuf import descriptor_pool, message_factory
    schema = descriptor_pb2.FileDescriptorProto(name='subset_tensorboard_event.proto', package='tensorflow', syntax='proto2')
    summary = schema.message_type.add(name='Summary')
    value = summary.nested_type.add(name='Value')
    value.field.add(name='tag', number=1, type=9, label=1)
    value.field.add(name='simple_value', number=2, type=2, label=1)
    summary.field.add(name='value', number=1, type=11, type_name='.tensorflow.Summary.Value', label=3)
    event = schema.message_type.add(name='Event')
    event.field.add(name='wall_time', number=1, type=1, label=1)
    event.field.add(name='step', number=2, type=3, label=1)
    event.field.add(name='file_version', number=3, type=9, label=1)
    event.field.add(name='summary', number=5, type=11, type_name='.tensorflow.Summary', label=1)
    pool = descriptor_pool.DescriptorPool()
    pool.Add(schema)
    Event = message_factory.GetMessageClass(pool.FindMessageTypeByName('tensorflow.Event'))
    writer = ScalarWriter(tmp_path)
    writer.add_scalar('loss/policy', -0.125, 2048)
    writer.close()
    data = writer.path.read_bytes()
    messages = []
    offset = 0
    while offset < len(data):
        length = struct.unpack('<Q', data[offset:offset + 8])[0]
        messages.append(Event.FromString(data[offset + 12:offset + 12 + length]))
        offset += 16 + length
    assert messages[0].file_version == 'brain.Event:2'
    assert messages[1].step == 2048
    assert messages[1].summary.value[0].tag == 'loss/policy'
    assert messages[1].summary.value[0].simple_value == -0.125
    assert messages[1].wall_time > 0
