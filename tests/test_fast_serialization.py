import io

import numpy as np
import pytest

from tiptop.serialization import savez_fast


def test_standard_npz_reader_recovers_every_value_dtype_shape_and_order(tmp_path):
    values = dict(rgb=np.arange(180, dtype=np.uint8).reshape(6, 10, 3),
                  depth=np.array([[np.nan, np.inf, -0., 1.25]], dtype=np.float32),
                  mask=np.zeros((6, 10), bool), intrinsics=np.eye(3),
                  class_id='oriox_bubble_hand_wash_pump', observation_id='새 영상',
                  empty=np.empty((0, 3), '>i2'), strided=np.arange(100)[::3],
                  fortran=np.asfortranarray(np.arange(12).reshape(3, 4)))
    for destination in (io.BytesIO(), tmp_path/'packet'):
        savez_fast(destination, **values)
        source = io.BytesIO(destination.getvalue()) if isinstance(destination, io.BytesIO) else str(destination)+'.npz'
        with np.load(source, allow_pickle=False) as actual:
            assert actual.files == list(values)
            for key, value in values.items():
                expected = np.asanyarray(value)
                assert actual[key].dtype == expected.dtype and actual[key].shape == expected.shape
                np.testing.assert_array_equal(actual[key], expected)


def test_positional_mask_packets_keep_the_existing_arr_zero_name():
    packet = io.BytesIO()
    savez_fast(packet, np.ones((2, 3), bool))
    with np.load(io.BytesIO(packet.getvalue()), allow_pickle=False) as arrays:
        assert arrays.files == ['arr_0'] and arrays['arr_0'].all()
    with pytest.raises(ValueError):
        savez_fast(io.BytesIO(), np.arange(2), arr_0=np.arange(2))
