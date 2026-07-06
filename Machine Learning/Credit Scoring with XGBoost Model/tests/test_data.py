
def test_data_load(raw_data):
    assert raw_data is not None
    assert raw_data.shape[0] > 0
