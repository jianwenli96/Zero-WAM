import json
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from wan_va.dataset.lerobot_latent_dataset import load_action_parquets


def test_new_hf_list_metadata_preserves_numeric_action_values(tmp_path, monkeypatch):
    monkeypatch.setattr('datasets.config.HF_DATASETS_CACHE', tmp_path / 'cache')
    files = []
    for index in range(2):
        table = pa.table({'action': pa.array([[index + 0.25, 2.0]], type=pa.list_(pa.float32())),
                          'state': pa.array([[3.0, index + 4.0]], type=pa.list_(pa.float32(), 2)),
                          'video': ['unused.mp4']})
        # Newer HF metadata is incompatible with the pinned datasets version.
        metadata = {'info': {'features': {
            'action': {'_type': 'List', 'feature': {'_type': 'Value', 'dtype': 'float32'}},
            'state': {'_type': 'List', 'length': 2, 'feature': {'_type': 'Value', 'dtype': 'float32'}},
            'video': {'_type': 'SomeFutureVideoType'}}}}
        table = table.replace_schema_metadata({b'huggingface': json.dumps(metadata).encode()})
        path = tmp_path / f'{index}.parquet'
        pq.write_table(table, path)
        files.append(path)
    dataset = load_action_parquets(files, ['action', 'state'])
    assert dataset.column_names == ['action', 'state']
    batch = dataset.with_format('torch')[:]
    torch.testing.assert_close(batch['action'], torch.tensor([[0.25, 2.0], [1.25, 2.0]]))
    torch.testing.assert_close(batch['state'], torch.tensor([[3.0, 4.0], [3.0, 5.0]]))
    # Reading the local cached Arrow files must also be independent of List metadata.
    from datasets import Dataset
    restored = Dataset.from_file(dataset.cache_files[0]['filename'])
    assert restored.to_dict() == dataset.to_dict()
