1. 每个文件一个metadata_cache
storage:
petrel_conf: ~/petreloss.conf
parquet_cache_size: 8
metadata_cache: data/petrel_cache/meta/{remote}
path_index_cache: data/petrel_cache/path_index/{remote}

# TODO single dataset
cache_templates = { 
    "local_meta_dir": storage.get("metadata_cache"),
    "path_index_dir": storage.get("path_index_cache"),
}


2. norm stats 实现没有使用 stats_group 分组

3. sampling:
  state_frames: 1
  action_horizon: 16
  stride: 1
  allow_end_padding: false
  training_block_size: 16
  training_frame_stride: 1   training_block_size的设置以及dataloader的shuffle

4.图像分配并不是按照字符串进行分配，而是对应数据集中video_keys进行顺序分配

5.dataloader的 epoch没有保证能报所有的数据都过一遍 

############################### confused

1.