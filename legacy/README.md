# Legacy Upstream MoP-GRU Implementation

This directory preserves the original word-level dynamic sequence modeling implementation inherited from the upstream research repository by **Barathi Subramanian et al. (2022)** (*Scientific Reports*, DOI: `10.1038/s41598-022-15998-7`):

- `mopgru_cell.py`: Custom recurrent Gated Recurrent Unit (MoP-GRU) cell implementation.
- `collect_keypoints.py`: Holistic landmark acquisition script for 30-frame sequence recordings.
- `preprocess_keypoints.py`: Landmark temporal sequence preprocessing and normalization.
- `realtime_inference.py`: Prototype webcam real-time inference loop for isolated word signs.
- `train_word_model.py`: Training harness for 30-frame sequence models.

**Note:** Heavy binary model checkpoints (`mopgru_model.h5`, `actionGRU.h5`) are archived in `reserve/old_models/` to avoid repository bloat. Our team's active work focuses on the real-time fingerspelling alphabet recognition, bimanual 144D feature extraction, multi-modal decision fusion, multilingual translation, and multi-tier speech output located in `src/`.
