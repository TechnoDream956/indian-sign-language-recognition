# Legacy Upstream MoP-GRU Implementation

This directory preserves the original dynamic word sequence modeling implementation inherited from the upstream research repository by **Barathi Subramanian et al. (2022)** (*Scientific Reports*, DOI: `10.1038/s41598-022-15998-7`).

In their published paper, Subramanian et al. evaluated MoP-GRU on an author-collected 13-word ISL dataset ("There are 13 sign gestures in the dataset, as described in Table 1", achieving 95% test accuracy), LSA64 (99.92%), and WLASL100 (63.18%). The released training scripts configure these 13 word classes (`DEFAULT_ACTIONS = ["fail", "friend", "good", "hello", "iloveyou", "like", "location", "meet", "phonecall", "takecare", "thanks", "think", "you"]`):

- `mopgru_cell.py`: Custom recurrent Gated Recurrent Unit (MoP-GRU) cell implementation (configured for 13 classes).
- `collect_keypoints.py`: Holistic landmark acquisition script for 30-frame sequence recordings of the 13 gestures.
- `preprocess_keypoints.py`: Landmark temporal sequence preprocessing and normalization.
- `realtime_inference.py`: Prototype webcam real-time inference loop for word signs.
- `train_word_model.py`: Training harness for 30-frame sequence models.

**Note:** Heavy binary model checkpoints (`mopgru_model.h5`, `actionGRU.h5`) are archived in `reserve/old_models/` to avoid repository bloat. Our team's active work focuses on the 26-letter fingerspelling alphabet recognition, bimanual 144D feature extraction, multi-modal decision fusion, WordBuffer debouncing, multilingual translation, and multi-tier speech output located in `src/`.
