# Face Recognition with Landmark Heatmaps

Minimal, production-style PyTorch project for face ID with optional landmark heatmap channels.

## Install
```bash
# Conda (recommended)
conda env create -f environment.yml
conda activate facerec

# OR via pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- On Apple Silicon, PyTorch uses MPS automatically if available (fallback to CPU otherwise).
- MediaPipe works CPU-only; no extra weights needed.
- Dlib requires the 68-point predictor; set `DLIB_LANDMARK_MODEL=/path/to/shape_predictor_68_face_landmarks.dat`. If missing, a dummy pattern is used so code still runs.

## Run
RGB baseline:
```bash
python -m src.train --data_root data/faces --use_heatmaps false
```

RGB + heatmaps + mask (default MediaPipe):
```bash
python -m src.train --data_root data/faces --use_heatmaps true --add_mask_channel true
```

Switch to dlib:
```bash
python -m src.train --data_root data/faces --extractor dlib
```

Use all landmarks as independent heatmaps (dlib-68 or mediapipe-468):
```bash
python -m src.train --data_root data/faces --extractor dlib --use_heatmaps true
# set group_all_landmarks=True in config or when constructing FaceDataset manually
```

Key flags: `--img_size 112`, `--cache_heatmaps`, `--cache_landmarks`, `--use_cosface true`, `--use_heatmaps false` (ablation).

## Design
- Landmarks → 6 grouped Gaussian heatmaps (eyes, nose, mouth corners, chin) + optional valid mask.
- Stack channels: `[RGB + heatmaps (+mask)]`.
- ResNet-50 backbone with first conv expanded to extra channels (extra filters init with RGB mean).
- Margin head: ArcFace (default) or CosFace.
- Optimizer: AdamW, cosine decay + warmup, mixed precision if CUDA.
- Joint transforms keep landmarks aligned; horizontal flip swaps eye/mouth groups.
- Evaluation: classification acc + verification ROC AUC; yaw-binned metrics if yaw provided, otherwise skipped with a note.

## Using VGGFace2 from Hugging Face
Build an ImageFolder from Hugging Face (requires `datasets` and optional HF auth). You can limit samples/classes or stream a subset to avoid downloading the full dataset:
```bash
python scripts/build_vggface2.py --dataset logasja/vggface2 --split train --out_dir data_vggface2 --limit_classes 500 --max_samples 20000
# or stream + subsample
python scripts/build_vggface2.py --dataset logasja/vggface2 --split train --out_dir data_vggface2 --streaming --sample_prob 0.2 --max_samples 50
```
Then train:
```bash
python -m src.train --data_root data_vggface2 --use_heatmaps true --add_mask_channel true --extractor mediapipe --device cuda
```

## Landmark Autoencoder (pose → canonical)
Chunks from `build_vggface2.py` can feed a small MLP that maps posed normalized landmarks + head pose to canonical (straight) landmarks:
```bash
# Build chunked landmarks without saving images
python scripts/build_vggface2.py \
  --dataset logasja/vggface2 --split train --out_dir data_vggface2 \
  --chunk_size 1000 --chunk_dir data_vggface2/chunks \
  --csv_path data_vggface2_first_row.csv --save_images false --streaming

# Train autoencoder on chunks
python -m autoencoder_train --chunk_dir data_vggface2/chunks --device cuda --epochs 10
```
Model: `LandmarkAutoencoder` (input: normalized landmarks + yaw/pitch/roll; target: canonical landmarks per identity with minimal absolute pose). Data loading/aggregation in `autoencoder_data.py` (uses pyspark if available, else numpy).

## Tests
Run a quick smoke test:
```bash
pytest tests/test_smoke.py
```
