# NurtureNet

**Explainable Five-Modality Video Safety Assessment for Children and Adolescents Using Gated Cross-Attention**

NurtureNet combines visual, acoustic, spoken, on-screen and contextual information to classify short-form videos into four content categories:

- Safe
- Harmful Content
- Adult Content
- Suicide

The final model, **GCAN V4**, achieves **80.06% test accuracy** and **79.93% Macro-F1** on 702 held-out videos.

## Dataset

The dataset combines **3,948 TikHarm records** and **775 TikHarm Extended records**, giving 4,723 initial records. Cleaning removes records with conflicting labels and resolves duplicate video IDs, producing **4,676 unique videos**.

Class distribution:

- Safe: **1,240**
- Harmful Content: **1,183**
- Adult Content: **1,118**
- Suicide: **1,135**

Stratified splits use seed **42**:

- Training: **3,273 videos**
- Validation: **701 videos**
- Test: **702 videos**

Video IDs are unique across splits. The notebook uses the saved manifest to preserve the original split membership.

## Feature Representations

Five feature vectors represent each video:

- **Video:** VideoMAE embeddings from 16 sampled RGB frames, with mean token pooling — **768 dimensions**.
- **Audio:** BEATs embeddings from 16-kHz mono audio — **768 dimensions**.
- **Speech (ASR):** Whisper large-v3-turbo transcripts encoded with multilingual BERT — **768 dimensions**.
- **On-screen text (OCR):** PP-OCRv5 text encoded with multilingual BERT — **768 dimensions**.
- **Metadata:** multilingual BERT text embeddings combined with 21 structured features — **789 dimensions**.

Metadata imputation and scaling are fitted on training data only. Saved binary masks indicate modality availability in this order:

```text
[video, audio, ASR, OCR, metadata]
```

The notebook loads precomputed features from the original project assets.

## Model Architecture

GCAN V4 projects each modality into a shared **256-dimensional** space. An **8-head attention** layer models interactions between modality tokens, while availability masks exclude missing inputs. A feed-forward network refines these representations, and learned gates determine each modality's contribution to the fused representation.

```text
Five modality features + availability masks
                    ↓
       Modality projections → 256
                    ↓
           8-head attention
                    ↓
       Feed-forward: 256 → 512 → 256
                    ↓
       Modality gates: 256 → 128 → 1
                    ↓
             Weighted fusion
                    ↓
         Classifier: 256 → 128 → 4
```

- Dropout: **0.35**
- Modality dropout: **0.10**
- Total parameters: **1,586,053**

Gate weights provide inspectable modality contributions; they are not causal explanations.

## Training Configuration

- Optimizer: **AdamW**
- Learning rate: **1e-4**
- Weight decay: **1e-4**
- Batch size: **64**
- Loss: **weighted cross-entropy**
- Class weights: **[1.00, 1.15, 1.00, 1.00]**, in the class order listed above
- Gradient clipping: **1.0**
- Maximum epochs: **40**
- Early-stopping patience: **8 epochs**
- Scheduler: **ReduceLROnPlateau**, factor **0.5**, patience **3**, minimum learning rate **1e-6**
- Checkpoint selection: **highest validation Macro-F1**
- Final selected checkpoint: **epoch 12**

The notebook includes a training loop reconstructed from the available training code and saved V4 configuration. Its default mode evaluates the original checkpoint; enabling training starts a new experiment.

## Repository Files

```text
README.md
requirements.txt
NurtureNet_Final_Training_Pipeline.ipynb
```

The notebook covers dataset verification, cleaning, splits, feature loading, masks, data loaders, GCAN, training, checkpoint selection, test evaluation, baseline comparison, ablations, per-class performance, confusion matrix, qualitative examples and inference benchmarking.

## Installation

Use **Python 3.11**. Create and activate a virtual environment:

```bash
python -m venv .venv
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r req.txt
```

CPU evaluation is supported. The notebook automatically uses CUDA when available. Dependencies are unpinned; the requirements file is not a lockfile for the historical training environment.

## Project Assets

Place the original manifests, features and model artifacts in a project directory with the following structure. These assets are separate from the notebook.

```text
NurtureNet/
├── manifests/
│   └── final_manifest.csv
├── processed/
│   └── training_features/
│       ├── feature_index.csv
│       ├── video_features.npy
│       ├── audio_features.npy
│       ├── asr_features.npy
│       ├── ocr_features.npy
│       ├── metadata_features.npy
│       └── modality_masks.npy
└── models/
    └── gcan/
        └── final_model/
            ├── gcan_final.pth
            ├── final_model_summary.json
            ├── training_history.csv
            ├── test_classification_report.csv
            ├── test_confusion_matrix.csv
            ├── final_v4_qualitative_examples.csv
            └── missing_modality_experiments/
                ├── video_only_vs_gcan.csv
                ├── video_only_classification_report.csv
                ├── video_only_training_history.csv
                └── single_modality_removal_results.csv
```

Feature rows must align with `feature_index.csv`. Preserve the original feature arrays, masks, labels and split assignments together.

## Running the Notebook

Launch JupyterLab:

```bash
python -m jupyterlab NurtureNet_Final_Training_Pipeline.ipynb
```

In section 3, set the project root:

```python
NURTURENET_ROOT = "/path/to/NurtureNet"
```

Windows example:

```python
NURTURENET_ROOT = r"G:\My Drive\NurtureNet"
```

### Evaluate the saved model

Keep these settings, then restart the kernel and run all cells:

```python
RUN_TRAINING = False
AUDIT_RAW_FILES = False
RUN_BENCHMARK = True
```

This loads the epoch-12 checkpoint, evaluates the test set, checks results against the saved artifacts and runs single-modality ablations.

### Train a new model

Set:

```python
RUN_TRAINING = True
```

New training artifacts are saved to `RUN_DIR`, which defaults to `nurturenet_run`:

```text
nurturenet_run/
├── gcan_v4_reconstructed_best.pth
└── training_history.csv
```

The evaluation stages then use the new run's best checkpoint. Original model files remain unchanged. Results may vary with hardware, package versions and random initialization.

### Audit raw dataset cleaning

Set `AUDIT_RAW_FILES = True` to check counts, label conflicts and duplicate removal against the original media directory trees. This requires:

```text
NurtureNet/Dataset_3948/raw_dataset/
NurtureNet/Dataset_775/raw_dataset/
```

### Google Colab

Upload the notebook, set `MOUNT_GOOGLE_DRIVE = True` in section 3, and use your mounted project path:

```python
NURTURENET_ROOT = "/content/drive/MyDrive/NurtureNet"
```

Adjust the path if the project is stored in a shared folder. Colab provides Google Drive mounting through its built-in package.

## Results

### GCAN V4 test performance

- Test loss: **0.6340**
- Accuracy: **80.06%**
- Macro precision: **80.31%**
- Macro recall: **80.07%**
- Macro-F1: **79.93%**

### Per-class F1

- Safe: **79.90%**
- Harmful Content: **74.30%**
- Adult Content: **76.11%**
- Suicide: **89.40%**

### Video-only comparison

The separately trained **VideoMAE + MLP** baseline achieves **71.51% accuracy** and **71.53% Macro-F1**. GCAN improves accuracy by **8.55 percentage points** and Macro-F1 by **8.40 percentage points**.

The notebook reads the baseline's saved reports and training history; it does not retrain the baseline.

### Missing-modality ablations

Removing one modality at inference changes Macro-F1 relative to the full model as follows:

- Without video: **−5.23 percentage points**
- Without metadata: **−2.89 percentage points**
- Without audio: **−1.17 percentage points**
- Without OCR: **−0.86 percentage points**
- Without ASR: **+0.36 percentage points**

These experiments mask inputs to the trained GCAN without retraining it.

## Validation and Benchmarking

The notebook's default evaluation workflow was executed sequentially using the original project assets. Test metrics, per-class scores, the confusion matrix, ablations and qualitative predictions matched the saved V4 results. Dataset checks confirmed the expected counts and zero video-ID leakage. The training code also passed a real-feature backward-pass and optimizer-step check; a full training run was not repeated.

The computational benchmark measures GCAN forward-pass latency and throughput on precomputed features. It excludes video decoding, feature extraction and host-to-device transfer, so it represents model inference performance rather than end-to-end video processing time.

## Dataset and Model Artifacts

The processed multimodal features, dataset manifests,
metadata, and trained model artifacts are available on Kaggle:

[NurtureNet Five-Modality Video Safety Artifacts](https://www.kaggle.com/datasets/abdulrehman142/nurturenet-five-modality-video-safety-artifacts)

The original raw videos are available through the
TikHarm and TikHarm Extended datasets.
