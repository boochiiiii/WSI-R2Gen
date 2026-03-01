# WSI-R2Gen

This project adapts R2Gen for pathology `WSI` report generation.  
The input is pre-extracted WSI patch features (`.pt`), and the output is a natural-language pathology report.

## 1. Overview

- Task: `WSI features -> report text generation`
- Framework: PyTorch + DDP (`torch.distributed`)
- Entry point: `main.py`
- Core model: `models/r2gen.py` + `modules/encoder_decoder.py`
- Evaluation metrics: `BLEU-1/2/3/4`, `METEOR`, `ROUGE_L`

## 2. Project Structure

```text
WSI-R2Gen/
├── main.py
├── models/
│   └── r2gen.py
├── modules/
│   ├── dataloaders.py
│   ├── datasets.py
│   ├── encoder_decoder.py
│   ├── metrics.py
│   ├── tokenizers.py
│   └── trainer.py
├── baselines/                 # MIL and other baseline models
└── ocr/
    ├── pdf2text.py            # PDF -> text preprocessing script
    └── dataset_csv/           # example split CSV files
```

## 3. Environment & Dependencies

Recommended: Python 3.8+ with a CUDA version compatible with your local PyTorch build.

Minimum required packages:

- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `tqdm`
- `Pillow`
- `pycocoevalcap`
- `pytesseract` (only needed for OCR preprocessing)
- `PyMuPDF` (only needed for OCR preprocessing)

Example installation (adjust CUDA-specific PyTorch wheels if needed):

```bash
pip install torch torchvision
pip install numpy pandas tqdm pillow pycocoevalcap pytesseract pymupdf
```

## 4. Data Preparation

### 4.1 `image_dir` (WSI feature directory)

`modules/datasets.py` reads files in the format:

```text
{image_dir}/{case_name}.pt
```

For example: `TCGA-XX-YYYY-ZZ...pt`

### 4.2 `ann_path` (annotation directory)

Each case should have a subdirectory containing a file named `annotation`:

```text
{ann_path}/TCGA-XX-YYYY/annotation
```

The code loads this file with `json.loads`, so it should contain a JSON string (the report text).

### 4.3 `split_path` (split CSV)

The CSV must contain at least three columns: `train`, `val`, `test`.  
Each column stores case names (or slide names). The code maps them to the first three TCGA segments (e.g., `TCGA-XX-YYYY`) and matches against `ann_path`.

Example in this repo: `ocr/dataset_csv/splits_0.csv`

## 5. Training and Testing

`main.py` supports two modes: `Train` and `Test`.

### 5.1 Training

```bash
python main.py \
  --mode Train \
  --n_gpu 0 \
  --dataset_name tcga_organ \
  --image_dir /path/to/pt_files \
  --ann_path /path/to/TCGA_BRCA \
  --split_path /path/to/splits_0.csv \
  --save_dir results/BRCA \
  --record_dir records/
```

Multi-GPU example:

```bash
python main.py --mode Train --n_gpu 0,1
```

### 5.2 Testing

```bash
python main.py \
  --mode Test \
  --n_gpu 0 \
  --checkpoint_dir results/BRCA \
  --image_dir /path/to/pt_files \
  --ann_path /path/to/TCGA_BRCA \
  --split_path /path/to/splits_0.csv
```

Test mode loads:

```text
{checkpoint_dir}/model_best.pth
```

## 6. Outputs

- `save_dir/current_checkpoint.pth`: periodic checkpoint
- `save_dir/model_best.pth`: best model on validation metric
- `record_dir/{dataset_name}.csv`: experiment record log
- `save_dir/reports/*.txt`: per-sample test predictions and references

## 7. Key Arguments (`main.py`)

- `--max_fea_length`: maximum number of patch features per sample
- `--max_seq_length`: maximum report token length
- `--threshold`: vocabulary cutoff frequency (rare words -> `<unk>`)
- `--beam_size`: beam width for beam search decoding
- `--epochs / --start_val / --epochs_val`: training duration and validation schedule
- `--monitor_metric`: metric used for early stopping and best-model tracking (default: `BLEU_4`)

## 8. OCR Preprocessing (Optional)

`ocr/pdf2text.py` converts PDF reports to text and pairs them with WSIs, which is useful for building the original dataset.  
If you already have standardized `annotation` files and `.pt` features, you can skip this step.

## 9. Troubleshooting

- `RuntimeError: NCCL ...`: verify CUDA/NCCL environment and `--n_gpu` settings.
- Model not found in test mode: make sure `model_best.pth` exists under `--checkpoint_dir`.
- `pycocoevalcap` errors during evaluation: ensure it is correctly installed with all required dependencies.
- Dataset size is 0: check naming consistency across `split_path`, `ann_path`, and `image_dir`.

---

If needed, this README can be further extended with a paper-style section (method figure, experimental setup, and results tables).
