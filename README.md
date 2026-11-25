# Field Assistant — On-Device Mobile Journalism (MoJo) Assistant

Source code, formative evaluation benchmarks, HPC dataset enrichment scripts, and neural network training pipeline for the ECE Diploma Thesis:

> **"Algorithmic Assistance in Capturing, Processing, and Managing Multimedia Content in Mobile Journalism (Mobile Journalism)"**  
> **Author:** Kyriazis Lialios — School of Electrical and Computer Engineering, Aristotle University of Thessaloniki (2026)  
> **Supervisor:** Prof. Charalampos Dimoulas

- **Enriched Speech DSP Dataset & Trained Model (Zenodo DOI):** https://doi.org/10.5281/zenodo.24530146

---

## Repository Structure

```text
FieldAssistant/
├── app/                                            # Native Android application (Kotlin, Jetpack Compose, Camera2Interop)
│   └── src/main/
│       ├── java/.../                               # UI, CameraX/Camera2 manual controls, YuvConverter, Vosk ASR, PyTorch Lite
│       ├── python/                                 # Chaquopy Python kernels (CompositionAnalyzer, ExposureAnalyzer, AudioRenderer)
│       └── assets/                                 # Offline models (voxeffects_mobile.ptl, Vosk, SSD ResNet-10, SED)
└── python_scripts/
    ├── hpc/                                        # Aristotle HPC cluster (SLURM) dataset enrichment & preprocessing
    │   ├── alldaps.py                              # 6-level nested tree evaluation (2,520 VoxEffects chains × 1,300 DAPS files)
    │   ├── audit_dataset.py                        # Completeness & integrity verification of raw JSONL outputs
    │   ├── normalize_dataset.py                    # MAD outlier clipping & [0, 1] quality score normalization
    │   ├── verify_bounds.py                        # Range & outlier distribution verification
    │   ├── compare_rankings.py                     # Comparison of aggregation & ranking formulations (L1/L2/marginalized)
    │   ├── diagnose_limiter.py                     # Limiter bias & zero-variance audit (justifying head pruning)
    │   ├── extract_soft_targets.py                 # Module-specific marginalization & temperature-scaled Softmax targets
    │   ├── prepare_ml_dataset.py                   # 80/20 speaker-independent split, 20 s slicing & HD linear STFT tensors
    │   └── verify_dataset.py                       # Final target tensor & speaker-disjointness audit
    ├── colab_test/                                 # Google Colab training & formative evaluation notebooks
    │   ├── model_train.ipynb                       # KYROS-Net (ResNet-18, 5 output heads, SpecAugment, KLDivLoss) & .ptl export
    │   ├── vosk_test.ipynb                         # Offline Vosk ASR Word Error Rate (WER) evaluation across 13 DAPS environments
    │   ├── wb_benchmark.ipynb                      # 5 AWB methods & OLS calibration on SimpleCube++ (2,234 images)
    │   └── exposure_benchmark_statistics_for_thresholds.ipynb  # IEA clipping percentile & threshold analysis
    └── local_test/                                 # Local PC / Canon EOS 750D formative evaluation scripts
        ├── benchmark_on_steroids.py                # 8 exposure metrics benchmark across 4,983 IEA bracketing groups
        ├── exposure_stochastic_benchmark.xlsx      # Aggregated IEA hit-rate & stochastic offset error results
        └── blur.py                                 # Real-time DSLR focus-operator contrast benchmark (LAP1/2/4, GRA6, MIS2)
