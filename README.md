# CORSAI — Pipeline README

Acoustic feature extraction, similarity modelling, and curatorial interface for underground electronic music.  
Built for the CORS Records corpus. Runs locally on macOS (Apple Silicon).

---

## Requirements

- macOS with [Miniforge](https://github.com/conda-forge/miniforge) installed
- Python 3.10 (managed via conda)
- Audio files in MP3 320kbps or WAV format

---

## 1. Environment Setup

If you have not yet created the environment, run the following in Terminal:

```bash
conda create -n cors python=3.10 -y
conda activate cors
conda install pip -y

pip install essentia librosa scikit-learn umap-learn pandas numpy \
            scipy matplotlib plotly joblib tqdm soundfile \
            streamlit jupyter ipywidgets
```

To activate the environment in future sessions:

```bash
conda activate cors
```

---

## 2. Open the Notebook in VSCode

1. Open VSCode
2. Open your project folder (e.g. `~/Desktop` or `~/Downloads`)
3. Open `cors_pipeline_v4.ipynb`
4. Select the `cors` interpreter:
   - `Cmd+Shift+P` → Python: Select Interpreter
   - Choose the one ending in `.../envs/cors/bin/python`
   - Or enter the path manually: `/opt/homebrew/Caskroom/miniforge/base/envs/cors/bin/python`

---

## 3. Configure the Pipeline

In **Cell 1 (Config)**, set your tracks folder and output directory:

```python
TRACKS_DIR = '/path/to/your/tracks'   # folder containing MP3/WAV files
OUTPUT_DIR = Path('./output')          # where all output files are saved
```

All output files are written to `OUTPUT_DIR`. The `out("filename")` helper returns the full path to any output file.

---

## 4. Run the Pipeline

Run cells top to bottom. Each stage builds on the outputs of the previous one.

| Stage | Cell | What it does | Output files |
|---|---|---|---|
| 1 | Feature extraction | Extracts ~80 acoustic features per track using librosa and Essentia. Analysis window: 45 seconds starting at 70 seconds. | `features.csv` |
| 2 | Essentia embeddings | Produces a ~35-dim tonal/rhythmic embedding per track. Stored for display only — not used in PCA. | `essentia_embeddings.csv` |
| 3 | Semantic mapping | Translates raw features into 9 mid-level descriptors (brightness, compactness, density, rhythm\_strength, texture, warmth, tonality, punch, bass\_weight, groove\_regularity). Applies threshold-based tags and BPM-constrained compound tags. | — |
| 4 | Combine → PCA | Merges features and embeddings. Fits StandardScaler and PCA (95% variance). Librosa features only go into PCA; Essentia stored for display. | `scaler.joblib`, `pca.joblib`, `pca_variance.png` |
| 5 | Cosine similarity | Computes full pairwise cosine distance matrix in PCA space. | `distance_matrix.npy` |
| 6 | K-means clustering | Sweeps k=3 to k=10, selects optimal k by silhouette score. Also computes Ward linkage dendrogram. | `silhouette.png`, `dendrogram.png`, `silhouette_scores.csv` |
| 7 | UMAP | Projects PCA space to 2D for visualisation only. Saves final dataframe. | `features_final.csv` |
| 8 | Interactive plot | Plotly UMAP scatter coloured by cluster. Hover for descriptors and tags. | — |
| 9 | Annotation tool | In-notebook widget to rate tracks on 7 curator dimensions (0–10). Skips already-rated tracks on re-run. | `annotations.csv` |
| 10 | SVR validation | Trains one linear SVR per annotation dimension on 80% of annotated tracks. Reports R² and MAE. | `svr_models/`, `svr_metrics.csv` |
| 11 | Summary | Prints pipeline stats and checks all output files are present. | — |

---

## 5. Re-running and Resuming

**Resuming extraction after interruption:**  
The extraction cell checkpoints every 50 tracks. If interrupted, simply rerun — already-extracted tracks are skipped automatically based on `track_id`.

**Forcing a full re-extraction:**  
Delete `output/features.csv` before running. This is required whenever:
- The analysis window parameters change (`WINDOW_START_S`, `WINDOW_DUR_S`)
- New features are added to `extract_features()`
- The tracks folder changes

**Re-running PCA/clustering/UMAP after extraction changes:**  
Delete the following files and rerun from Stage 4:
```
output/scaler.joblib
output/pca.joblib
output/distance_matrix.npy
output/features_final.csv
```
Keep `annotations.csv` — it is decoupled from the pipeline and persists across re-runs.

---

## 6. Manual Ambient Flag

Ambient and drone tracks cannot be reliably detected from signal features alone. Flag them manually in the run-mapping cell:

```python
AMBIENT_TRACKS = [
    'track_id_here',
    'another_track_id',
]
```

Flagged tracks have their punch and rhythm\_strength descriptors forced to 0.0 and receive the `ambient` tag.

---

## 7. Output Files Reference

| File | Description |
|---|---|
| `features.csv` | Raw acoustic features for all tracks (~80 columns) |
| `essentia_embeddings.csv` | Essentia ~35-dim embeddings (display only) |
| `features_final.csv` | Complete dataframe: features + descriptors + tags + cluster + UMAP coordinates |
| `distance_matrix.npy` | Full pairwise cosine distance matrix (n × n) |
| `scaler.joblib` | Fitted StandardScaler — required for SVR and new track inference |
| `pca.joblib` | Fitted PCA — required for SVR and new track inference |
| `silhouette_scores.csv` | Silhouette score per k value |
| `silhouette.png` | Silhouette score curve plot |
| `dendrogram.png` | Ward linkage hierarchical clustering dendrogram |
| `pca_variance.png` | PCA cumulative explained variance curve |
| `annotations.csv` | Curator annotation scores (0–10 per dimension per track) |
| `svr_metrics.csv` | R² and MAE per annotation dimension |
| `svr_models/` | Serialised SVR model per dimension (.joblib) |
| `manual_tags.csv` | Manual tag overrides assigned via the app (persists across pipeline re-runs) |

---

## 8. Launch the App

Once the pipeline has completed and `features_final.csv` and `distance_matrix.npy` exist:

```bash
cd /path/to/your/project/folder
conda activate cors
python -m streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## 9. Descriptor Reference

| Descriptor | Source feature | High value | Low value |
|---|---|---|---|
| brightness | spectral\_centroid\_mean | Treble-heavy, bright | Dark, bass-dominated |
| compactness | 1 / spectral\_spread\_mean | Spectrally focused, narrow-band | Diffuse, full-bandwidth |
| density | rms\_mean + mfcc\_mean\_0 | Loud, energetically full | Sparse, quiet |
| rhythm\_strength | beat\_count / window\_dur | Strong frequent beats | Sparse or irregular |
| texture | mean spectral contrast (7 bands) | Clear frequency layer separation | Homogeneous, blurred |
| warmth | inv. brightness + bass\_weight | Dark, bassy, analogue | Bright, cold, digital |
| tonality | inv. chroma\_entropy | Clear tonal centre | Atonal, noise-like |
| punch | low-freq onset sharpness | Compressed kick attacks | Soft, gradual onsets |
| bass\_weight | low\_freq\_ratio (<200Hz) | Dominant sub-bass / kick | Mid/high frequency focus |
| groove\_regularity | inv. ibi\_cv | Metronomic 4/4 | Broken, syncopated |

---

## 10. Known Limitations

- **Key detection** is unreliable for atonal or heavily synthesised tracks (confidence < 0.15 = treat as unknown)
- **Density descriptor** does not always correspond to the practitioner concept of fullness — heavily compressed minimal tracks may appear acoustically sparse despite sounding full (flagged in evaluation, Vera 2026)
- **Ambient detection** is not automated — use the manual flag in the mapping cell
- **UMAP layout** is spatially unstable at corpus sizes below ~200 tracks — treat as exploratory at pilot scale
- **Tag thresholds** are calibrated for the pilot corpus of ~50 tracks at 0.60; recalibrate to 0.70 when the full corpus is processed
- **ibi\_cv broken tag threshold** (0.085) is provisional — recalibrate from the full corpus distribution histogram
