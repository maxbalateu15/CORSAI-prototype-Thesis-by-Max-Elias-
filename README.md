# CORSAI-prototype-Thesis-by-Max-Elias-
A prototype tool for organising and navigating underground electronic music through acoustic similarity, practitioner-legible semantic descriptors, and corpus-relative classification. Developed as part of an MA thesis in Media Studies at the University of Amsterdam (2026).

Pipeline README

Acoustic feature extraction, similarity modelling, and curatorial interface for underground electronic music.

Built for the CORS Records corpus. Runs locally on macOS (Apple Silicon).


Requirements


macOS with Miniforge installed
Python 3.10 (managed via conda)
Audio files in MP3 320kbps or WAV format



1. Environment Setup

If you have not yet created the environment, run the following in Terminal:

bashconda create -n cors python=3.10 -y
conda activate cors
conda install pip -y

pip install essentia librosa scikit-learn umap-learn pandas numpy \
            scipy matplotlib plotly joblib tqdm soundfile \
            streamlit jupyter ipywidgets

To activate the environment in future sessions:

bashconda activate cors


2. Open the Notebook in VSCode


Open VSCode
Open your project folder (e.g. ~/Desktop or ~/Downloads)
Open cors_pipeline_v4.ipynb
Select the cors interpreter:

Cmd+Shift+P → Python: Select Interpreter
Choose the one ending in .../envs/cors/bin/python
Or enter the path manually: /opt/homebrew/Caskroom/miniforge/base/envs/cors/bin/python






3. Configure the Pipeline

In Cell 1 (Config), set your tracks folder and output directory:

pythonTRACKS_DIR = '/path/to/your/tracks'   # folder containing MP3/WAV files
OUTPUT_DIR = Path('./output')          # where all output files are saved

All output files are written to OUTPUT_DIR. The out("filename") helper returns the full path to any output file.


4. Run the Pipeline

Run cells top to bottom. Each stage builds on the outputs of the previous one.

StageCellWhat it doesOutput files1Feature extractionExtracts ~80 acoustic features per track using librosa and Essentia. Analysis window: 45 seconds starting at 70 seconds.features.csv2Essentia embeddingsProduces a ~35-dim tonal/rhythmic embedding per track. Stored for display only — not used in PCA.essentia_embeddings.csv3Semantic mappingTranslates raw features into 9 mid-level descriptors (brightness, compactness, density, rhythm_strength, texture, warmth, tonality, punch, bass_weight, groove_regularity). Applies threshold-based tags and BPM-constrained compound tags.—4Combine → PCAMerges features and embeddings. Fits StandardScaler and PCA (95% variance). Librosa features only go into PCA; Essentia stored for display.scaler.joblib, pca.joblib, pca_variance.png5Cosine similarityComputes full pairwise cosine distance matrix in PCA space.distance_matrix.npy6K-means clusteringSweeps k=3 to k=10, selects optimal k by silhouette score. Also computes Ward linkage dendrogram.silhouette.png, dendrogram.png, silhouette_scores.csv7UMAPProjects PCA space to 2D for visualisation only. Saves final dataframe.features_final.csv8Interactive plotPlotly UMAP scatter coloured by cluster. Hover for descriptors and tags.—9Annotation toolIn-notebook widget to rate tracks on 7 curator dimensions (0–10). Skips already-rated tracks on re-run.annotations.csv10SVR validationTrains one linear SVR per annotation dimension on 80% of annotated tracks. Reports R² and MAE.svr_models/, svr_metrics.csv11SummaryPrints pipeline stats and checks all output files are present.—


5. Re-running and Resuming

Resuming extraction after interruption:

The extraction cell checkpoints every 50 tracks. If interrupted, simply rerun — already-extracted tracks are skipped automatically based on track_id.

Forcing a full re-extraction:

Delete output/features.csv before running. This is required whenever:


The analysis window parameters change (WINDOW_START_S, WINDOW_DUR_S)
New features are added to extract_features()
The tracks folder changes


Re-running PCA/clustering/UMAP after extraction changes:

Delete the following files and rerun from Stage 4:

output/scaler.joblib
output/pca.joblib
output/distance_matrix.npy
output/features_final.csv

Keep annotations.csv — it is decoupled from the pipeline and persists across re-runs.


6. Manual Ambient Flag

Ambient and drone tracks cannot be reliably detected from signal features alone. Flag them manually in the run-mapping cell:

pythonAMBIENT_TRACKS = [
    'track_id_here',
    'another_track_id',
]

Flagged tracks have their punch and rhythm_strength descriptors forced to 0.0 and receive the ambient tag.


7. Output Files Reference

FileDescriptionfeatures.csvRaw acoustic features for all tracks (~80 columns)essentia_embeddings.csvEssentia ~35-dim embeddings (display only)features_final.csvComplete dataframe: features + descriptors + tags + cluster + UMAP coordinatesdistance_matrix.npyFull pairwise cosine distance matrix (n × n)scaler.joblibFitted StandardScaler — required for SVR and new track inferencepca.joblibFitted PCA — required for SVR and new track inferencesilhouette_scores.csvSilhouette score per k valuesilhouette.pngSilhouette score curve plotdendrogram.pngWard linkage hierarchical clustering dendrogrampca_variance.pngPCA cumulative explained variance curveannotations.csvCurator annotation scores (0–10 per dimension per track)svr_metrics.csvR² and MAE per annotation dimensionsvr_models/Serialised SVR model per dimension (.joblib)manual_tags.csvManual tag overrides assigned via the app (persists across pipeline re-runs)


8. Launch the App

Once the pipeline has completed and features_final.csv and distance_matrix.npy exist:

bashcd /path/to/your/project/folder
conda activate cors
python -m streamlit run app.py

The app opens automatically at http://localhost:8501.


9. Descriptor Reference

DescriptorSource featureHigh valueLow valuebrightnessspectral_centroid_meanTreble-heavy, brightDark, bass-dominatedcompactness1 / spectral_spread_meanSpectrally focused, narrow-bandDiffuse, full-bandwidthdensityrms_mean + mfcc_mean_0Loud, energetically fullSparse, quietrhythm_strengthbeat_count / window_durStrong frequent beatsSparse or irregulartexturemean spectral contrast (7 bands)Clear frequency layer separationHomogeneous, blurredwarmthinv. brightness + bass_weightDark, bassy, analogueBright, cold, digitaltonalityinv. chroma_entropyClear tonal centreAtonal, noise-likepunchlow-freq onset sharpnessCompressed kick attacksSoft, gradual onsetsbass_weightlow_freq_ratio (<200Hz)Dominant sub-bass / kickMid/high frequency focusgroove_regularityinv. ibi_cvMetronomic 4/4Broken, syncopated


10. Known Limitations


Key detection is unreliable for atonal or heavily synthesised tracks (confidence < 0.15 = treat as unknown)
Density descriptor does not always correspond to the practitioner concept of fullness — heavily compressed minimal tracks may appear acoustically sparse despite sounding full (flagged in evaluation, Vera 2026)
Ambient detection is not automated — use the manual flag in the mapping cell
UMAP layout is spatially unstable at corpus sizes below ~200 tracks — treat as exploratory at pilot scale
Tag thresholds are calibrated for the pilot corpus of ~50 tracks at 0.60; recalibrate to 0.70 when the full corpus is processed
ibi_cv broken tag threshold (0.085) is provisional — recalibrate from the full corpus distribution histogram
