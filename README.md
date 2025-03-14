# MADERA 

Metagenomic Ancient DNA Evaluation and Reference-free Analysis (MADERA) is a computational pipeline designed for processing, analyzing, and clustering ancient DNA without requiring reference genomes.

[![Rust](https://img.shields.io/badge/language-Rust-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![GitHub last commit](https://img.shields.io/github/last-commit/michi-sxc/GRAFT.svg)](https://github.com/michi-sxc/GRAFT/commits/main)
[![GitHub issues](https://img.shields.io/github/issues/michi-sxc/GRAFT.svg)](https://github.com/michi-sxc/GRAFT/issues)

## Overview

MADERA processes ancient DNA datasets by analyzing intrinsic sequence properties (k-mer frequencies, GC content, damage patterns) to cluster reads and determine taxonomic origins without requiring reference genomes. This approach addresses core challenges in paleogenomics: DNA fragmentation, nucleotide modifications, and sample contamination.

Key features:

- **Reference-independent analysis**: Uses sequence characteristics rather than mapping to reference genomes
- **Ancient DNA authentication**: Detects characteristic damage patterns (e.g., C→T at 5' ends, G→A at 3' ends)
- **Feature-based clustering**: Combines k-mer frequencies, GC content, codon usage, and damage scores
- **Interactive visualization**: Web-based dashboard for exploring clustering results
- **Automated parameter optimization**: Includes methods for automatic parameter selection
- **Cluster exporting**: Export clustered reads to their own sequence files

## Installation

### Option 1: Pre-compiled Binaries (Linux only for now)

1. Download the appropriate binary for your platform from the [Releases](https://github.com/michi-sxc/MADERA/releases) page

2. Extract the archive:
   ```bash
   # Linux
   unzip madera.zip
   chmod +x madera
   ```

3. Install required dependencies:
   - **Linux**: `sudo apt-get install libopenblas-dev`

4. Move the binary to a location in your PATH (optional):
   ```bash
   # Linux
   sudo mv madera /usr/local/bin/
   ```

### Option 2: Build from Source

#### Prerequisites

- Rust (latest stable version)
- OpenBLAS (for linear algebra operations)

#### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/michi-sxc/MADERA.git
   cd madera_pipeline
   ```

2. Build with Cargo:
   ```bash
   cargo build --release
   ```

3. The binary will be available at `target/release/madera`

## Usage

### Basic Command

```bash
madera --fastq samples.fastq
```

### Common Options

```bash
# Process with automatic epsilon determination
madera --fastq samples.fastq --auto_epsilon

# Launch interactive dashboard
madera --fastq samples.fastq --dashboard

# Adjust sensitivity for short reads
madera --fastq samples.fastq --min_length 25 --k 3 
```

### Full Options Reference

| Option | Description | Default |
|--------|-------------|---------|
| `--fastq` | Input FASTQ file with ancient DNA reads | (required) |
| `--min_length` | Minimum read length after QC | 30 |
| `--k` | K-mer length for frequency analysis | 4 |
| `--output` | Output CSV file for cluster report | cluster_report.csv |
| `--dashboard` | Launch interactive dashboard | false |
| `--min_samples` | Minimum samples for DBSCAN | 5 |
| `--eps` | Epsilon radius for DBSCAN clustering | 0.5 |
| `--auto_epsilon` | Automatically determine optimal epsilon | false |
| `--pca_components` | Number of PCA components | 5 |
| `--damage_window` | Window size for damage assessment | 5 |
| `--sim_threshold` | Similarity threshold for taxonomic assignment | 0.5 |
| `--conf_threshold` | Confidence threshold for taxonomy | 0.1 |
| `--damage_threshold` | Damage threshold for authenticity | 0.5 |
| `--use_codon` | Include codon usage features | false |
| `--batch_size` | Batch size for incremental PCA | 1000 |
| `--export-clusters` | Export sequence clusters to new files | false |
| `--export-min-size` | Minimum cluster size to export | 0 |
| `--export-format` | Export format (same=original format, fasta=convert to FASTA) | same |
| `--export-output` | Output filename for exported clusters | exported_clusters.zip |
| `--export-with-metadata` | Include cluster metadata in exported files | 1000 |

## Pipeline Components

### 1. Quality Control
Filters reads based on length and quality scores to remove likely artifacts and extremely short fragments.

### 2. Feature Extraction
Extracts multiple sequence characteristics:
- K-mer frequency distributions
- GC content percentages
- Terminal damage patterns
- Optional codon usage biases

### 3. Dimensionality Reduction
Performs incremental Principal Component Analysis (PCA) to reduce the high-dimensional feature space while preserving cluster separability.

### 4. Clustering
Uses DBSCAN (Density-Based Spatial Clustering of Applications with Noise) with optimized spatial indexing through KD-trees for efficient neighbor searching.

### 5. Damage Assessment
Quantifies characteristic ancient DNA damage patterns to assign authenticity scores to clusters.

### 6. Taxonomic Assignment
Assigns preliminary taxonomic classifications based on cluster characteristics, primarily using GC content and damage profiles. (Rudimentary placeholder implementation)

### 7. Visualization Dashboard
Provides interactive web-based visualization of:
- PCA projections of reads colored by cluster
- Damage score distributions
- K-distance plots for parameter optimization
- Cluster statistics and summaries

## Example Workflow

```bash
# Step 1: Process FASTQ file with automatic epsilon detection
madera --fastq ancient_sample.fastq --auto_epsilon --min_length 35 --k 4

# Step 2: Review the cluster report
cat cluster_report.csv

# Step 3: Launch interactive dashboard to explore results
madera --fastq ancient_sample.fastq --dashboard --auto_epsilon
```

## Dashboard Features

The interactive dashboard runs on http://127.0.0.1:8080 and includes:

- PCA cluster visualization with color coding options
- Damage score distribution histograms
- K-distance plot for epsilon selection
- Cluster statistics with size, GC content, and damage metrics
- Interactive filtering and selection tools

## Implementation Details

MADERA is implemented in Rust for performance and memory safety, with key optimizations:

- Parallel processing using Rayon
- Spatial indexing with KD-trees for efficient clustering
- Incremental PCA with batch processing for memory efficiency
- WebAssembly-compatible visualization dashboard

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use MADERA in your research, please cite:

```
Schneider, M. (2025). MADERA: Metagenomic Ancient DNA Evaluation and Reference-free Analysis. 
GitHub repository: https://github.com/michi-sxc/MADERA
```

## Acknowledgments

- Inspired by existing damage assessment tools like PyDamage, PMDtools, mapDamage
