//! MADERA Pipeline in Rust: Metagenomic Ancient DNA Evaluation and Reference-free Analysis
//! 
//! This program performs quality control, feature extraction (including k-mer frequencies,
//! optional codon usage, GC content, and damage scores), incremental PCA with progress updates,
//! and clustering using DBSCAN with spatial indexing (via a kd-tree) and parallel neighbor precomputation.
//! 
//! Michael Schneider 2025
//! https://github.com/michi-sxc/MADERA
//! 
//! v0.3.0

use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use env_logger;
use log::{error, info, warn};
use ndarray::{Array1, Array2};
use serde::Serialize;
use serde_json;
use clap::{Parser, Command, Arg, ArgGroup};
use colored::*;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use rayon::prelude::*;
use bio::alignment::{pairwise::*, Alignment};
use lazy_static::lazy_static;
use std::io::Write;
use std::fs::File;
use zip::{ZipWriter, write::FileOptions};


// Module declaration for BAM utilities
mod bam_utils;
use bam_utils::*;

lazy_static! {
    static ref GLOBAL_CACHE: Arc<ReadPairCache> = Arc::new(ReadPairCache::new());
}

/// Command-line arguments and automatic clustering parameters
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Input FASTQ file for analysis
    #[arg(long, group = "input")]
    fastq: Option<String>,

    /// Input BAM file for analysis
    #[arg(long, group = "input")]
    bam: Option<String>,

    /// For BAM input, include only unmapped, only mapped, or all reads
    #[arg(long, default_value = "all", value_parser = ["unmapped", "mapped", "all"])]
    bam_filter: String,

    /// Minimum read length after QC (default: 30)
    #[arg(long, default_value_t = 30)]
    min_length: usize,

    /// K-mer length for frequency analysis (default: 4)
    #[arg(long, default_value_t = 4)]
    k: usize,

    /// Output CSV file for cluster report
    #[arg(long, default_value = "cluster_report.csv")]
    output: String,

    /// Launch interactive dashboard (web-based)
    #[arg(long, default_value_t = false)]
    dashboard: bool,

    /// Minimum samples for DBSCAN (default: 5)
    #[arg(long, default_value_t = 5)]
    min_samples: usize,

    /// Epsilon radius for DBSCAN clustering (default: 0.5)
    #[arg(long, default_value_t = 0.5)]
    eps: f64,

    /// Automatically determine optimal epsilon using k-distance method
    #[arg(long, default_value_t = false)]
    auto_epsilon: bool,

    /// Number of PCA components (default: 5)
    #[arg(long, default_value_t = 5)]
    pca_components: usize,

    /// Window size for damage assessment (default: 5)
    #[arg(long, default_value_t = 5)]
    damage_window: usize,

    /// Similarity threshold for taxonomic assignment (default: 0.5)
    #[arg(long, default_value_t = 0.5)]
    sim_threshold: f64,

    /// Confidence threshold (difference between top two similarities) (default: 0.1)
    #[arg(long, default_value_t = 0.1)]
    conf_threshold: f64,

    /// Damage threshold for authenticity (default: 0.5)
    #[arg(long, default_value_t = 0.5)]
    damage_threshold: f64,

    /// Include codon usage features in the analysis
    #[arg(long, default_value_t = false)]
    use_codon: bool,

    /// Batch size for incremental PCA (default: 1000)
    #[arg(long, default_value_t = 1000)]
    batch_size: usize,

    /// Export sequence clusters to new files
    #[arg(long, default_value_t = false)]
    export_clusters: bool,

    /// Minimum cluster size to export (clusters smaller than this will be skipped)
    #[arg(long, default_value_t = 0)]
    export_min_size: usize,

    /// Export format (same=original format, fasta=convert to FASTA)
    #[arg(long, default_value = "same", value_parser = ["same", "fasta"])]
    export_format: String,

    /// Output filename for exported clusters (will be a zip file)
    #[arg(long, default_value = "exported_clusters.zip")]
    export_output: String,
    
    /// Include cluster metadata in exported files 
    #[arg(long, default_value_t = false)]
    export_with_metadata: bool,
}

#[derive(Serialize, Clone)]
struct DashboardData {
    // PCA data
    pca: Vec<(f64, f64)>,  // Only first two dimensions for visualization
    clusters: Vec<isize>,  // Cluster assignments
    
    // Read metadata
    read_ids: Vec<String>,
    gc_content: Vec<f64>,
    
    // Damage scores
    damage_combined: Vec<f64>,
    damage5: Vec<f64>,
    damage3: Vec<f64>,
    
    // Clustering data
    k_distance: Vec<f64>,
    optimal_eps: f64,
    
    // Cluster statistics
    cluster_stats: Vec<ClusterSummary>,
    
    // Overall statistics
    total_reads: usize,
    num_clusters: usize,
    noise_percentage: f64,
    avg_damage: f64,
    
    // PCA metadata
    explained_variance: Vec<f64>,
    eigenvalues: Vec<f64>,
    total_features: usize
}

#[derive(Serialize, Clone)]
struct ClusterSummary {
    cluster_id: isize,
    size: usize,
    avg_gc: f64,
    avg_damage: f64,
    avg_damage5: f64,
    avg_damage3: f64,
    taxonomy: String,
}

// Required structs for processing
#[derive(Clone, Debug)]
pub struct Read {
    id: String,
    seq: String,
    quality: Option<Vec<u8>>,
}

#[derive(Debug, Clone)]
struct DamageScore {
    damage5: f64,
    damage3: f64, 
    combined: f64,
}

#[derive(Debug)]
struct ClusterStat {
    cluster: isize,
    num_reads: usize,
    avg_gc: f64,
    avg_damage: f64,
    avg_damage5: f64,
    avg_damage3: f64,
    taxonomy: String,
}

/// Export clusters to sequence files based on export settings (v0.3.0)
fn export_clusters(
    reads: &[Read],
    clusters: &[isize],
    cluster_stats: &HashMap<isize, ClusterStat>,
    file_type: &str,
    damage_scores: &HashMap<String, DamageScore>,
    args: &Args
) -> Result<(), Box<dyn std::error::Error>> {
    use std::path::Path;
    use std::collections::HashMap;
    use indicatif::{ProgressBar, ProgressStyle};
    
    if !args.export_clusters {
        return Ok(());
    }
    
    info!("Exporting clusters to sequence files (min size: {}, format: {})...", 
          args.export_min_size, args.export_format);
    
    // Create a map of read IDs to cluster assignments
    let mut read_to_cluster: HashMap<&str, isize> = HashMap::new();
    for (i, &cluster_id) in clusters.iter().enumerate() {
        if i < reads.len() {
            read_to_cluster.insert(&reads[i].id, cluster_id);
        }
    }
    
    // Count clusters to export for progress bar
    let clusters_to_export: Vec<isize> = cluster_stats.keys()
        .filter(|&&id| id >= 0 && cluster_stats.get(&id).map_or(0, |s| s.num_reads) >= args.export_min_size)
        .cloned()
        .collect();
    
    if clusters_to_export.is_empty() {
        info!("No clusters meet the minimum size threshold ({}). Nothing to export.", args.export_min_size);
        return Ok(());
    }
    
    info!("Preparing to export {} clusters...", clusters_to_export.len());
    
    // Create zip file
    let path = Path::new(&args.export_output);
    let file = File::create(path)?;
    let mut zip = ZipWriter::new(file);
    let options = FileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated)
        .unix_permissions(0o755);
    
    // Set up progress bar
    let pb = ProgressBar::new(clusters_to_export.len() as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} clusters exported"
        )
        .unwrap()
        .progress_chars("##-")
    );
    
    // Export each cluster
    for cluster_id in &clusters_to_export {
        let cluster_size = cluster_stats.get(cluster_id).map_or(0, |s| s.num_reads);
        let avg_gc = cluster_stats.get(cluster_id).map_or(0.0, |s| s.avg_gc);
        let avg_damage = cluster_stats.get(cluster_id).map_or(0.0, |s| s.avg_damage);
        
        // Create filename based on format
        let file_ext = match args.export_format.as_str() {
            "fasta" => "fa",
            _ => file_type,
        };
        
        // Create filename with cluster info
        let filename = format!("cluster_{}_size_{}_gc_{:.3}_damage_{:.3}.{}", 
                             cluster_id, cluster_size, avg_gc, avg_damage, file_ext);
        
        // Start file in zip
        zip.start_file(&filename, options)?;
        
        // Filter reads for this cluster
        let cluster_reads: Vec<&Read> = reads.iter()
            .filter(|r| read_to_cluster.get(r.id.as_str()) == Some(cluster_id))
            .collect();
        
        // Write reads based on format
        match args.export_format.as_str() {
            "fasta" => {
                // Export as FASTA
                for read in &cluster_reads {
                    // Create FASTA header with optional metadata
                    let header = if args.export_with_metadata {
                        let gc = compute_gc_content(&read.seq);
                        let damage = damage_scores.get(&read.id).map_or(
                            DamageScore {
                                damage5: 0.0,
                                damage3: 0.0,
                                combined: 0.0,
                            },
                            |d| d.clone()
                        );
                        format!(">{} cluster={} gc={:.3} damage5={:.3} damage3={:.3} combined={:.3}\n", 
                                read.id, cluster_id, gc, damage.damage5, damage.damage3, damage.combined)
                    } else {
                        format!(">{} cluster={}\n", read.id, cluster_id)
                    };
                    
                    // Write header and sequence
                    zip.write_all(header.as_bytes())?;
                    zip.write_all(read.seq.as_bytes())?;
                    zip.write_all(b"\n")?;
                }
            },
            _ => {
                // Export in original format
                if file_type == "fastq" {
                    // Export as FASTQ
                    for read in &cluster_reads {
                        // Convert quality scores if available
                        let qual_str = match &read.quality {
                            Some(qual) => {
                                // Convert quality scores to ASCII characters
                                qual.iter()
                                    .map(|&q| (q + 33) as char)
                                    .collect::<String>()
                            },
                            None => {
                                // Default quality (all '/' characters, Phred+33 score of 14)
                                std::iter::repeat('/').take(read.seq.len()).collect::<String>()
                            }
                        };
                        
                        // Write FASTQ entry (4 lines per read)
                        let id_line = if args.export_with_metadata {
                            format!("@{} cluster={}\n", read.id, cluster_id)
                        } else {
                            format!("@{}\n", read.id)
                        };
                        
                        zip.write_all(id_line.as_bytes())?;
                        zip.write_all(read.seq.as_bytes())?;
                        zip.write_all(b"\n+\n")?;
                        zip.write_all(qual_str.as_bytes())?;
                        zip.write_all(b"\n")?;
                    }
                } else if file_type == "bam" {
                    // For BAM files, we'll need to reference the BAM utils
                    // For simplicity, we'll use FASTQ format inside the zip
                    // but maintain all the read metadata (bam specific metadata is left out)
                    for read in &cluster_reads {
                        // Similar to FASTQ export but with note about original BAM
                        let qual_str = match &read.quality {
                            Some(qual) => qual.iter()
                                .map(|&q| (q + 33) as char)
                                .collect::<String>(),
                            None => std::iter::repeat('/').take(read.seq.len()).collect::<String>()
                        };
                        
                        let id_line = if args.export_with_metadata {
                            format!("@{} cluster={} original_format=BAM\n", read.id, cluster_id)
                        } else {
                            format!("@{} original_format=BAM\n", read.id)
                        };
                        
                        zip.write_all(id_line.as_bytes())?;
                        zip.write_all(read.seq.as_bytes())?;
                        zip.write_all(b"\n+\n")?;
                        zip.write_all(qual_str.as_bytes())?;
                        zip.write_all(b"\n")?;
                    }
                    
                    // Add a note about conversion
                    let note = format!("# Note: Original BAM format was converted to FASTQ for export.\n");
                    zip.write_all(note.as_bytes())?;
                }
            }
        }
        
        pb.inc(1);
    }
    
    // Finalize the zip file
    zip.finish()?;
    
    pb.finish_with_message(format!("Exported {} clusters to {}", clusters_to_export.len(), args.export_output));
    
    Ok(())
}

fn export_bam_clusters(
    reads: &[Read],
    clusters: &[isize],
    cluster_stats: &HashMap<isize, ClusterStat>,
    original_bam_path: &str,
    args: &Args
) -> Result<(), Box<dyn std::error::Error>> {
    use std::path::Path;
    use rust_htslib::bam::{self, Read as BamRead};
    use rust_htslib::bam::header::{Header, HeaderRecord};
    use std::collections::{HashMap, HashSet};
    use std::fs::File;
    use std::io::Write;
    use zip::ZipWriter;
    use zip::write::FileOptions;

    if !args.export_clusters || args.export_format != "same" {
        return Ok(());
    }

    info!("Exporting clusters to BAM files (min size: {})...", args.export_min_size);

    // Map read IDs to cluster assignments
    let mut read_id_to_cluster: HashMap<String, isize> = HashMap::new();
    for (i, &cluster_id) in clusters.iter().enumerate() {
        if i < reads.len() {
            read_id_to_cluster.insert(reads[i].id.clone(), cluster_id);
        }
    }

    // Filter clusters by minimum size
    let clusters_to_export: Vec<isize> = cluster_stats.keys()
        .filter(|&&id| id >= 0 && cluster_stats.get(&id).map_or(0, |s| s.num_reads) >= args.export_min_size)
        .cloned()
        .collect();

    if clusters_to_export.is_empty() {
        info!("No clusters meet the minimum size threshold ({}). Nothing to export.", args.export_min_size);
        return Ok(());
    }

    // Build a HashSet of read IDs per cluster
    let mut cluster_read_ids: HashMap<isize, HashSet<String>> = HashMap::new();
    for &cluster_id in &clusters_to_export {
        let read_ids: HashSet<String> = reads.iter()
            .filter(|r| read_id_to_cluster.get(&r.id) == Some(&cluster_id))
            .map(|r| r.id.clone())
            .collect();
        cluster_read_ids.insert(cluster_id, read_ids);
    }

    // Identify which reference IDs are used by reads in each cluster
    let mut cluster_references: HashMap<isize, HashSet<i32>> = HashMap::new();
    for &cluster_id in &clusters_to_export {
        cluster_references.insert(cluster_id, HashSet::new());
    }
    info!("Scanning BAM file to identify reference sequences for each cluster...");
    {
        let mut reader = bam::Reader::from_path(original_bam_path)?;
        for record_result in reader.records() {
            let record = record_result?;
            let read_id = String::from_utf8_lossy(record.qname()).to_string();
            if let Some(&cluster_id) = read_id_to_cluster.get(&read_id) {
                if clusters_to_export.contains(&cluster_id) && !record.is_unmapped() {
                    if let Some(ref_set) = cluster_references.get_mut(&cluster_id) {
                        ref_set.insert(record.tid());
                    }
                }
            }
        }
    }

    // Open a new reader to get the header
    let reader = bam::Reader::from_path(original_bam_path)?;
    let header_view = reader.header();

    // Create a temporary directory for cluster BAM files
    let temp_dir = tempfile::tempdir()?;
    let mut bam_paths = Vec::new();

    info!("Creating optimized BAM files for each cluster...");
    for &cluster_id in &clusters_to_export {
        let cluster_size = cluster_stats.get(&cluster_id).map_or(0, |s| s.num_reads);
        let avg_gc = cluster_stats.get(&cluster_id).map_or(0.0, |s| s.avg_gc);
        let avg_damage = cluster_stats.get(&cluster_id).map_or(0.0, |s| s.avg_damage);

        // Get the set of reference IDs used in this cluster
        let empty_set = HashSet::new();
        let ref_ids = cluster_references.get(&cluster_id).unwrap_or(&empty_set);

        let mut minimal_header = Header::new();

        // Add core header lines (HD, PG, RG, CO) from the original header
        let header_owned = header_view.to_owned();
        let header_bytes = header_owned.as_bytes();
        let header_str = std::str::from_utf8(header_bytes).unwrap();
        for line in header_str.lines() {
            if line.starts_with("@HD") || line.starts_with("@PG") ||
               line.starts_with("@RG") || line.starts_with("@CO") {
                let fields: Vec<&str> = line.split('\t').collect();
                if !fields.is_empty() {
                    let tag = &fields[0][1..]; // remove '@'
                    let mut record = HeaderRecord::new(tag.as_bytes());
                    for field in fields.iter().skip(1) {
                        if let Some(pos) = field.find(':') {
                            let key = &field[..pos];
                            let value = &field[pos + 1..];
                            record.push_tag(key.as_bytes(), value);
                        }
                    }
                    minimal_header.push_record(&record);
                }
            }
        }

        // Build a mapping from original reference IDs to new IDs.
        let mut new_tid_map: HashMap<i32, i32> = HashMap::new();
        let mut new_index = 0;
        for (i, name) in header_view.target_names().iter().enumerate() {
            let old_tid = i as i32;
            if ref_ids.contains(&old_tid) {
                new_tid_map.insert(old_tid, new_index);
                let length = header_view.target_len(i as u32).unwrap_or(0);
                let mut sq_record = HeaderRecord::new(b"SQ");
                sq_record.push_tag(b"SN", std::str::from_utf8(name).unwrap());
                sq_record.push_tag(b"LN", length as i32);
                minimal_header.push_record(&sq_record);
                new_index += 1;
            }
        }
        // If no references were used, add a dummy record for unmapped reads.
        if ref_ids.is_empty() {
            let mut sq_record = HeaderRecord::new(b"SQ");
            sq_record.push_tag(b"SN", "unmapped");
            sq_record.push_tag(b"LN", 1_i32);
            minimal_header.push_record(&sq_record);
        }

        // Build a filename and create a writer with the minimal header.
        let bam_filename = format!("cluster_{}_size_{}_gc_{:.3}_damage_{:.3}.bam",
                                   cluster_id, cluster_size, avg_gc, avg_damage);
        let bam_path = temp_dir.path().join(&bam_filename);
        let mut writer = bam::Writer::from_path(&bam_path, &minimal_header, bam::Format::Bam)?;
        bam_paths.push((bam_path.to_string_lossy().to_string(), bam_filename));

        // Write reads belonging to this cluster, updating the tid to match the new header.
        let mut cluster_reader = bam::Reader::from_path(original_bam_path)?;
        let read_ids = cluster_read_ids.get(&cluster_id).unwrap();
        for record_result in cluster_reader.records() {
            let mut record = record_result?;
            let read_id = String::from_utf8_lossy(record.qname()).to_string();
            if read_ids.contains(&read_id) {
                if !record.is_unmapped() {
                    // Remap the reference id using new_tid_map.
                    if let Some(&new_tid) = new_tid_map.get(&record.tid()) {
                        record.set_tid(new_tid);
                    } else {
                        // If the mapping is missing, skip this record.
                        continue;
                    }
                }
                writer.write(&record)?;
            }
        }
    }

    // Package all cluster BAM files into a zip archive.
    info!("Packaging optimized BAM files into zip archive...");
    let zip_path = Path::new(&args.export_output);
    let file = File::create(zip_path)?;
    let mut zip = ZipWriter::new(file);
    let options = FileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated)
        .unix_permissions(0o755);
    for (path, filename) in bam_paths {
        let bam_data = std::fs::read(path)?;
        zip.start_file(filename, options)?;
        zip.write_all(&bam_data)?;
    }
    zip.finish()?;

    info!("Exported {} clusters to {}", clusters_to_export.len(), args.export_output);
    info!("Each BAM file contains only the references used by its reads");

    Ok(())
}



/// Quality control for FASTQ files
fn quality_control(fastq_path: &str, min_length: usize) -> Vec<Read> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use std::path::Path;
    
    let path = Path::new(fastq_path);
    let file = match File::open(path) {
        Ok(file) => file,
        Err(e) => {
            error!("Failed to open FASTQ file: {}", e);
            return Vec::new();
        }
    };
    
    let reader = BufReader::new(file);
    let mut reads = Vec::new();
    let mut total_reads = 0;
    let mut short_count = 0;
    let mut passed_reads = 0;
    
    // Read FASTQ format: 4 lines per read
    // Line 1: Header (starts with @)
    // Line 2: Sequence
    // Line 3: + (sometimes followed by repeat of header)
    // Line 4: Quality scores
    
    let mut lines = reader.lines();
    while let Some(Ok(header_line)) = lines.next() {
        total_reads += 1;
        
        // FASTQ header should start with @
        if !header_line.starts_with('@') {
            warn!("Invalid FASTQ format, header doesn't start with @: {}", header_line);
            continue;
        }
        
        // Extract ID from header (remove @ and take until first whitespace)
        let id = header_line[1..].split_whitespace().next().unwrap_or("unknown").to_string();
        
        // Get sequence line
        let seq = if let Some(Ok(line)) = lines.next() {
            line
        } else {
            warn!("Invalid FASTQ format, missing sequence for read: {}", id);
            continue;
        };
        
        // Skip the + line
        if let Some(Ok(_)) = lines.next() {
            // This is just the + line, sometimes with repeat of header
        } else {
            warn!("Invalid FASTQ format, missing + line for read: {}", id);
            continue;
        }
        
        // Get quality line
        let quality = if let Some(Ok(line)) = lines.next() {
            line
        } else {
            warn!("Invalid FASTQ format, missing quality for read: {}", id);
            continue;
        };
        
        // Check if sequence meets minimum length requirement
        if seq.len() < min_length {
            short_count += 1;
            continue;
        }
        
        // Convert quality string to numeric values
        let quality_values = quality.bytes().map(|b| b - 33).collect();
        
        // Create Read object and add to collection
        let read = Read {
            id: id.clone(),
            seq: seq,
            quality: Some(quality_values),
        };
        
        reads.push(read);
        passed_reads += 1;
        
        // Log progress periodically
        if total_reads % 100000 == 0 {
            info!("Processed {} FASTQ records...", total_reads);
        }
    }
    
    // Debug stats
    info!("FASTQ statistics:");
    info!("  Total records: {}", total_reads);
    info!("  Too short: {}", short_count);
    info!("FASTQ QC: {} out of {} reads passed filtering and minimum length", 
           passed_reads, total_reads);
    
    reads
}

/// Compute GC content for a collection of reads
fn compute_gc_for_reads(reads: &[Read]) -> HashMap<String, f64> {
    use rayon::prelude::*;
    use std::sync::{Arc, Mutex};
    use indicatif::{ProgressBar, ProgressStyle};
    
    let total_reads = reads.len();
    let pb = Arc::new(ProgressBar::new(total_reads as u64));
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} reads processed (GC content)"
        )
        .unwrap()
        .progress_chars("##-")
    );
    
    // counter
    let processed_count = Arc::new(Mutex::new(0));
    
    // results hashmap
    let gc_scores = Arc::new(Mutex::new(HashMap::with_capacity(total_reads)));
    
    let chunk_size = std::cmp::max(1, total_reads / rayon::current_num_threads());
    
    let chunks: Vec<_> = reads.chunks(chunk_size).collect();
    
    chunks.par_iter().for_each(|chunk| {
        let mut local_scores = HashMap::with_capacity(chunk.len());
        
        // Process each read in the chunk
        for read in *chunk {
            let gc = compute_gc_content(&read.seq);
            local_scores.insert(read.id.clone(), gc);
        }
        
        // Update the global scores
        let mut scores = gc_scores.lock().unwrap();
        scores.extend(local_scores);
        
        // Update progress
        let pb_clone = Arc::clone(&pb);
        let processed = Arc::clone(&processed_count);
        let mut count = processed.lock().unwrap();
        *count += chunk.len();
        pb_clone.set_position(*count as u64);
    });
    
    pb.finish_with_message("GC content calculation complete");
    
    // Return the results
    let result = Arc::try_unwrap(gc_scores)
        .expect("Failed to unwrap Arc")
        .into_inner()
        .expect("Failed to unwrap Mutex");
    
    result
}

/// Compute GC content for a single sequence
fn compute_gc_content(seq: &str) -> f64 {
    if seq.is_empty() {
        return 0.0;
    }
    
    let gc_count = seq.chars()
        .filter(|&c| c == 'G' || c == 'g' || c == 'C' || c == 'c')
        .count();
    
    gc_count as f64 / seq.len() as f64
}

/// Calculate damage scores for ancient DNA reads
fn damage_assessment(reads: &[Read], window_size: usize) -> HashMap<String, DamageScore> {
    use rayon::prelude::*;
    use std::sync::{Arc, Mutex};
    use indicatif::{ProgressBar, ProgressStyle};
    
    let total_reads = reads.len();
    let pb = Arc::new(ProgressBar::new(total_reads as u64));
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} reads processed (Damage assessment)"
        )
        .unwrap()
        .progress_chars("##-")
    );
    
    let processed_count = Arc::new(Mutex::new(0));
    
    // Pre-allocate the results hashmap
    let damage_scores = Arc::new(Mutex::new(HashMap::with_capacity(total_reads)));
    
    let chunk_size = std::cmp::max(1, total_reads / rayon::current_num_threads());
    
    let chunks: Vec<_> = reads.chunks(chunk_size).collect();
    
    chunks.par_iter().for_each(|chunk| {
        let mut local_scores = HashMap::with_capacity(chunk.len());
        
        // Process each read in the chunk
        for (i, read) in chunk.iter().enumerate() {
            let damage = compute_damage_scores(read, window_size);
            local_scores.insert(read.id.clone(), damage);
            
            // Periodically update progress for very large chunks
            if i % 10 == 0 && chunk.len() > 100 {
                let pb_clone = Arc::clone(&pb);
                let processed = Arc::clone(&processed_count);
                let mut count = processed.lock().unwrap();
                *count += 10;
                pb_clone.set_position(*count as u64);
            }
        }
        
        // Update the global scores
        let mut scores = damage_scores.lock().unwrap();
        scores.extend(local_scores);
        
        // Final progress update for this chunk
        let pb_clone = Arc::clone(&pb);
        let processed = Arc::clone(&processed_count);
        let mut count = processed.lock().unwrap();
        
        // Only count remaining reads that weren't already counted in the periodic updates
        let remaining = chunk.len() - (chunk.len() / 10) * 10;
        *count += remaining;
        
        pb_clone.set_position(*count as u64);
    });
    
    pb.finish_with_message("Damage assessment complete");
    
    // Return the results
    let result = Arc::try_unwrap(damage_scores)
        .expect("Failed to unwrap Arc")
        .into_inner()
        .expect("Failed to unwrap Mutex");
    
    result
}

// Add this debugging helper function 
fn debug_damage_computation(read: &Read, window_size: usize) -> DamageScore {
    // Add timestamps for benchmarking each step
    let start = std::time::Instant::now();
    
    // Get the max window size to analyze (we'll use a dynamic approach)
    let max_window = std::cmp::min(window_size * 3, read.seq.len() / 4);
    
    // 1. First approach: Change point analysis
    let change_point_start = std::time::Instant::now();
    let change_point_results = detect_damage_change_points(read, max_window);
    let change_point_duration = change_point_start.elapsed();
    log::debug!("Change point analysis took {:?}", change_point_duration);
    
    // 2. Second approach: Position-specific decay model
    let decay_model_start = std::time::Instant::now();
    let decay_model_results = fit_decay_damage_model(read, max_window);
    let decay_model_duration = decay_model_start.elapsed();
    log::debug!("Decay model fitting took {:?}", decay_model_duration);
    
    // 3. Third approach: Read pair overlap analysis if paired data is available
    let read_pair_start = std::time::Instant::now();
    let read_pair_results = analyze_read_pair_overlaps(read);
    let read_pair_duration = read_pair_start.elapsed();
    log::debug!("Read pair analysis took {:?}", read_pair_duration);
    
    // 4. Combine results using an ensemble approach
    let combine_start = std::time::Instant::now();
    let combined_score = combine_damage_evidence(
        &change_point_results, 
        &decay_model_results,
        &read_pair_results,
        read
    );
    let combine_duration = combine_start.elapsed();
    log::debug!("Evidence combination took {:?}", combine_duration);
    
    // Get total duration
    let total_duration = start.elapsed();
    log::debug!("Total damage computation took {:?}", total_duration);
    
    // Return the comprehensive damage score
    DamageScore {
        damage5: combined_score.prime5_score,
        damage3: combined_score.prime3_score,
        combined: combined_score.authenticity_score,
    }
}

/// Enhanced structs needed for the new damage assessment (some parameters are still unused)
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct DamagePosition {
    position: usize,
    base: char,
    context: String,
    quality: Option<u8>,
    damage_prob: f64,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct DamagePattern {
    positions: Vec<DamagePosition>,
    decay_rate: f64,
    change_point: Option<usize>,
    log_likelihood_ratio: f64,
}


/// Main damage assessment function - integrates multiple methods
fn compute_damage_scores(read: &Read, window_size: usize) -> DamageScore {
    // Get the max window size to analyze (we'll use a dynamic approach)
    let max_window = std::cmp::min(window_size * 3, read.seq.len() / 4);
    
    // 1. First approach: Change point analysis
    let change_point_results = detect_damage_change_points(read, max_window);
    
    // 2. Second approach: Position-specific decay model
    let decay_model_results = fit_decay_damage_model(read, max_window);
    
    // 3. Third approach: Read pair overlap analysis if paired data is available
    // This would be implemented via a cache of read pairs - we simulate it here
    let read_pair_results = analyze_read_pair_overlaps(read);
    
    // 4. Combine results using an ensemble approach
    let combined_score = combine_damage_evidence(
        &change_point_results, 
        &decay_model_results,
        &read_pair_results,
        read
    );
    
    // Better damage scoring logic than in v0.1.0:
    // If we couldn't find a read pair but the change point and decay model
    // provide strong signals, still assign a reasonable damage score
    let final_damage = if read_pair_results.has_overlap {
        // We have paired read data, use the full combined score
        DamageScore {
            damage5: combined_score.prime5_score,
            damage3: combined_score.prime3_score,
            combined: combined_score.authenticity_score,
        }
    } else {
        // No paired data, but we can still estimate damage
        // Compute a slightly reduced score based only on single-read methods
        let damage5 = combined_score.prime5_score * 0.9;
        let damage3 = combined_score.prime3_score * 0.9;
        let combined = damage5 * 0.5 + damage3 * 0.5;
        
        DamageScore {
            damage5,
            damage3,
            combined,
        }
    };
    
    final_damage
}

/// APPROACH 1: Change-point analysis for damage detection
/// This method finds where damage patterns significantly change along the read
fn detect_damage_change_points(read: &Read, window_size: usize) -> ChangePointResults {
    let seq = &read.seq;
    let seq_len = seq.len();
    
    // Skip very short reads
    if seq_len < 15 {
        return ChangePointResults {
            prime5_pvalue: 1.0,
            prime3_pvalue: 1.0,
            prime5_score: 0.0,
            prime3_score: 0.0,
        };
    }
    
    // Extract sequence segments for analysis
    let five_prime = &seq[0..std::cmp::min(window_size, seq_len / 2)];
    let three_prime = &seq[seq_len.saturating_sub(std::cmp::min(window_size, seq_len / 2))..];
    
    // Collect base frequencies at each position
    let five_freqs = collect_position_frequencies(five_prime, true);
    let three_freqs = collect_position_frequencies(three_prime, false);
    
    // Run change point detection algorithm
    let five_change = find_change_point(&five_freqs, 'T', 'C');
    let three_change = find_change_point(&three_freqs, 'A', 'G');
    
    // Return results
    ChangePointResults {
        prime5_pvalue: five_change.pvalue,
        prime3_pvalue: three_change.pvalue,
        prime5_score: calculate_change_point_score(&five_freqs, five_change.position),
        prime3_score: calculate_change_point_score(&three_freqs, three_change.position),
    }
}

/// Collect frequency of each base at each position
fn collect_position_frequencies(segment: &str, _is_five_prime: bool) -> Vec<[f64; 4]> {
    let len = segment.len();
    let mut freqs = vec![[0.0; 4]; len]; // [A, C, G, T] at each position
    
    let bases = segment.chars().collect::<Vec<_>>();
    
    for (i, base) in bases.iter().enumerate() {
        let idx = match base.to_ascii_uppercase() {
            'A' => 0,
            'C' => 1,
            'G' => 2,
            'T' => 3,
            _ => continue, // Skip non-ACGT bases
        };
        
        // We're analyzing just one read, so frequency is either 0 or 1
        freqs[i][idx] = 1.0;
    }
    
    freqs
}

/// Find the point where base frequencies significantly change
/// Using a statistical change point detection algorithm
fn find_change_point(position_freqs: &Vec<[f64; 4]>, target_base_char: char, _source_base_char: char) -> ChangePointResult {
    let target_base = base_to_index(target_base_char);
    let num_positions = position_freqs.len();
    
    if num_positions < 6 {
        return ChangePointResult {
            position: None,
            pvalue: 1.0,
        };
    }
    
    // Compute cumulative frequencies
    let mut cumul_target = vec![0.0; num_positions + 1];
    let mut cumul_total = vec![0.0; num_positions + 1];
    
    for i in 0..num_positions {
        cumul_target[i+1] = cumul_target[i] + position_freqs[i][target_base];
        cumul_total[i+1] = cumul_total[i] + 1.0; // Each position has one base
    }
    
    // Find the position with maximum CUSUM statistic
    let mut max_statistic = 0.0;
    let mut change_pos = None;
    
    // Calculate background rate (from positions far from end)
    let background_start = std::cmp::min(6, num_positions / 2);
    let background_end = num_positions;
    let background_count = cumul_target[background_end] - cumul_target[background_start];
    let background_total = cumul_total[background_end] - cumul_total[background_start];
    let background_rate = if background_total > 0.0 {
        background_count / background_total
    } else {
        0.25 // Default to uniform distribution
    };
    
    // Binary segmentation to find the change point
    for k in 2..(num_positions - 2) {
        let n1 = cumul_total[k] - cumul_total[0];
        let n2 = cumul_total[num_positions] - cumul_total[k];
        
        if n1 < 3.0 || n2 < 3.0 {
            continue; // Need at least 3 positions for statistics
        }
        
        let count1 = cumul_target[k] - cumul_target[0];
        let count2 = cumul_target[num_positions] - cumul_target[k];
        
        let p1 = count1 / n1;
        let p2 = count2 / n2;
        
        // Calculate CUSUM statistic
        let statistic = (p1 - p2).abs() * (n1 * n2 / (n1 + n2)).sqrt();
        
        if statistic > max_statistic {
            max_statistic = statistic;
            change_pos = Some(k);
        }
    }
    
    // Calculate p-value using bootstrap method
    let pvalue = if max_statistic > 0.0 {
        let z_score = max_statistic / background_rate.sqrt();
        approximate_pvalue(z_score)
    } else {
        1.0
    };
    
    ChangePointResult {
        position: change_pos,
        pvalue,
    }
}

/// Convert base character to array index
fn base_to_index(base: char) -> usize {
    match base.to_ascii_uppercase() {
        'A' => 0,
        'C' => 1,
        'G' => 2,
        'T' => 3,
        _ => 0, // Default
    }
}

/// Calculate score based on change point
fn calculate_change_point_score(position_freqs: &Vec<[f64; 4]>, change_point: Option<usize>) -> f64 {
    let cp = match change_point {
        Some(pos) => pos,
        None => return 0.1, // No significant change point
    };
    
    // If change point is detected, calculate damage score
    if cp < 3 || cp >= position_freqs.len() - 2 {
        // Change point too close to edges - less reliable
        return 0.3;
    }
    
    // Calculate average target base frequency before and after change point
    let t_idx = base_to_index('T');
    let a_idx = base_to_index('A');
    
    let before_t_freq: f64 = position_freqs[0..cp].iter().map(|f| f[t_idx]).sum::<f64>() / cp as f64;
    let after_t_freq: f64 = position_freqs[cp..].iter().map(|f| f[t_idx]).sum::<f64>() / (position_freqs.len() - cp) as f64;
    
    let before_a_freq: f64 = position_freqs[0..cp].iter().map(|f| f[a_idx]).sum::<f64>() / cp as f64;
    let after_a_freq: f64 = position_freqs[cp..].iter().map(|f| f[a_idx]).sum::<f64>() / (position_freqs.len() - cp) as f64;
    
    // Higher score if we see a dramatic drop at the change point
    let t_ratio = if after_t_freq > 0.0 { before_t_freq / after_t_freq } else { 1.0 };
    let a_ratio = if after_a_freq > 0.0 { before_a_freq / after_a_freq } else { 1.0 };
    
    let score = ((t_ratio - 1.0) + (a_ratio - 1.0)) / 2.0;
    0.2 + 0.6 * (score.min(2.0) / 2.0) // Normalize to [0.2, 0.8]
}


/// Calculate pvalue (this replaces deprecated approximation calculation)
fn approximate_pvalue(z_score: f64) -> f64 {
    // For z < 0, we define p-value as 1.0 (one-tailed test)
    if z_score < 0.0 {
        return 1.0;
    }
    // Use the complementary error function for a m approximation.
    // The one-tailed p-value is given by 0.5 * erfc(z / sqrt(2)).
    0.5 * libm::erfc(z_score / 1.4142135623730951)
}


/// APPROACH 2: Exponential decay model for damage 
/// This method fits a model where damage decays exponentially from read ends
fn fit_decay_damage_model(read: &Read, window_size: usize) -> DecayModelResults {
    let seq = &read.seq;
    let seq_len = seq.len();
    
    if seq_len < 10 {
        return DecayModelResults {
            prime5_lambda: 0.0,
            prime3_lambda: 0.0,
            prime5_model_fit: 0.0,
            prime3_model_fit: 0.0,
            log_likelihood_ratio: 0.0,
        };
    }
    
    // Extract sequence segments
    let five_prime = &seq[0..std::cmp::min(window_size, seq_len / 2)];
    let three_prime = &seq[seq_len.saturating_sub(std::cmp::min(window_size, seq_len / 2))..];
    
    // Determine if bases are potential deaminated sites
    let five_deam = identify_potential_deamination(five_prime, true);
    let three_deam = identify_potential_deamination(three_prime, false);
    
    // Fit exponential decay model to observed damage patterns
    let five_model = fit_exponential_model(&five_deam);
    let three_model = fit_exponential_model(&three_deam);
    
    // Calculate log-likelihood ratio (damage model vs. uniform model)
    let llr = calculate_log_likelihood_ratio(&five_deam, &three_deam, &five_model, &three_model);
    
    DecayModelResults {
        prime5_lambda: five_model.lambda,
        prime3_lambda: three_model.lambda,
        prime5_model_fit: five_model.fit_quality,
        prime3_model_fit: three_model.fit_quality,
        log_likelihood_ratio: llr,
    }
}

/// Identify positions that may have undergone deamination
fn identify_potential_deamination(segment: &str, is_five_prime: bool) -> Vec<(usize, f64)> {
    let mut deam_sites = Vec::new();
    let bases = segment.chars().collect::<Vec<_>>();
    
    for (i, &base) in bases.iter().enumerate() {
        let (damage_prob, is_deaminated) = if is_five_prime {
            // 5' end: Look for C→T transitions
            match base.to_ascii_uppercase() {
                'T' => {
                    // Context analysis
                    let (_, prob) = analyze_five_prime_context(i, &bases);
                    (prob, true)
                },
                'C' => (0.1, false), // C that could deaminate but hasn't
                _ => (0.0, false),   // Other bases not relevant
            }
        } else {
            // 3' end: Look for G→A transitions
            match base.to_ascii_uppercase() {
                'A' => {
                    // A could be a deaminated G
                    // Context analysis for 3' end
                    let (_, prob) = analyze_three_prime_context(i, &bases);
                    (prob, true)
                },
                'G' => {
                    // G that could deaminate but hasn't yet
                    // Susceptibility analysis
                    let (_, prob) = analyze_g_susceptibility(i, &bases);
                    (prob, false)
                },
                _ => (0.0, false),   // Other bases not relevant
            }
        };
        
        if is_deaminated || damage_prob > 0.0 {
            deam_sites.push((i, damage_prob));
        }
    }
    
    deam_sites
}

/// Analyze the sequence context at 5' end for more accurate damage assessment
fn analyze_five_prime_context(pos: usize, bases: &[char]) -> (&'static str, f64) {
    if pos + 1 >= bases.len() {
        return ("end", 0.6);
    }
    
    let next_base = bases[pos + 1].to_ascii_uppercase();
    
    // CpG has highest deamination rate
    if next_base == 'G' {
        ("CpG", 0.85)
    } 
    // Check if within GC-rich context (higher deamination propensity)
    else if pos + 3 < bases.len() {
        let gc_count = bases[pos+1..pos+4].iter()
            .filter(|&&b| b.to_ascii_uppercase() == 'G' || b.to_ascii_uppercase() == 'C')
            .count();
        
        if gc_count >= 2 {
            ("GC-rich", 0.75)
        } else {
            ("standard", 0.6)
        }
    } else {
        ("standard", 0.6)
    }
}

/// Analyze 3' end context for A that may be deaminated G
fn analyze_three_prime_context(pos: usize, bases: &[char]) -> (&'static str, f64) {
    // Distance from 3' end (for weighting)
    let distance_from_end = bases.len().saturating_sub(pos + 1);
    
    // Position factor - damage decreases with distance from end
    let position_factor = match distance_from_end {
        0 => 1.0,             // Last base
        1 => 0.9,             // Second-to-last base
        2 => 0.8,             // Third-to-last base
        d if d < 5 => 0.7,    // Positions 3-4 from end
        d if d < 10 => 0.6,   // Positions 5-9 from end
        _ => 0.5,             // Further positions
    };
    
    // Terminal position - special case
    if pos == 0 {
        return ("terminal", 0.6 * position_factor);
    }
    
    // CpG context check (highest deamination rate)
    let prev_base = bases[pos - 1].to_ascii_uppercase();
    if prev_base == 'C' {
        return ("CpG", 0.8 * position_factor);
    }
    
    // GC content analysis (higher GC = higher deamination rate)
    if pos >= 3 {
        let window_size = std::cmp::min(3, pos);
        let gc_count = bases[pos-window_size..pos].iter()
            .filter(|&&b| b.to_ascii_uppercase() == 'G' || b.to_ascii_uppercase() == 'C')
            .count();
        
        let gc_ratio = gc_count as f64 / window_size as f64;
        
        if gc_ratio > 0.66 {
            return ("GC-rich", (0.7 * position_factor));
        } else if gc_ratio > 0.33 {
            return ("mixed-GC", (0.6 * position_factor));
        }
    }
    
    // Default context
    ("standard", 0.55 * position_factor)
}

/// Analyze C for deamination susceptibility (when it hasn't deaminated yet)
#[allow(dead_code)]
fn analyze_c_susceptibility(pos: usize, bases: &[char]) -> (&'static str, f64) {
    // CpG context check (highest susceptibility)
    if pos + 1 < bases.len() && bases[pos + 1].to_ascii_uppercase() == 'G' {
        return ("CpG", 0.2);
    }
    
    // Position from 5' end
    let position_factor = if pos < 3 {
        0.15  // Higher susceptibility near 5' end
    } else if pos < 7 {
        0.12
    } else {
        0.08  // Lower susceptibility further from end
    };
    
    ("standard", position_factor)
}

/// Analyze G for deamination susceptibility (when it hasn't deaminated yet)
fn analyze_g_susceptibility(pos: usize, bases: &[char]) -> (&'static str, f64) {
    // CpG context check (highest susceptibility)
    if pos > 0 && bases[pos - 1].to_ascii_uppercase() == 'C' {
        return ("CpG", 0.18);
    }
    
    // Distance from 3' end
    let distance_from_end = bases.len().saturating_sub(pos + 1);
    let position_factor = if distance_from_end < 3 {
        0.14  // Higher susceptibility near 3' end
    } else if distance_from_end < 7 {
        0.1
    } else {
        0.07  // Lower susceptibility further from end
    };
    
    ("standard", position_factor)
}

/// Fit an exponential decay model to potential deamination sites
fn fit_exponential_model(deam_sites: &[(usize, f64)]) -> ExponentialModel {
    if deam_sites.len() < 3 {
        return ExponentialModel {
            lambda: 0.0,
            amplitude: 0.0,
            fit_quality: 0.0,
        };
    }
    
    // Calculate initial estimates based on observed pattern
    // Lambda controls how quickly damage decreases from end
    let pos_probs: Vec<(f64, f64)> = deam_sites.iter()
        .map(|&(pos, prob)| (pos as f64, prob))
        .collect();
    
    // Try a range of lambda values and find the best fit
    // This is a simplified numerical optimization
    let mut best_lambda = 0.0;
    let mut best_amplitude = 0.0;
    let mut best_fit = 0.0;
    
    for lambda_test in (1..50).map(|x| x as f64 * 0.05) {
        let (amplitude, fit) = test_exponential_fit(&pos_probs, lambda_test);
        if fit > best_fit {
            best_fit = fit;
            best_lambda = lambda_test;
            best_amplitude = amplitude;
        }
    }
    
    ExponentialModel {
        lambda: best_lambda,
        amplitude: best_amplitude,
        fit_quality: best_fit,
    }
}

/// Test how well a specific lambda value fits the data
fn test_exponential_fit(positions: &[(f64, f64)], lambda: f64) -> (f64, f64) {
    if positions.is_empty() || lambda <= 0.0 {
        return (0.0, 0.0);
    }
    
    // Exponential model: damage_prob = amplitude * exp(-lambda * position)
    // Estimate amplitude from first position
    let mut sum_predicted = 0.0;
    let mut sum_actual = 0.0;
    let mut sum_squared_diff = 0.0;
    let mut count = 0.0;
    
    for &(pos, prob) in positions {
        if prob > 0.0 {
            let predicted = (-lambda * pos).exp();
            sum_predicted += predicted;
            sum_actual += prob;
            sum_squared_diff += (predicted - prob).powi(2);
            count += 1.0;
        }
    }
    
    // Calculate amplitude that minimizes error
    let amplitude = if sum_predicted > 0.0 {
        sum_actual / sum_predicted
    } else {
        0.0
    };
    
    // Calculate fit quality (1.0 = perfect fit, 0.0 = no fit)
    let fit_quality = if count > 0.0 && sum_actual > 0.0 {
        1.0 - (sum_squared_diff / count).sqrt() / (sum_actual / count)
    } else {
        0.0
    };
    
    (amplitude, fit_quality.max(0.0))
}

/// Calculate log-likelihood ratio between damage model and null model
fn calculate_log_likelihood_ratio(
    deam5: &[(usize, f64)],
    deam3: &[(usize, f64)],
    model5: &ExponentialModel,
    model3: &ExponentialModel
) -> f64 {
    if deam5.is_empty() && deam3.is_empty() {
        return 0.0;
    }
    
    let mut ll_damage = 0.0;
    let mut ll_null = 0.0;
    
    // Calculate likelihood under damage model for 5' end
    for &(pos, prob) in deam5 {
        if prob > 0.0 {
            let model_prob = model5.amplitude * (-model5.lambda * pos as f64).exp();
            ll_damage += prob * model_prob.ln() + (1.0 - prob) * (1.0 - model_prob).ln();
            
            // Null model: uniform probability of damage
            let null_prob: f64 = 0.25; // Assuming 25% chance of any base
            ll_null += prob * null_prob.ln() + (1.0 - prob) * (1.0 - null_prob).ln();
        }
    }
    
    // Calculate likelihood under damage model for 3' end
    for &(pos, prob) in deam3 {
        if prob > 0.0 {
            let model_prob = model3.amplitude * (-model3.lambda * pos as f64).exp();
            ll_damage += prob * model_prob.ln() + (1.0 - prob) * (1.0 - model_prob).ln();
            
            // Null model: uniform probability of damage
            let null_prob: f64 = 0.25; // Assuming 25% chance of any base
            ll_null += prob * null_prob.ln() + (1.0 - prob) * (1.0 - null_prob).ln();
        }
    }
    
    // Return log-likelihood ratio
    ll_damage - ll_null
}


/// APPROACH 3: Read pair overlap analysis for damage detection
/// New read pair overlap analysis system for ancient DNA authentication (v0.2.0)
/// This subsystem provides reference-free detection of damage patterns
/// by analyzing mismatches in overlapping regions of paired-end reads

/// Cache for paired reads to avoid redundant alignment calculations
#[derive(Default)]
pub struct ReadPairCache {
    // Maps read ID to its paired read information
    cache: RwLock<HashMap<String, PairedReadInfo>>,
    // Cache overlap analysis results
    overlap_results: RwLock<HashMap<String, ReadPairResults>>,
    // Tracks paired read IDs for batch resolution
    unprocessed_pairs: Mutex<HashSet<String>>,
    // Statistics for cache performance
    stats: Mutex<CacheStats>,
}

#[derive(Default, Clone)]
pub struct CacheStats {
    lookups: usize,
    hits: usize,
    inserts: usize,
    pair_found: usize,
    pair_not_found: usize,
    successful_alignments: usize,
}

/// Information about a read and its paired read
#[derive(Clone)]
struct PairedReadInfo {
    #[allow(dead_code)]
    read1_id: String,
    #[allow(dead_code)]
    read2_id: String,
    read1_seq: String, 
    read2_seq: String,
    read1_qual: Option<Vec<u8>>,
    read2_qual: Option<Vec<u8>>,
    #[allow(dead_code)]
    processed: bool,
}

/// Detailed mismatch information for overlap analysis
#[derive(Clone, Debug)]
struct MismatchInfo {
    position: usize,      // Position within the overlap region
    read1_pos: usize,     // Original position in read1
    #[allow(dead_code)]
    read2_pos: usize,     // Original position in read2
    #[allow(dead_code)]
    read1_base: char,     // Base in read1
    #[allow(dead_code)]
    read2_base: char,     // Base in read2
    read1_qual: Option<u8>, // Quality score for read1 base
    read2_qual: Option<u8>, // Quality score for read2 base
    is_damage_pattern: bool, // Whether this matches an aDNA damage pattern
    pattern_type: DamagePatternType, // Type of damage pattern if any
}

/// Types of damage patterns in ancient DNA
#[derive(Clone, Copy, Debug, PartialEq)]
enum DamagePatternType {
    CTTransition5Prime,  // C→T at 5' end (read1)
    GATransition3Prime,  // G→A at 3' end (read1)
    CTTransition3Prime,  // C→T at 3' end (read2)
    GATransition5Prime,  // G→A at 5' end (read2)
    OtherMismatch,       // Any other mismatch
}

impl ReadPairCache {
    pub fn new() -> Self {
        ReadPairCache {
            cache: RwLock::new(HashMap::new()),
            overlap_results: RwLock::new(HashMap::new()),
            unprocessed_pairs: Mutex::new(HashSet::new()),
            stats: Mutex::new(CacheStats::default()),
        }
    }
    /// Prune old entries from the cache when it grows too large
    #[allow(dead_code)]
    fn prune_cache(&self, max_size: usize) {
        let mut cache = self.cache.write().unwrap();
        if cache.len() > max_size {
            // Remove oldest entries first (would typically use an LRU strategy)
            // This simplified version just keeps the most recent entries
            let mut entries: Vec<_> = cache.drain().collect();
            entries.sort_by(|(_, a), (_, b)| {
                // Sort by most recently added/updated
                a.processed.cmp(&b.processed).reverse()
            });
            
            // Keep only the max_size most recent entries
            for (k, v) in entries.into_iter().take(max_size) {
                cache.insert(k, v);
            }
        }
    }
    // Completely redone add_read method to fix all ownership issues
    pub fn add_read(&self, read: &Read) -> bool {
        // Extract the base ID without /1 or /2 suffix
        let (base_id, read_number) = parse_read_id(&read.id);
        
        // Skip if we can't determine read number
        if read_number == 0 {
            return false;
        }
        
        let mut cache = self.cache.write().unwrap();
        let mut stats = self.stats.lock().unwrap();
        let mut unprocessed = self.unprocessed_pairs.lock().unwrap();
        
        // First, handle the case of finding a pair
        let mut found_pair = false;
        
        if read_number == 1 {
            // Looking for read2
            let pair_id = format!("{}/2", base_id);
            
            if let Some(pair_info) = cache.get(&pair_id) {
                // Found read2
                stats.pair_found += 1;
                
                // Create the complete paired read info
                let complete_pair = PairedReadInfo {
                    read1_id: read.id.clone(),
                    read2_id: pair_id.clone(),
                    read1_seq: read.seq.clone(),
                    read2_seq: pair_info.read1_seq.clone(),
                    read1_qual: read.quality.clone(),
                    read2_qual: pair_info.read1_qual.clone(),
                    processed: false,
                };
                
                // Save it and mark for processing
                cache.insert(base_id.clone(), complete_pair);
                unprocessed.insert(base_id);
                found_pair = true;
            } else {
                // No pair found yet, store self for future pairing
                stats.pair_not_found += 1;
                
                let self_info = PairedReadInfo {
                    read1_id: read.id.clone(),
                    read2_id: String::new(),
                    read1_seq: read.seq.clone(),
                    read2_seq: String::new(),
                    read1_qual: read.quality.clone(),
                    read2_qual: None,
                    processed: false,
                };
                
                cache.insert(format!("{}/1", base_id), self_info);
            }
        } else {
            // This is read2, look for read1
            let pair_id = format!("{}/1", base_id);
            
            if let Some(pair_info) = cache.get(&pair_id) {
                // Found read1
                stats.pair_found += 1;
                
                // Create the complete paired read info
                let complete_pair = PairedReadInfo {
                    read1_id: pair_info.read1_id.clone(),
                    read2_id: read.id.clone(),
                    read1_seq: pair_info.read1_seq.clone(),
                    read2_seq: read.seq.clone(),
                    read1_qual: pair_info.read1_qual.clone(),
                    read2_qual: read.quality.clone(),
                    processed: false,
                };
                
                // Save it and mark for processing
                cache.insert(base_id.clone(), complete_pair);
                unprocessed.insert(base_id);
                found_pair = true;
            } else {
                // No pair found yet, store self for future pairing
                stats.pair_not_found += 1;
                
                let self_info = PairedReadInfo {
                    read1_id: read.id.clone(),
                    read2_id: String::new(),
                    read1_seq: read.seq.clone(),
                    read2_seq: String::new(),
                    read1_qual: read.quality.clone(),
                    read2_qual: None,
                    processed: false,
                };
                
                cache.insert(format!("{}/2", base_id), self_info);
            }
        }
        
        stats.inserts += 1;
        found_pair
    }
    
    /// Process all unprocessed read pairs in the cache
    pub fn process_all_pairs(&self) {
        let unprocessed_ids: Vec<String> = {
            let mut unprocessed = self.unprocessed_pairs.lock().unwrap();
            let ids = unprocessed.iter().cloned().collect();
            unprocessed.clear();
            ids
        };
        
        // Process in parallel using Rayon
        unprocessed_ids.par_iter().for_each(|id| {
            self.process_read_pair(id);
        });
    }
    
    /// Process a specific read pair
    fn process_read_pair(&self, base_id: &str) {
        // First check if we already have analyzed this pair
        {
            let results = self.overlap_results.read().unwrap();
            if results.contains_key(base_id) {
                return;
            }
        }
        
        // Get the paired read info
        let pair_info = {
            let cache = self.cache.read().unwrap();
            match cache.get(base_id) {
                Some(info) => info.clone(),
                None => return,
            }
        };
        
        // Skip if not a complete pair
        if pair_info.read2_seq.is_empty() {
            return;
        }
        
        // Analyze the overlap between the paired reads
        let results = analyze_read_pair_overlap(&pair_info);
        
        // Store the results
        {
            let mut results_cache = self.overlap_results.write().unwrap();
            results_cache.insert(base_id.to_string(), results);
        }
        
        // Update stats
        {
            let mut stats = self.stats.lock().unwrap();
            stats.successful_alignments += 1;
        }
    }
    
    /// Get the overlap analysis results for a specific read
    pub fn get_overlap_results(&self, read_id: &str) -> Option<ReadPairResults> {
        let mut stats = self.stats.lock().unwrap();
        stats.lookups += 1;
        
        // Extract the base ID
        let (base_id, _) = parse_read_id(read_id);
        
        // Check cache first
        {
            let results = self.overlap_results.read().unwrap();
            if let Some(cached_results) = results.get(&base_id) {
                stats.hits += 1;
                return Some((*cached_results).clone());
            }
        }
        
        // Check if we have the pair but haven't processed it yet
        {
            let cache = self.cache.read().unwrap();
            if let Some(_info) = cache.get(&base_id) {
                // Process this pair specifically
                drop(cache); // Release lock before processing
                self.process_read_pair(&base_id);
                
                // Now try again to get the results
                let results = self.overlap_results.read().unwrap();
                return results.get(&base_id).cloned();
            }
        }
        
        None
    }
    
    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        self.stats.lock().unwrap().clone()
    }
}

/// Parse a read ID to extract the base ID and read number
/// Returns (base_id, read_number) where read_number is 1, 2, or 0 if unknown
fn parse_read_id(read_id: &str) -> (String, u8) {
    // Common formats:
    // - SEQID/1 and SEQID/2
    // - SEQID.1 and SEQID.2
    // - SEQID_1 and SEQID_2
    // - SEQID-1 and SEQID-2
    // - SEQID:1 and SEQID:2
    // - SEQID R1 and SEQID R2
    // - SEQID 1 and SEQID 2 (with space)
    // - SEQID.pair.1 and SEQID.pair.2
    // - SEQID#1 and SEQID#2
    
    if read_id.ends_with("/1") {
        (read_id.replace("/1", ""), 1)
    } else if read_id.ends_with("/2") {
        (read_id.replace("/2", ""), 2)
    } else if read_id.ends_with(".1") {
        (read_id.replace(".1", ""), 1)
    } else if read_id.ends_with(".2") {
        (read_id.replace(".2", ""), 2)
    } else if read_id.ends_with("_1") {
        (read_id.replace("_1", ""), 1)
    } else if read_id.ends_with("_2") {
        (read_id.replace("_2", ""), 2)
    } else if read_id.ends_with("-1") {
        (read_id.replace("-1", ""), 1)
    } else if read_id.ends_with("-2") {
        (read_id.replace("-2", ""), 2)
    } else if read_id.ends_with(":1") {
        (read_id.replace(":1", ""), 1)
    } else if read_id.ends_with(":2") {
        (read_id.replace(":2", ""), 2)
    } else if read_id.ends_with(" R1") {
        (read_id.replace(" R1", ""), 1)
    } else if read_id.ends_with(" R2") {
        (read_id.replace(" R2", ""), 2)
    } else if read_id.ends_with(" 1") {
        (read_id.replace(" 1", ""), 1)
    } else if read_id.ends_with(" 2") {
        (read_id.replace(" 2", ""), 2)
    } else if read_id.ends_with(".pair.1") {
        (read_id.replace(".pair.1", ""), 1)
    } else if read_id.ends_with(".pair.2") {
        (read_id.replace(".pair.2", ""), 2)
    } else if read_id.ends_with("#1") {
        (read_id.replace("#1", ""), 1)
    } else if read_id.ends_with("#2") {
        (read_id.replace("#2", ""), 2)
    } else if read_id.contains("_R1_") {
        // Handle formats like SEQID_R1_001
        (read_id.replace("_R1_", "_"), 1)
    } else if read_id.contains("_R2_") {
        (read_id.replace("_R2_", "_"), 2)
    } else if read_id.contains("_1_") && !read_id.contains("_10_") && !read_id.contains("_11_") {
        // Careful with _1_ vs _10_, _11_, etc.
        (read_id.replace("_1_", "_"), 1)
    } else if read_id.contains("_2_") && !read_id.contains("_20_") && !read_id.contains("_21_") {
        (read_id.replace("_2_", "_"), 2)
    } else {
        // Can't determine read number - let's check the sequence
        // and see if we can guess based on length or content
        (read_id.to_string(), 0)
    }
}

/// Analyze a read using the read pair overlap approach
/// This is the main entry point for read pair analysis from other parts of the pipeline
pub fn analyze_read_pair_overlaps(read: &Read) -> ReadPairResults {
    // Access the global cache
    lazy_static! {
        static ref GLOBAL_CACHE: Arc<ReadPairCache> = Arc::new(ReadPairCache::new());
    }
    
    // First, add this read to the cache
    let paired = GLOBAL_CACHE.add_read(read);
    
    // If this completed a pair, process it
    if paired {
        GLOBAL_CACHE.process_all_pairs();
    }
    
    // Try to get results for this read
    if let Some(results) = GLOBAL_CACHE.get_overlap_results(&read.id) {
        return results;
    }
    
    // If no results found, return default
    ReadPairResults {
        has_overlap: false,
        overlap_size: 0,
        c_to_t_mismatches: 0,
        g_to_a_mismatches: 0,
        other_mismatches: 0,
        pair_support_score: 0.5, // Neutral score
    }
}

/// This is the internal function that analyzes a pair of reads
/// It is called by the ReadPairCache when processing pairs
fn analyze_read_pair_overlap(pair_info: &PairedReadInfo) -> ReadPairResults {
    // First check if both reads exist
    if pair_info.read1_seq.is_empty() || pair_info.read2_seq.is_empty() {
        return ReadPairResults {
            has_overlap: false,
            overlap_size: 0,
            c_to_t_mismatches: 0,
            g_to_a_mismatches: 0,
            other_mismatches: 0,
            pair_support_score: 0.0,
        };
    }
    
    // Reverse complement read2 for alignment with read1
    let read2_rc = reverse_complement(&pair_info.read2_seq);
    
    // Perform semi-global alignment to find the overlap
    let alignment = align_reads(&pair_info.read1_seq, &read2_rc);
    
    // Extract the overlapping region
    let (overlap_region, overlap_size) = extract_overlap_region(&alignment);
    
    // If no significant overlap found, return empty results
    if overlap_size < 10 {
        return ReadPairResults {
            has_overlap: false,
            overlap_size: 0,
            c_to_t_mismatches: 0,
            g_to_a_mismatches: 0,
            other_mismatches: 0,
            pair_support_score: 0.0,
        };
    }
    
    // Analyze mismatches in the overlap
    let mismatches = analyze_mismatches(
        &pair_info.read1_seq, 
        &read2_rc, 
        &overlap_region, 
        &pair_info.read1_qual, 
        &pair_info.read2_qual
    );
    
    // Count damage-specific mismatches
    let (c_to_t, g_to_a, other) = count_damage_mismatches(&mismatches);
    
    // Calculate support score based on damage pattern
    let support_score = calculate_damage_support_score(&mismatches, overlap_size);
    
    // Return the results
    ReadPairResults {
        has_overlap: true,
        overlap_size,
        c_to_t_mismatches: c_to_t,
        g_to_a_mismatches: g_to_a,
        other_mismatches: other,
        pair_support_score: support_score,
    }
}

/// Reverse complement a DNA sequence
fn reverse_complement(seq: &str) -> String {
    seq.chars()
        .rev()
        .map(|base| match base {
            'A' => 'T',
            'T' => 'A',
            'G' => 'C',
            'C' => 'G',
            'a' => 't',
            't' => 'a',
            'g' => 'c',
            'c' => 'g',
            'N' => 'N',
            'n' => 'n',
            _ => 'N',
        })
        .collect()
}

/// Align two reads to find their overlap
fn align_reads(read1: &str, read2: &str) -> Alignment {
    // Configure the aligner for DNA sequences
    // Using semi-global alignment to allow overhangs
    let score = |a: u8, b: u8| {
        if a == b {
            1i32  // Match
        } else {
            -1i32 // Mismatch
        }
    };
    
    let mut aligner = Aligner::new(-5, -1, score);
    
    // Convert strings to byte slices for the aligner
    let read1_bytes = read1.as_bytes();
    let read2_bytes = read2.as_bytes();
    
    // Perform the alignment
    aligner.semiglobal(read1_bytes, read2_bytes)
}

/// Extract the overlap region from an alignment
fn extract_overlap_region(alignment: &Alignment) -> ((usize, usize), usize) {
    let read1_start = alignment.xstart;
    let read1_end = alignment.xend;
    let overlap_size = read1_end - read1_start;
    
    ((read1_start, read1_end), overlap_size)
}

/// Analyze mismatches in the overlapping region
fn analyze_mismatches(
    read1: &str,
    read2: &str,
    overlap: &(usize, usize),
    qual1: &Option<Vec<u8>>,
    qual2: &Option<Vec<u8>>
) -> Vec<MismatchInfo> {
    let (start, end) = *overlap;
    let mut mismatches = Vec::new();
    
    // Check if reads are long enough
    if start >= read1.len() || end > read1.len() || start >= read2.len() || end > read2.len() {
        return mismatches;
    }
    
    // Extract the overlapping regions
    let read1_overlap = &read1[start..end];
    let read2_overlap = &read2[start..end];
    
    let read1_chars: Vec<char> = read1_overlap.chars().collect();
    let read2_chars: Vec<char> = read2_overlap.chars().collect();
    
    for i in 0..read1_chars.len() {
        if i >= read2_chars.len() {
            break;
        }
        
        let base1 = read1_chars[i];
        let base2 = read2_chars[i];
        
        // Skip non-standard bases
        if !is_standard_base(base1) || !is_standard_base(base2) {
            continue;
        }
        
        // Get quality scores if available
        let qual1_val = if let Some(ref quality) = qual1 {
            if start + i < quality.len() {
                Some(quality[start + i])
            } else {
                None
            }
        } else {
            None
        };
        
        let qual2_val = if let Some(ref quality) = qual2 {
            if start + i < quality.len() {
                Some(quality[start + i])
            } else {
                None
            }
        } else {
            None
        };
        
        // Check if bases match
        if base1.to_ascii_uppercase() != base2.to_ascii_uppercase() {
            // Determine if this is a damage pattern
            let (is_damage, pattern_type) = classify_damage_pattern(base1, base2, start + i, read1.len());
            
            mismatches.push(MismatchInfo {
                position: i,
                read1_pos: start + i,
                read2_pos: start + i,
                read1_base: base1,
                read2_base: base2,
                read1_qual: qual1_val,
                read2_qual: qual2_val,
                is_damage_pattern: is_damage,
                pattern_type,
            });
        }
    }
    
    mismatches
}

/// Check if a character is a standard DNA base
fn is_standard_base(base: char) -> bool {
    matches!(base.to_ascii_uppercase(), 'A' | 'C' | 'G' | 'T')
}

/// Classify whether a mismatch is a typical ancient DNA damage pattern
fn classify_damage_pattern(base1: char, base2: char, position: usize, read_length: usize) -> (bool, DamagePatternType) {
    let base1_upper = base1.to_ascii_uppercase();
    let base2_upper = base2.to_ascii_uppercase();
    
    // Determine if we're near the 5' end (start) or 3' end
    // Use a dynamic threshold based on read length
    let terminal_threshold = std::cmp::min(10, read_length / 5);
    
    let near_5_prime = position < terminal_threshold;
    let near_3_prime = position >= read_length.saturating_sub(terminal_threshold);
    
    // C→T at 5' end of read1
    if near_5_prime && base1_upper == 'T' && base2_upper == 'C' {
        return (true, DamagePatternType::CTTransition5Prime);
    }
    
    // G→A at 3' end of read1
    if near_3_prime && base1_upper == 'A' && base2_upper == 'G' {
        return (true, DamagePatternType::GATransition3Prime);
    }
    
    // C→T at 3' end of read2 (appears as G→A in read1)
    if near_3_prime && base1_upper == 'G' && base2_upper == 'A' {
        return (true, DamagePatternType::CTTransition3Prime);
    }
    
    // G→A at 5' end of read2 (appears as C→T in read1)
    if near_5_prime && base1_upper == 'C' && base2_upper == 'T' {
        return (true, DamagePatternType::GATransition5Prime);
    }
    
    // Any other mismatch
    (false, DamagePatternType::OtherMismatch)
}

/// Count the different types of damage-related mismatches
fn count_damage_mismatches(mismatches: &[MismatchInfo]) -> (usize, usize, usize) {
    let mut c_to_t = 0;
    let mut g_to_a = 0;
    let mut other = 0;
    
    for mismatch in mismatches {
        match mismatch.pattern_type {
            DamagePatternType::CTTransition5Prime | DamagePatternType::GATransition5Prime => c_to_t += 1,
            DamagePatternType::GATransition3Prime | DamagePatternType::CTTransition3Prime => g_to_a += 1,
            DamagePatternType::OtherMismatch => other += 1,
        }
    }
    
    (c_to_t, g_to_a, other)
}

/// Calculate a support score based on damage patterns
fn calculate_damage_support_score(mismatches: &[MismatchInfo], overlap_size: usize) -> f64 {
    if mismatches.is_empty() || overlap_size < 10 {
        return 0.5; // Neutral score for no data
    }
    
    // Count by pattern type
    let mut c_to_t_5p = 0;
    let mut g_to_a_3p = 0;
    let mut other_mm = 0;
    
    for mismatch in mismatches {
        match mismatch.pattern_type {
            DamagePatternType::CTTransition5Prime => c_to_t_5p += 1,
            DamagePatternType::GATransition3Prime => g_to_a_3p += 1,
            DamagePatternType::OtherMismatch => other_mm += 1,
            _ => {}
        }
    }
    
    let total_mismatches = c_to_t_5p + g_to_a_3p + other_mm;
    if total_mismatches == 0 {
        return 0.5; // Neutral score
    }
    
    // Calculate the damage ratio (proportion of mismatches that match aDNA pattern)
    let damage_ratio = (c_to_t_5p + g_to_a_3p) as f64 / total_mismatches as f64;
    
    // Calculate position-weighted score (damage should be concentrated at ends)
    let mut position_score = 0.0;
    let mut total_weight = 0.0;
    
    for mismatch in mismatches {
        // Only consider damage pattern mismatches
        if !mismatch.is_damage_pattern {
            continue;
        }
        
        // Apply quality score weighting
        let quality_weight = match (mismatch.read1_qual, mismatch.read2_qual) {
            (Some(q1), Some(q2)) if q1 > 20 && q2 > 20 => 1.0,  // High quality mismatch
            (Some(q1), Some(q2)) if q1 > 15 && q2 > 15 => 0.8,  // Medium quality
            (Some(_), Some(_)) => 0.5,                         // Lower quality
            _ => 0.7,                                          // No quality scores
        };
        
        // Determine position relative to appropriate end
        let end_distance = match mismatch.pattern_type {
            DamagePatternType::CTTransition5Prime | DamagePatternType::GATransition5Prime => 
                mismatch.read1_pos,
            DamagePatternType::GATransition3Prime | DamagePatternType::CTTransition3Prime => 
                overlap_size - mismatch.position,
            _ => overlap_size / 2, // Middle position for non-specific
        };
        
        // Calculate weight (higher for positions closer to ends)
        let position_weight = (10.0 / (end_distance as f64 + 1.0)).min(1.0);
        position_score += position_weight * quality_weight;  // Apply quality weighting
        total_weight += quality_weight;
    }
    
    // Normalize position score
    let end_concentration = if total_weight > 0.0 { 
        position_score / total_weight
    } else { 
        0.5 
    };
    
    // Combine damage ratio and position concentration
    let support_score = 0.7 * damage_ratio + 0.3 * end_concentration;
    
    // Normalize to 0.0-1.0 range
    support_score.max(0.0).min(1.0)
}
/// Combine evidence from multiple methods and return final scores
fn combine_damage_evidence(
    change_point: &ChangePointResults,
    decay_model: &DecayModelResults,
    read_pair: &ReadPairResults,
    read: &Read
) -> CombinedDamageResults {
    // Adjust weights based on fragment length
    let fragment_length = read.seq.len();
    
    // For very short fragments, prioritize change point detection
    let cp_base_weight = if fragment_length < 50 {
        0.5  // Short fragments: rely more on change point detection
    } else if fragment_length < 100 {
        0.4  // Medium fragments
    } else {
        0.3  // Longer fragments
    };
    
    // Weight change point evidence by statistical significance
    let cp_weight = if change_point.prime5_pvalue < 0.05 || change_point.prime3_pvalue < 0.05 {
        cp_base_weight  // Higher weight if change point is statistically significant
    } else {
        cp_base_weight * 0.5  // Lower weight if not significant
    };
    
    // Weight decay model by quality of fit and log-likelihood ratio
    let model_weight = if decay_model.log_likelihood_ratio > 2.0 && 
                        decay_model.prime5_model_fit > 0.6 && 
                        decay_model.prime3_model_fit > 0.4 {
        0.5  // Strong evidence from decay model
    } else if decay_model.log_likelihood_ratio > 0.5 {
        0.4  // Moderate evidence
    } else {
        0.3  // Weak evidence
    };
    
    // Weight read pair evidence by quality of overlap and damage pattern
    let pair_weight = if read_pair.has_overlap && read_pair.overlap_size > 20 {
        // Calculate the ratio of damage-specific mismatches to total mismatches
        let total_mismatches = read_pair.c_to_t_mismatches + read_pair.g_to_a_mismatches + read_pair.other_mismatches;
        let damage_ratio = if total_mismatches > 0 {
            (read_pair.c_to_t_mismatches + read_pair.g_to_a_mismatches) as f64 / total_mismatches as f64
        } else {
            0.0
        };
        
        if damage_ratio > 0.7 && total_mismatches > 3 {
            0.5  // Strong damage signal in overlap
        } else if damage_ratio > 0.5 {
            0.4  // Moderate damage signal
        } else if read_pair.has_overlap {
            0.2  // Weak signal but has overlap
        } else {
            0.0  // No relevant signal
        }
    } else {
        0.0  // No overlap or too small
    };
    
    // Normalize weights to sum to 1.0
    let total_weight = cp_weight + model_weight + pair_weight;
    let cp_norm = cp_weight / total_weight;
    let model_norm = model_weight / total_weight;
    let pair_norm = pair_weight / total_weight;
    
    // Compute weighted scores for each end
    let prime5_score = 
        cp_norm * change_point.prime5_score +
        model_norm * normalize_score(decay_model.prime5_model_fit, 0.0, 1.0, 0.1, 0.9) +
        pair_norm * read_pair.pair_support_score;
    
    let prime3_score = 
        cp_norm * change_point.prime3_score +
        model_norm * normalize_score(decay_model.prime3_model_fit, 0.0, 1.0, 0.1, 0.9) +
        pair_norm * read_pair.pair_support_score;
    
    // Calculate authenticity factors
    let model_support = if decay_model.log_likelihood_ratio > 0.0 {
        normalize_score(decay_model.log_likelihood_ratio, 0.0, 10.0, 0.0, 1.0)
    } else {
        0.0
    };
    
    let cp_support = if change_point.prime5_pvalue < 0.10 || change_point.prime3_pvalue < 0.10 {
        1.0 - (change_point.prime5_pvalue.min(change_point.prime3_pvalue) * 10.0).min(1.0)
    } else {
        0.0
    };
    
    // Consider fragment length (shorter fragments more likely ancient)
    let length_factor = fragment_length_factor(read.seq.len());
    
    // Consider balance between 5' and 3' damage (authentic aDNA typically has both)
    let balance_factor = if prime5_score > 0.2 && prime3_score > 0.2 {
        // Calculate how balanced the damage is between ends
        1.0 - (prime5_score - prime3_score).abs() / (prime5_score + prime3_score).max(0.1)
    } else {
        0.5  // Neutral factor when damage scores are low
    };
    
    // Calculate final authenticity score with multiple factors
    let authenticity = 
        0.35 * ((prime5_score + prime3_score) / 2.0) +
        0.25 * ((model_support + cp_support) / 2.0) +
        0.25 * length_factor +
        0.15 * balance_factor;
    
    CombinedDamageResults {
        prime5_score,
        prime3_score,
        authenticity_score: authenticity,
        method_weights: [cp_norm, model_norm, pair_norm],
    }
}

/// Helper function to normalize scores to a specific range
fn normalize_score(value: f64, min_in: f64, max_in: f64, min_out: f64, max_out: f64) -> f64 {
    if max_in == min_in {
        return min_out;
    }
    let normalized = (value - min_in) / (max_in - min_in);
    min_out + normalized * (max_out - min_out)
}

/// Calculate a factor based on fragment length
fn fragment_length_factor(length: usize) -> f64 {
    match length {
        0..=50 => 0.9,    // Very short: likely ancient
        51..=100 => 0.7,  // Short: may be ancient
        101..=150 => 0.5, // Medium: uncertain
        151..=200 => 0.3, // Long: less likely ancient
        _ => 0.1,         // Very long: unlikely ancient
    }
}

/// Output structure for the change point method
#[derive(Debug)]
struct ChangePointResult {
    position: Option<usize>,
    pvalue: f64,
}

#[derive(Debug)]
struct ChangePointResults {
    prime5_pvalue: f64,
    prime3_pvalue: f64,
    prime5_score: f64,
    prime3_score: f64,
}

/// Output structure for the exponential decay model
#[derive(Debug)]
struct ExponentialModel {
    lambda: f64,      // Decay rate parameter
    amplitude: f64,   // Initial amplitude
    fit_quality: f64, // How well the model fits (0-1)
}

#[derive(Debug)]
struct DecayModelResults {
    prime5_lambda: f64,
    prime3_lambda: f64,
    prime5_model_fit: f64,
    prime3_model_fit: f64,
    log_likelihood_ratio: f64,
}

/// Output structure for read pair analysis
#[derive(Debug, Clone)]
pub struct ReadPairResults {
    has_overlap: bool,
    overlap_size: usize,
    c_to_t_mismatches: usize,
    g_to_a_mismatches: usize,
    other_mismatches: usize,
    pair_support_score: f64,
}

/// Combined results from all methods
#[derive(Debug)]
struct CombinedDamageResults {
    prime5_score: f64,
    prime3_score: f64,
    authenticity_score: f64,
    #[allow(dead_code)]
    method_weights: [f64; 3], // Weights used for change-point, decay model, and read pair
}

/// Generate all possible k-mers of a given length
fn get_all_kmers(k: usize) -> Vec<String> {
    if k == 0 {
        return vec![String::new()];
    }
    
    let bases = vec!['A', 'C', 'G', 'T'];
    let mut kmers = Vec::new();
    
    // Generate all k-mers recursively
    let prev_kmers = get_all_kmers(k - 1);
    for kmer in prev_kmers {
        for &base in &bases {
            let mut new_kmer = kmer.clone();
            new_kmer.push(base);
            kmers.push(new_kmer);
        }
    }
    
    kmers
}

/// Compute k-mer frequencies for a sequence
fn compute_kmer_freq(seq: &str, k: usize, kmers_list: &[String]) -> HashMap<String, f64> {
    let mut freq = HashMap::new();
    
    // Initialize all k-mers with zero frequency
    for kmer in kmers_list {
        freq.insert(kmer.clone(), 0.0);
    }
    
    if seq.len() < k {
        return freq;
    }
    
    // Count k-mers in the sequence
    let total_kmers = seq.len() - k + 1;
    for i in 0..total_kmers {
        let kmer = &seq[i..i+k];
        *freq.entry(kmer.to_string()).or_insert(0.0) += 1.0;
    }
    
    // Normalize frequencies
    for value in freq.values_mut() {
        *value /= total_kmers as f64;
    }
    
    freq
}

/// Compute codon usage frequencies
fn compute_codon_usage(seq: &str) -> HashMap<String, f64> {
    // Ensure length is divisible by 3
    let truncated_len = seq.len() - (seq.len() % 3);
    
    if truncated_len < 3 {
        // Return empty map for sequences shorter than a codon
        return HashMap::new();
    }
    
    let mut codon_counts = HashMap::new();
    
    // Count codons
    for i in (0..truncated_len).step_by(3) {
        if i + 3 <= seq.len() {
            let codon = &seq[i..i+3];
            *codon_counts.entry(codon.to_string()).or_insert(0.0) += 1.0;
        }
    }
    
    // Normalize
    let total_codons = truncated_len / 3;
    for count in codon_counts.values_mut() {
        *count /= total_codons as f64;
    }
    
    codon_counts
}

/// Incremental PCA implementation
fn incremental_pca(
    reads: &[Read],
    batch_size: usize,
    feature_extractor: impl Fn(&Read) -> Vec<f64> + Sync,
    total_features: usize,
    n_components: usize
) -> (Array1<f64>, Array2<f64>, Vec<f64>, Array2<f64>) {
    use ndarray::{Array1, Array2};
    use ndarray_linalg::Eig;
    use indicatif::{ProgressBar, ProgressStyle};
    use std::sync::{Arc, Mutex};
    use rayon::prelude::*;
    
    // First, extract features for all reads and filter out any problematic ones
    info!("Extracting features for PCA...");
    let pb = ProgressBar::new(reads.len() as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} reads processed (Feature extraction)"
        )
        .unwrap()
        .progress_chars("##-")
    );
    
    // Store both the feature vectors and associated read indices
    let feature_vectors = Arc::new(Mutex::new(Vec::new()));
    let chunk_size = std::cmp::max(1, reads.len() / rayon::current_num_threads());
    
    reads.par_chunks(chunk_size).for_each(|chunk| {
        let mut local_features = Vec::with_capacity(chunk.len());
        
        for read in chunk {
            // Try to extract features, skipping reads that cause problems
            let features = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| feature_extractor(read))) {
                Ok(features) => features,
                Err(_) => {
                    // Skip this read if feature extraction fails
                    continue;
                }
            };
            
            // Verify the feature vector is the correct length
            if features.len() == total_features {
                local_features.push((read.id.clone(), features));
            }
        }
        
        // Update global features list
        let mut all_features = feature_vectors.lock().unwrap();
        all_features.extend(local_features);
        pb.inc(chunk.len() as u64);
    });
    
    pb.finish_with_message("Feature extraction complete");
    
    // Get the feature vectors and prepare for PCA
    let all_feature_vectors = Arc::try_unwrap(feature_vectors)
        .expect("Failed to unwrap feature vectors Arc")
        .into_inner()
        .expect("Failed to unwrap feature vectors Mutex");
    
    let n_samples = all_feature_vectors.len();
    info!("Successfully extracted features for {} reads", n_samples);
    
    if n_samples == 0 {
        error!("No valid feature vectors extracted. Cannot perform PCA.");
        return (
            Array1::<f64>::zeros(total_features),
            Array2::<f64>::zeros((total_features, 1)),
            vec![0.0],
            Array2::<f64>::zeros((1, 1))
        );
    }
    
    // Initial parameters for PCA
    let mut mean = Array1::<f64>::zeros(total_features);
    let mut components = Array2::<f64>::zeros((total_features, total_features));
    let mut explained_variance = Vec::with_capacity(n_components);
    let mut n_samples_seen = 0;
    
    // Progress bar for the PCA
    let pb = ProgressBar::new(n_samples as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} reads processed (PCA)"
        )
        .unwrap()
        .progress_chars("##-")
    );
    
    // Convert feature vectors to a data matrix
    let mut data_matrix = Array2::<f64>::zeros((n_samples, total_features));
    let mut read_ids = Vec::with_capacity(n_samples);
    
    for (i, (id, features)) in all_feature_vectors.iter().enumerate() {
        read_ids.push(id.clone());
        for (j, &val) in features.iter().enumerate() {
            data_matrix[[i, j]] = val;
        }
    }
    
    // Process in batches
    for batch_start in (0..n_samples).step_by(batch_size) {
        let batch_end = std::cmp::min(batch_start + batch_size, n_samples);
        let batch_size = batch_end - batch_start;
        
        // Extract this batch of data
        let batch_data = data_matrix.slice(ndarray::s![batch_start..batch_end, ..]);
        
        // Incremental fit
        if n_samples_seen == 0 {
            // First batch - initialize
            for i in 0..batch_size {
                mean += &batch_data.row(i);
            }
            mean /= batch_size as f64;
            
            // Center data
            let mut centered = batch_data.to_owned();
            for i in 0..batch_size {
                for j in 0..total_features {
                    centered[[i, j]] -= mean[j];
                }
            }
            
            // Compute covariance
            let cov = centered.t().dot(&centered) / (batch_size as f64 - 1.0);
            
            // Eigendecomposition
            let eig = match cov.eig() {
                Ok(eig) => eig,
                Err(e) => {
                    error!("Eigendecomposition failed: {:?}", e);
                    return (
                        mean, 
                        Array2::<f64>::zeros((total_features, n_components)),
                        vec![0.0; n_components],
                        Array2::<f64>::zeros((n_samples, n_components))
                    );
                }
            };
            
            let (eigenvalues, eigenvectors) = eig;
            
            // Convert complex eigenvectors to real values
            for i in 0..total_features {
                for j in 0..total_features {
                    components[[i, j]] = eigenvectors[[i, j]].re;
                }
            }
            
            // Extract real part of eigenvalues
            explained_variance = eigenvalues.iter()
                .map(|&c| c.re)
                .collect();
        } else {
            // Subsequent batches - update mean and components
            let old_mean = mean.clone();
            let old_sample_count = n_samples_seen as f64;
            let new_sample_count = batch_size as f64;
            let total_samples = old_sample_count + new_sample_count;
            
            // Update mean
            let batch_mean = batch_data.sum_axis(ndarray::Axis(0)) / new_sample_count;
            mean = (old_mean * old_sample_count + batch_mean * new_sample_count) / total_samples;
            
            // Center data
            let mut centered = batch_data.to_owned();
            for i in 0..batch_size {
                for j in 0..total_features {
                    centered[[i, j]] -= mean[j];
                }
            }
            
            // Update covariance
            let batch_cov = centered.t().dot(&centered) / (new_sample_count - 1.0);
            let old_cov = components.dot(&components.t());
            let new_cov = (old_cov * old_sample_count + batch_cov * new_sample_count) / total_samples;
            
            // Eigendecomposition on updated covariance
            let eig = match new_cov.eig() {
                Ok(eig) => eig,
                Err(e) => {
                    error!("Eigendecomposition failed: {:?}", e);
                    return (
                        mean, 
                        Array2::<f64>::zeros((total_features, n_components)),
                        vec![0.0; n_components],
                        Array2::<f64>::zeros((n_samples, n_components))
                    );
                }
            };
            
            let (eigenvalues, eigenvectors) = eig;
            
            // Convert complex eigenvectors to real values
            for i in 0..total_features {
                for j in 0..total_features {
                    components[[i, j]] = eigenvectors[[i, j]].re;
                }
            }
            
            // Extract real part of eigenvalues
            explained_variance = eigenvalues.iter()
                .map(|&c| c.re)
                .collect();
        }
        
        n_samples_seen += batch_size;
        pb.set_position(n_samples_seen as u64);
    }
    
    pb.finish_with_message("PCA complete");
    
    // Sort eigenvalues and eigenvectors in descending order
    let mut eigen_pairs: Vec<(f64, usize)> = explained_variance
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();
    
    eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    
    let sorted_indices: Vec<usize> = eigen_pairs.iter().map(|&(_, i)| i).collect();
    let mut sorted_components = Array2::<f64>::zeros((total_features, n_components));
    let mut sorted_variance = Vec::with_capacity(n_components);
    
    for (i, &idx) in sorted_indices.iter().take(n_components).enumerate() {
        if idx < components.ncols() {  // Ensure we're not accessing out of bounds
            sorted_components.column_mut(i).assign(&components.column(idx));
            sorted_variance.push(explained_variance[idx]);
        } else {
            // Fill with zeros if we somehow have an invalid index
            sorted_components.column_mut(i).fill(0.0);
            sorted_variance.push(0.0);
        }
    }
    
    // Transform the data
    let mut transformed = Array2::<f64>::zeros((n_samples, n_components));
    
    for (i, (_, features)) in all_feature_vectors.iter().enumerate() {
        let mut features_array = Array1::<f64>::zeros(total_features);
        for (j, &value) in features.iter().enumerate() {
            features_array[j] = value;
        }
        
        // Center the data
        features_array -= &mean;
        
        // Project using components
        for j in 0..n_components {
            transformed[[i, j]] = features_array.dot(&sorted_components.column(j));
        }
    }
    
    info!("PCA transformation complete: {} samples, {} components", n_samples, n_components);
    
    (mean, sorted_components, sorted_variance, transformed)
}

/// Function to execute DBSCAN clustering for different dimensions
fn dbscan_clustering<const D: usize>(
    data: &ndarray::Array2<f64>,
    eps: f64,
    min_samples: usize
) -> Vec<isize> where [(); D]: DimensionalDBSCAN {
    <[(); D]>::dbscan_clustering(data, eps, min_samples)
}

// Add this 1D clustering implementation
fn cluster_1d(data: &Array2<f64>, eps: f64, min_samples: usize) -> Vec<isize> {
    use std::collections::HashMap;
    
    let n_points = data.nrows();
    let mut clusters = vec![-1isize; n_points];
    
    // Extract the 1D values and their indices
    let mut values_with_indices: Vec<(f64, usize)> = Vec::with_capacity(n_points);
    for i in 0..n_points {
        values_with_indices.push((data[[i, 0]], i));
    }
    
    // Sort by value for efficient neighbor finding
    values_with_indices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    
    // Create a mapping from original index to sorted index
    let mut index_map = HashMap::with_capacity(n_points);
    for (sorted_idx, &(_, original_idx)) in values_with_indices.iter().enumerate() {
        index_map.insert(original_idx, sorted_idx);
    }
    
    // Main clustering algorithm
    let mut cluster_id = 0;
    
    for i in 0..n_points {
        if clusters[i] != -1 {
            continue; // Skip points already assigned to clusters
        }
        
        // Find neighbors efficiently using binary search in the sorted array
        let sorted_idx = *index_map.get(&i).unwrap();
        let value = values_with_indices[sorted_idx].0;
        
        // Find neighbors using a linear scan in the sorted array 
        // (more efficient than checking every point in 1D)
        let mut neighbors = Vec::new();
        
        // Search left
        let mut left_idx = sorted_idx;
        while left_idx > 0 {
            left_idx -= 1;
            let neighbor_value = values_with_indices[left_idx].0;
            if (value - neighbor_value).abs() <= eps {
                neighbors.push(values_with_indices[left_idx].1);
            } else {
                break; // No more neighbors in this direction
            }
        }
        
        // Search right
        let mut right_idx = sorted_idx;
        while right_idx < n_points - 1 {
            right_idx += 1;
            let neighbor_value = values_with_indices[right_idx].0;
            if (value - neighbor_value).abs() <= eps {
                neighbors.push(values_with_indices[right_idx].1);
            } else {
                break; // No more neighbors in this direction
            }
        }
        
        // Include the point itself
        neighbors.push(i);
        
        // Check if it's a core point
        if neighbors.len() < min_samples {
            continue; // Not a core point
        }
        
        // Create a new cluster
        cluster_id += 1;
        
        // Process all neighbors
        let mut to_process = neighbors.clone();
        clusters[i] = cluster_id; // Mark the current point
        
        while let Some(current) = to_process.pop() {
            // Skip already processed points
            if clusters[current] != -1 {
                continue;
            }
            
            // Add current point to cluster
            clusters[current] = cluster_id;
            
            // Find neighbors of current point
            let current_sorted_idx = *index_map.get(&current).unwrap();
            let current_value = values_with_indices[current_sorted_idx].0;
            
            let mut current_neighbors = Vec::new();
            
            // Search left
            let mut left_idx = current_sorted_idx;
            while left_idx > 0 {
                left_idx -= 1;
                let neighbor_value = values_with_indices[left_idx].0;
                if (current_value - neighbor_value).abs() <= eps {
                    current_neighbors.push(values_with_indices[left_idx].1);
                } else {
                    break;
                }
            }
            
            // Search right
            let mut right_idx = current_sorted_idx;
            while right_idx < n_points - 1 {
                right_idx += 1;
                let neighbor_value = values_with_indices[right_idx].0;
                if (current_value - neighbor_value).abs() <= eps {
                    current_neighbors.push(values_with_indices[right_idx].1);
                } else {
                    break;
                }
            }
            
            // If it's a core point, add its neighbors to processing queue
            if current_neighbors.len() >= min_samples {
                for &neighbor in &current_neighbors {
                    if clusters[neighbor] == -1 {
                        to_process.push(neighbor);
                    }
                }
            }
        }
    }
    
    clusters
}

/// Compute fs for each cluster
fn compute_cluster_stats(
    read_ids: &[String],
    clusters: &[isize],
    gc_scores: &HashMap<String, f64>,
    damage_scores: &HashMap<String, DamageScore>
) -> HashMap<isize, ClusterStat> {
    use std::collections::HashMap;
    
    let mut stats = HashMap::new();
    let mut cluster_counts = HashMap::new();
    
    // Collect data for each cluster
    for (i, &cluster_id) in clusters.iter().enumerate() {
        let read_id = &read_ids[i];
        let gc = gc_scores.get(read_id).cloned().unwrap_or(0.0);
        let damage = damage_scores.get(read_id).cloned().unwrap_or(
            DamageScore {
                damage5: 0.0,
                damage3: 0.0,
                combined: 0.0,
            }
        );
        
        // Get or create cluster stat
        let entry = stats.entry(cluster_id).or_insert(ClusterStat {
            cluster: cluster_id,
            num_reads: 0,
            avg_gc: 0.0,
            avg_damage: 0.0,
            avg_damage5: 0.0,
            avg_damage3: 0.0,
            taxonomy: String::new(),
        });
        
        // Update totals
        entry.avg_gc += gc;
        entry.avg_damage += damage.combined;
        entry.avg_damage5 += damage.damage5;
        entry.avg_damage3 += damage.damage3;
        entry.num_reads += 1;
        
        *cluster_counts.entry(cluster_id).or_insert(0) += 1;
    }
    
    // Compute averages
    for (cluster_id, stat) in stats.iter_mut() {
        let count = cluster_counts.get(cluster_id).cloned().unwrap_or(1) as f64;
        stat.avg_gc /= count;
        stat.avg_damage /= count;
        stat.avg_damage5 /= count;
        stat.avg_damage3 /= count;
    }
    
    stats
}

/// Taxonomic assignment (simplified implementation)
fn assign_taxonomy(
    cluster_stats: &HashMap<isize, ClusterStat>,
    pca_results: &Array2<f64>, // to implement
    clusters: &[isize], // to implement
    k: usize, // to implement
    sim_threshold: f64, // to implement
    conf_threshold: f64, // to implement
    damage_threshold: f64,
    mean: &Array1<f64>,
    eigenvectors: &Array2<f64>, // to implement
) -> HashMap<isize, String> {
    // This is a placeholder implementation
    
    let mut taxonomy = HashMap::new();
    
    for (&cluster_id, stat) in cluster_stats {
        // Skip noise cluster
        if cluster_id == -1 {
            taxonomy.insert(cluster_id, "Noise".to_string());
            continue;
        }
        
        // Use damage scores to determine authenticity
        let authentic = stat.avg_damage >= damage_threshold;
        
        // For demonstration, assign simple taxonomy based on GC content ranges
        // later implementation will use reference matching, ML models, etc.
        let tax_name = if authentic {
            match stat.avg_gc {
                gc if gc < 0.35 => "Low GC (authentic)",
                gc if gc < 0.50 => "Medium GC (authentic)",
                _ => "High GC (authentic)",
            }
        } else {
            match stat.avg_gc {
                gc if gc < 0.35 => "Low GC (modern)",
                gc if gc < 0.50 => "Medium GC (modern)",
                _ => "High GC (modern)",
            }
        };
        
        taxonomy.insert(cluster_id, tax_name.to_string());
    }
    
    taxonomy
}

/// Generate report and cluster statistics
fn generate_report(
    cluster_stats: &HashMap<isize, ClusterStat>,
    output_path: &str
) -> Result<(), Box<dyn std::error::Error>> {
    use csv::Writer;
    use std::fs::File;
    
    let mut writer = Writer::from_writer(File::create(output_path)?);
    
    // Write header
    writer.write_record(&[
        "Cluster ID",
        "Number of Reads",
        "Avg GC Content",
        "Avg Damage Score",
        "Avg 5' Damage",
        "Avg 3' Damage",
        "Taxonomy",
    ])?;
    
    // Write data for each cluster
    for stat in cluster_stats.values() {
        writer.write_record(&[
            stat.cluster.to_string(),
            stat.num_reads.to_string(),
            format!("{:.4}", stat.avg_gc),
            format!("{:.4}", stat.avg_damage),
            format!("{:.4}", stat.avg_damage5),
            format!("{:.4}", stat.avg_damage3),
            stat.taxonomy.clone(),
        ])?;
    }
    
    writer.flush()?;
    Ok(())
}

async fn run_dashboard(
    pca_results: Array2<f64>,
    clusters: Vec<isize>,
    gc_scores: HashMap<String, f64>,
    damage_scores: HashMap<String, DamageScore>,
    read_ids: Vec<String>,
    min_samples: usize,
    eigenvalues: Vec<f64>,
    total_features: usize,
) -> std::io::Result<()> {
    let n = pca_results.nrows();
    
    // Extract first two PCA dimensions for visualization
    let mut pca_vec = Vec::with_capacity(n);
    let mut gc_vec = Vec::with_capacity(n);
    let mut dam_combined_vec = Vec::with_capacity(n);
    let mut dam5_vec = Vec::with_capacity(n);
    let mut dam3_vec = Vec::with_capacity(n);
    
    for i in 0..n {
        // Only use first two dimensions for visualization
        pca_vec.push((pca_results[[i, 0]], pca_results[[i, 1]]));
        
        let rid = &read_ids[i];
        gc_vec.push(*gc_scores.get(rid).unwrap_or(&0.0));
        
        let dscore = damage_scores.get(rid).unwrap_or(&DamageScore {
            damage5: 0.0,
            damage3: 0.0,
            combined: 0.0
        });
        dam_combined_vec.push(dscore.combined);
        dam5_vec.push(dscore.damage5);
        dam3_vec.push(dscore.damage3);
    }
    
    // Compute k-distance and find optimal epsilon
    let k_distance = compute_k_distance_data(&pca_results, min_samples);
    let optimal_eps = find_optimal_eps(&k_distance);
    
    // Compute cluster statistics
    let unique_clusters: Vec<isize> = clusters.iter()
        .cloned()
        .collect::<std::collections::HashSet<isize>>()
        .into_iter()
        .collect();
    
    let mut cluster_stats = Vec::new();
    
    for &cluster_id in &unique_clusters {
        let indices: Vec<usize> = clusters.iter()
            .enumerate()
            .filter_map(|(i, &c)| if c == cluster_id { Some(i) } else { None })
            .collect();
            
        let size = indices.len();
        if size == 0 { continue; }
        
        // Calculate average measures
        let avg_gc = indices.iter()
            .map(|&i| gc_scores.get(&read_ids[i]).unwrap_or(&0.0))
            .sum::<f64>() / size as f64;
            
        let avg_damage = indices.iter()
            .map(|&i| damage_scores.get(&read_ids[i]).unwrap_or(&DamageScore {
                damage5: 0.0,
                damage3: 0.0,
                combined: 0.0
            }).combined)
            .sum::<f64>() / size as f64;
            
        let avg_damage5 = indices.iter()
            .map(|&i| damage_scores.get(&read_ids[i]).unwrap_or(&DamageScore {
                damage5: 0.0,
                damage3: 0.0,
                combined: 0.0
            }).damage5)
            .sum::<f64>() / size as f64;
            
        let avg_damage3 = indices.iter()
            .map(|&i| damage_scores.get(&read_ids[i]).unwrap_or(&DamageScore {
                damage5: 0.0,
                damage3: 0.0,
                combined: 0.0
            }).damage3)
            .sum::<f64>() / size as f64;
            
        // Default empty taxonomy (would be populated by the taxonomy assignment function)
        let taxonomy = if cluster_id == -1 { 
            "Noise".to_string() 
        } else { 
            format!("Cluster {}", cluster_id)
        };
        
        cluster_stats.push(ClusterSummary {
            cluster_id,
            size,
            avg_gc,
            avg_damage,
            avg_damage5,
            avg_damage3,
            taxonomy,
        });
    }
    
    // Compute overall statistics
    let noise_count = clusters.iter().filter(|&&c| c == -1).count();
    let noise_percentage = 100.0 * noise_count as f64 / n as f64;
    let num_clusters = unique_clusters.len() - (if noise_count > 0 { 1 } else { 0 });
    let avg_damage = dam_combined_vec.iter().sum::<f64>() / n as f64;
    
    // Calculate explained variance
    let total_variance: f64 = eigenvalues.iter().sum();
    let explained_variance: Vec<f64> = eigenvalues.iter()
        .map(|&val| (val / total_variance) * 100.0)
        .collect();
    
    // Create comprehensive dashboard data
    let dashboard_data = DashboardData {
        pca: pca_vec,
        clusters,
        read_ids,
        gc_content: gc_vec,
        damage_combined: dam_combined_vec,
        damage5: dam5_vec,
        damage3: dam3_vec,
        k_distance,
        optimal_eps,
        cluster_stats,
        total_reads: n,
        num_clusters,
        noise_percentage,
        avg_damage,
        explained_variance,
        eigenvalues: eigenvalues.to_vec(),
        total_features,
    };
    
    // Set up the web server with the dashboard data
    let data = web::Data::new(dashboard_data);
    
    HttpServer::new(move || {
        App::new()
            .app_data(data.clone())
            .route("/", web::get().to(dashboard_handler))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}

// Replace the dashboard_handler function in main.rs with this updated version
// This fixes the cluster statistics visualization and ensures all plots load properly

async fn dashboard_handler(data: web::Data<DashboardData>) -> impl Responder {
    let json_data = serde_json::to_string(&*data.get_ref()).unwrap();
    let html = format!(
r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>MADERA Pipeline Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1, h2 {{
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .plot-container {{
            background: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 15px;
            margin-bottom: 20px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .metric-card {{
            background: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 15px;
        }}
        .metric-title {{
            font-size: 14px;
            color: #666;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .controls {{
            background: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 15px;
            margin-bottom: 20px;
        }}
        select, input {{
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-right: 10px;
        }}
        button {{
            padding: 8px 16px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }}
        button:hover {{
            background: #45a049;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>MADERA Pipeline Dashboard</h1>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-title">Total Reads</div>
                <div class="metric-value" id="total-reads"></div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Number of Clusters</div>
                <div class="metric-value" id="num-clusters"></div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Noise Points</div>
                <div class="metric-value" id="noise-points"></div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Average Damage Score</div>
                <div class="metric-value" id="avg-damage"></div>
            </div>
        </div>
        
        <div class="controls">
            <h2>Visualization Controls</h2>
            <div>
                <label for="colorby">Color by:</label>
                <select id="colorby" onchange="updateVisualizations()">
                    <option value="cluster">Cluster</option>
                    <option value="damage">Damage Score</option>
                    <option value="gc">GC Content</option>
                </select>
                
                <label for="min-points">Min cluster size:</label>
                <input type="number" id="min-points" min="1" value="5" onchange="filterClusters()">
                
                <button onclick="resetView()">Reset View</button>
            </div>
        </div>
        
        <div class="plot-container">
            <h2>PCA Cluster Visualization</h2>
            <div id="pca_plot" style="width: 100%; height: 600px;"></div>
        </div>
        
        <div class="plot-container">
            <h2>K-distance Plot</h2>
            <p>This plot helps select the optimal epsilon value. The "elbow" point (where the curve sharply increases) is often a good choice for epsilon.</p>
            <div id="k_distance_plot" style="width: 100%; height: 400px;"></div>
        </div>
        
        <div class="plot-container">
            <h2>Damage Distribution</h2>
            <div id="damage_plot" style="width: 100%; height: 400px;"></div>
        </div>
        
        <div class="plot-container">
            <h2>Cluster Statistics</h2>
            <div id="cluster_stats_plot" style="width: 100%; height: 500px;"></div>
        </div>
    </div>
    
    <script>
        // Parse the data from the server
        const data = {json_data};
        let filteredData = JSON.parse(JSON.stringify(data)); // Deep copy
        
        // Calculate metrics
        const totalReads = data.read_ids.length;
        const uniqueClusters = [...new Set(data.clusters.filter(c => c !== -1))];
        const numClusters = uniqueClusters.length;
        const noisePoints = data.clusters.filter(c => c === -1).length;
        const avgDamage = data.damage_combined.reduce((sum, val) => sum + val, 0) / totalReads;
        
        // Update metrics display
        document.getElementById('total-reads').textContent = totalReads.toLocaleString();
        document.getElementById('num-clusters').textContent = numClusters;
        document.getElementById('noise-points').textContent = noisePoints.toLocaleString();
        document.getElementById('avg-damage').textContent = avgDamage.toFixed(4);
        
        // Create initial visualizations
        createPCAPlot();
        createKDistancePlot();
        createDamagePlot();
        createClusterStatsPlot();
        
        function updateVisualizations() {{
            Plotly.purge('pca_plot');
            createPCAPlot();
        }}
        
        function filterClusters() {{
            const minPoints = parseInt(document.getElementById('min-points').value) || 1;
            
            // Count points in each cluster
            const clusterCounts = {{}};
            data.clusters.forEach(cluster => {{
                clusterCounts[cluster] = (clusterCounts[cluster] || 0) + 1;
            }});
            
            // Create filtered data
            filteredData = {{
                pca: [],
                clusters: [],
                read_ids: [],
                gc_content: [],
                damage_combined: [],
                damage5: [],
                damage3: [],
                k_distance: data.k_distance // Keep k_distance unchanged
            }};
            
            // Filter points
            for (let i = 0; i < data.clusters.length; i++) {{
                const cluster = data.clusters[i];
                if (cluster === -1 || clusterCounts[cluster] >= minPoints) {{
                    filteredData.pca.push(data.pca[i]);
                    filteredData.clusters.push(data.clusters[i]);
                    filteredData.read_ids.push(data.read_ids[i]);
                    filteredData.gc_content.push(data.gc_content[i]);
                    filteredData.damage_combined.push(data.damage_combined[i]);
                    filteredData.damage5.push(data.damage5[i]);
                    filteredData.damage3.push(data.damage3[i]);
                }}
            }}
            
            // Update all visualizations
            Plotly.purge('pca_plot');
            createPCAPlot();
            Plotly.purge('damage_plot');
            createDamagePlot();
            Plotly.purge('cluster_stats_plot');
            createClusterStatsPlot();
        }}
        
        function resetView() {{
            document.getElementById('min-points').value = 5;
            document.getElementById('colorby').value = 'cluster';
            filteredData = JSON.parse(JSON.stringify(data)); // Reset to original data
            
            // Update all visualizations
            Plotly.purge('pca_plot');
            createPCAPlot();
            Plotly.purge('damage_plot');
            createDamagePlot();
            Plotly.purge('cluster_stats_plot');
            createClusterStatsPlot();
        }}
        
        function createPCAPlot() {{
            const colorby = document.getElementById('colorby').value;
            let colorValues, colorbarTitle;
            
            if (colorby === 'cluster') {{
                colorValues = filteredData.clusters;
                colorbarTitle = 'Cluster';
            }} else if (colorby === 'damage') {{
                colorValues = filteredData.damage_combined;
                colorbarTitle = 'Damage Score';
            }} else if (colorby === 'gc') {{
                colorValues = filteredData.gc_content;
                colorbarTitle = 'GC Content';
            }}
            
            const traces = [];
            
            // If coloring by cluster, create separate traces for each cluster for better visualization
            if (colorby === 'cluster') {{
                const uniqueClusters = [...new Set(filteredData.clusters)];
                uniqueClusters.sort((a, b) => a - b); // Sort clusters
                
                uniqueClusters.forEach(cluster => {{
                    const indices = filteredData.clusters.map((c, i) => c === cluster ? i : null).filter(i => i !== null);
                    const x = indices.map(i => filteredData.pca[i][0]);
                    const y = indices.map(i => filteredData.pca[i][1]);
                    const text = indices.map(i => filteredData.read_ids[i]);
                    
                    // Use different marker for noise points
                    const trace = {{
                        x: x,
                        y: y,
                        mode: 'markers',
                        type: 'scatter',
                        name: cluster === -1 ? 'Noise' : `Cluster ${{cluster}}`,
                        text: text,
                        marker: {{
                            size: 8,
                            color: cluster === -1 ? 'black' : null,
                            symbol: cluster === -1 ? 'x' : 'circle'
                        }},
                        hovertemplate: 
                            'Read ID: %{{text}}<br>' +
                            'PC1: %{{x:.4f}}<br>' +
                            'PC2: %{{y:.4f}}<br>' +
                            'Cluster: ' + (cluster === -1 ? 'Noise' : cluster) +
                            '<extra></extra>'
                    }};
                    traces.push(trace);
                }});
            }} else {{
                // For continuous values like damage or GC content
                const trace = {{
                    x: filteredData.pca.map(p => p[0]),
                    y: filteredData.pca.map(p => p[1]),
                    mode: 'markers',
                    type: 'scatter',
                    text: filteredData.read_ids,
                    marker: {{
                        size: 8,
                        color: colorValues,
                        colorscale: 'Viridis',
                        colorbar: {{
                            title: colorbarTitle
                        }},
                        showscale: true
                    }},
                    hovertemplate: 
                        'Read ID: %{{text}}<br>' +
                        'PC1: %{{x:.4f}}<br>' +
                        'PC2: %{{y:.4f}}<br>' +
                        `${{colorbarTitle}}: %{{marker.color:.4f}}` +
                        '<extra></extra>'
                }};
                traces.push(trace);
            }}
            
            const layout = {{
                title: 'PCA Cluster Visualization',
                xaxis: {{
                    title: 'Principal Component 1',
                    zeroline: true,
                    zerolinecolor: '#969696',
                    zerolinewidth: 1,
                    gridcolor: '#d9d9d9',
                }},
                yaxis: {{
                    title: 'Principal Component 2',
                    zeroline: true,
                    zerolinecolor: '#969696',
                    zerolinewidth: 1,
                    gridcolor: '#d9d9d9',
                }},
                hovermode: 'closest',
                legend: {{
                    title: {{
                        text: 'Clusters'
                    }}
                }},
                showlegend: colorby === 'cluster'
            }};
            
            Plotly.newPlot('pca_plot', traces, layout, {{
                responsive: true,
                modeBarButtonsToAdd: ['lasso2d', 'select2d'],
                modeBarButtonsToRemove: ['autoScale2d']
            }});
        }}
        
        function createKDistancePlot() {{
            // Sort the distances
            const sortedDistances = [...data.k_distance].sort((a, b) => a - b);
            
            // Filter outliers for better visualization
            const outlierThresholdIdx = Math.min(Math.floor(sortedDistances.length * 0.975), sortedDistances.length - 1);
            const filteredDistances = sortedDistances.slice(0, outlierThresholdIdx);
            
            const trace = {{
                x: Array.from(Array(filteredDistances.length).keys()),
                y: filteredDistances,
                mode: 'lines',
                type: 'scatter',
                line: {{
                    color: 'blue',
                    width: 2
                    }},
                name: 'K-distance'
            }};
            
            // Try to find the elbow point using multiple methods
            // Method 1: Find point of maximum curvature (normalized space)
            const x_norm = Array.from(Array(filteredDistances.length).keys())
                .map(i => i / (filteredDistances.length - 1));
            
            const min_dist = filteredDistances[0];
            const max_dist = filteredDistances[filteredDistances.length - 1];
            const range = max_dist - min_dist;
            
            const y_norm = filteredDistances.map(d => (d - min_dist) / range);
            
            // Knee method - find point with maximum distance to line
            let maxDistance = 0;
            let kneeIdx = 0;
            
            const startX = x_norm[0];
            const startY = y_norm[0];
            const endX = x_norm[x_norm.length - 1];
            const endY = y_norm[x_norm.length - 1];
            
            for (let i = 1; i < x_norm.length - 1; i++) {{
                const x0 = x_norm[i];
                const y0 = y_norm[i];
                
                // Line equation parameters: ax + by + c = 0
                const a = endY - startY;
                const b = startX - endX;
                const c = endX * startY - startX * endY;
                
                // Distance from point to line formula
                const distance = Math.abs(a * x0 + b * y0 + c) / Math.sqrt(a * a + b * b);
                
                if (distance > maxDistance) {{
                    maxDistance = distance;
                    kneeIdx = i;
                }}
            }}
            
            // Method 2: Slope change detection
            let maxSlopeDiff = 0;
            let slopeChangeIdx = 0;
            
            for (let i = 1; i < y_norm.length - 1; i++) {{
                const prevSlope = y_norm[i] - y_norm[i-1];
                const nextSlope = y_norm[i+1] - y_norm[i];
                const slopeDiff = nextSlope - prevSlope;
                
                if (slopeDiff > maxSlopeDiff) {{
                    maxSlopeDiff = slopeDiff;
                    slopeChangeIdx = i;
                }}
            }}
            
            // Choose the better elbow point
            const chosenIdx = Math.abs(kneeIdx - slopeChangeIdx) < filteredDistances.length / 4 ?
                kneeIdx : // Both methods agree
                (kneeIdx < slopeChangeIdx ? kneeIdx : slopeChangeIdx); // Prefer earlier elbow
            
            const elbowValue = filteredDistances[chosenIdx];
            
            // Add points for both methods to the visualization
            const kneeTrace = {{
                x: [kneeIdx],
                y: [filteredDistances[kneeIdx]],
                mode: 'markers',
                type: 'scatter',
                marker: {{
                    color: 'green',
                    size: 10,
                    symbol: 'circle'
                }},
                name: 'Knee Method',
                hovertemplate: 'Knee Method<br>Point: %{{x}}<br>Epsilon: %{{y:.4f}}<extra></extra>'
            }};
            
            const slopeTrace = {{
                x: [slopeChangeIdx],
                y: [filteredDistances[slopeChangeIdx]],
                mode: 'markers',
                type: 'scatter',
                marker: {{
                    color: 'orange',
                    size: 10,
                    symbol: 'diamond'
                }},
                name: 'Slope Method',
                hovertemplate: 'Slope Method<br>Point: %{{x}}<br>Epsilon: %{{y:.4f}}<extra></extra>'
            }};
            
            const chosenTrace = {{
                x: [chosenIdx],
                y: [elbowValue],
                mode: 'markers',
                type: 'scatter',
                marker: {{
                    color: 'red',
                    size: 12,
                    symbol: 'star'
                }},
                name: 'Suggested Epsilon',
                hovertemplate: 'Suggested Epsilon<br>Point: %{{x}}<br>Epsilon: %{{y:.4f}}<extra></extra>'
            }};
            
            const layout = {{
                title: 'K-distance Plot (Sorted kth Nearest Neighbor Distances)',
                xaxis: {{
                    title: 'Point Index (sorted)',
                    gridcolor: '#d9d9d9',
                }},
                yaxis: {{
                    title: 'Distance (epsilon)',
                    gridcolor: '#d9d9d9',
                }},
                annotations: [{{
                    x: chosenIdx,
                    y: elbowValue,
                    text: `Suggested ε = ${{elbowValue.toFixed(4)}}`,
                    showarrow: true,
                    arrowhead: 2,
                    arrowsize: 1,
                    arrowwidth: 2,
                    arrowcolor: '#636363',
                    ax: 40,
                    ay: -40
                }}]
            }};
            
            Plotly.newPlot('k_distance_plot', [trace, kneeTrace, slopeTrace, chosenTrace], layout, {{responsive: true}});
        }}
        
        function createDamagePlot() {{
            // Create stacked histogram for damage scores
            const damage5Trace = {{
                x: filteredData.damage5,
                type: 'histogram',
                opacity: 0.7,
                name: "5' Damage",
                marker: {{
                    color: 'rgba(255, 100, 102, 0.7)'
                }},
                nbinsx: 30
            }};
            
            const damage3Trace = {{
                x: filteredData.damage3,
                type: 'histogram',
                opacity: 0.7,
                name: "3' Damage",
                marker: {{
                    color: 'rgba(100, 149, 237, 0.7)'
                }},
                nbinsx: 30
            }};
            
            const layout = {{
                title: 'Damage Score Distribution',
                xaxis: {{
                    title: 'Damage Score',
                    range: [0, 1],
                    gridcolor: '#d9d9d9',
                }},
                yaxis: {{
                    title: 'Count',
                    gridcolor: '#d9d9d9',
                }},
                barmode: 'overlay',
                bargap: 0.05,
                bargroupgap: 0.2
            }};
            
            Plotly.newPlot('damage_plot', [damage5Trace, damage3Trace], layout, {{responsive: true}});
        }}
        

        function createClusterStatsPlot() {{
            // Compute statistics for each cluster
            const uniqueClusters = [...new Set(filteredData.clusters)].sort((a, b) => a - b);
            const clusterData = [];
            uniqueClusters.forEach(cluster => {{
                if (cluster === -1) return;
                const indices = filteredData.clusters.map((c, i) => c === cluster ? i : null).filter(i => i !== null);
                const size = indices.length;
                const avgDamageScore = indices.reduce((sum, i) => sum + filteredData.damage_combined[i], 0) / size;
                const avgGCContent = indices.reduce((sum, i) => sum + filteredData.gc_content[i], 0) / size;
                clusterData.push({{
                    id: cluster,
                    size: size,
                    avgGC: avgGCContent,
                    avgDamage: avgDamageScore
                }});
            }});
            clusterData.sort((a, b) => b.size - a.size);
            const maxClusterSize = clusterData.length > 0 ? clusterData[0].size : 0;
            const container = document.getElementById('cluster_stats_plot');
            
            // Updated HTML layout with fixed overall height and scrollable table
            container.innerHTML = `
            <div style="background: white; padding: 10px; border-radius: 5px; width: 100%; height: 650px; display: flex; flex-direction: column; box-sizing: border-box;">
                <div class="controls" style="margin-bottom: 15px; display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <label for="cluster-viz-type">Visualization Type:</label>
                        <select id="cluster-viz-type" style="padding: 4px; margin-right: 15px;">
                            <option value="log">Bubble Chart (Log Scale)</option>
                            <option value="sqrt">Bubble Chart (Square Root)</option>
                            <option value="rank">Bubble Chart (Rank-Based)</option>
                            <option value="flat">Scatter Plot (Equal Size)</option>
                        </select>
                    </div>
                    <span style="font-size: 0.9em; color: #666;">
                        Largest cluster: <strong>${{maxClusterSize.toLocaleString()}}</strong> reads
                    </span>
                </div>
                <div class="grid-container" style="flex: 1; height: 100%; display: grid; grid-template-columns: 1fr 1fr; gap: 15px; overflow: hidden;">
                    <div id="viz-panel" style="background: #f9f9f9; border-radius: 5px; padding: 5px; height: 100%;"></div>
                    <div style="background: #f9f9f9; border-radius: 5px; padding: 10px; height: 100%; overflow-y: auto;">
                        <h3 style="margin-top: 0; margin-bottom: 10px;">Top ${{Math.min(10, clusterData.length)}} Clusters by Size</h3>
                        <table id="cluster-table" style="width: 100%; border-collapse: collapse; font-size: 0.9em;">
                            <thead style="background: #f0f0f0;">
                                <tr>
                                    <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">ID</th>
                                    <th style="padding: 8px; border: 1px solid #ddd; text-align: right;">Size</th>
                                    <th style="padding: 8px; border: 1px solid #ddd; text-align: right;">% of Max</th>
                                    <th style="padding: 8px; border: 1px solid #ddd; text-align: right;">GC</th>
                                    <th style="padding: 8px; border: 1px solid #ddd; text-align: right;">Damage</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${{clusterData.slice(0, 10).map((cluster, i) => `
                                    <tr style="background: ${{i % 2 === 0 ? '#fff' : '#f9f9f9'}};">
                                        <td style="padding: 8px; border: 1px solid #ddd;">Cluster ${{cluster.id}}</td>
                                        <td style="padding: 8px; border: 1px solid #ddd; text-align: right;">${{cluster.size.toLocaleString()}}</td>
                                        <td style="padding: 8px; border: 1px solid #ddd; text-align: right;">${{(cluster.size / maxClusterSize * 100).toFixed(1)}}%</td>
                                        <td style="padding: 8px; border: 1px solid #ddd; text-align: right;">${{cluster.avgGC.toFixed(3)}}</td>
                                        <td style="padding: 8px; border: 1px solid #ddd; text-align: right;">${{cluster.avgDamage.toFixed(3)}}</td>
                                    </tr>
                                `).join('')}}
                            </tbody>
                        </table>
                        <div style="margin-top: 15px; font-size: 0.9em; color: #666;">
                            <h4 style="margin-bottom: 5px;">Visualization tips:</h4>
                            <ul style="padding-left: 20px; margin-top: 5px;">
                                <li>Try different scaling options to see relationships more clearly</li>
                                <li>Square root scaling provides better size differentiation</li>
                                <li>Rank-based sizing ignores absolute differences</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
            `;
            
            // Calculate sizing for different methods
            clusterData.forEach((cluster, index) => {{
                cluster.logSize = Math.max(10, Math.log10(cluster.size) * 15);
                cluster.sqrtSize = Math.max(5, Math.sqrt(cluster.size / 100) * 20);
                cluster.rankSize = 40 - (index * 2);
                cluster.percentSize = (cluster.size / maxClusterSize) * 100;
            }});
            
            // Visualization functions
            function createViz(sizeType) {{
                function getSizeValue(item) {{
                    switch (sizeType) {{
                        case 'log': return item.logSize;
                        case 'sqrt': return item.sqrtSize;
                        case 'rank': return item.rankSize;
                        case 'flat': return 15;
                        default: return item.logSize;
                    }}
                }}
                const sortedData = [...clusterData].sort((a, b) => b.size - a.size);
                
                // Dynamically get the current height of the viz-panel container
                const vizHeight = document.getElementById('viz-panel').clientHeight;
                const trace = {{
                    x: sortedData.map(d => d.avgGC),
                    y: sortedData.map(d => d.avgDamage),
                    mode: 'markers',
                    marker: {{
                        size: sortedData.map(d => getSizeValue(d)),
                        color: sortedData.map(d => d.id),
                        colorscale: 'Viridis',
                        opacity: sizeType === 'flat' ? 0.8 : 0.6,
                        line: {{
                            color: 'black',
                            width: 1
                        }}
                    }},
                    text: sortedData.map(d => `Cluster ${{d.id}}<br>Size: ${{d.size.toLocaleString()}} reads<br>GC: ${{d.avgGC.toFixed(3)}}<br>Damage: ${{d.avgDamage.toFixed(3)}}`),
                    hoverinfo: 'text'
                }};
                const layout = {{
                    title: `Cluster Statistics (${{sizeType === 'log' ? 'Logarithmic' : 
                            sizeType === 'sqrt' ? 'Square Root' : 
                            sizeType === 'rank' ? 'Rank-Based' : 'Equal'}} Scaling)`,
                    xaxis: {{
                        title: 'Average GC Content',
                        range: [0.1, 0.7]
                    }},
                    yaxis: {{
                        title: 'Average Damage Score',
                        range: [-0.5, 1.0]
                    }},
                    hovermode: 'closest',
                    margin: {{
                        l: 50,
                        r: 20,
                        t: 40,
                        b: 50
                    }},
                    height: vizHeight
                }};
                Plotly.newPlot('viz-panel', [trace], layout, {{
                    responsive: true,
                    displayModeBar: false
                }});
            }}
            
            // Initialize with the default visualization
            createViz('log');
            
            // Set up the event handler for changing visualization type
            document.getElementById('cluster-viz-type').addEventListener('change', function() {{
                createViz(this.value);
            }});
        }}


    </script>
</body>
</html>"#
    );
    HttpResponse::Ok().content_type("text/html").body(html)
}

/// DBSCAN clustering implementation
/// This handles potential dimensional mismatches and optimizes the clustering algorithm

/// Dynamic dimensional DBSCAN implementation using a trait to handle different dimensions
pub trait DimensionalDBSCAN {
    fn dbscan_clustering(data: &ndarray::Array2<f64>, eps: f64, min_samples: usize) -> Vec<isize>;
}

// Implementation for different PCA dimensions
macro_rules! impl_dbscan_for_dim {
    ($dim:expr) => {
        impl DimensionalDBSCAN for [(); $dim] {
            fn dbscan_clustering(data: &ndarray::Array2<f64>, eps: f64, min_samples: usize) -> Vec<isize> {
                // Ensure data dimensions match the expected dimension
                assert!(data.ncols() == $dim, "Data dimensions ({}) don't match the expected dimension ({})", 
                       data.ncols(), $dim);
                
                use indicatif::{ProgressBar, ProgressStyle};
                use std::collections::{VecDeque, HashMap, HashSet};
                use kiddo::float::kdtree::KdTree;
                use kiddo::float::distance::SquaredEuclidean;
                use std::cmp::min;
                
                const B: usize = 256; // Bucket size for KdTree (unchanged)
                let n_points = data.nrows();
                let eps_sq = eps * eps; // Consistent squaring of epsilon
                
                // Progress bar for tree building
                let pb = ProgressBar::new(n_points as u64);
                pb.set_style(
                    ProgressStyle::with_template(
                        "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) - {msg}"
                    ).unwrap()
                    .progress_chars("##-")
                );
                pb.set_message("Building kd-tree");
                
                // Build the KdTree
                let mut tree = KdTree::<f64, u64, $dim, B, u32>::new();
                
                for i in 0..n_points {
                    // Convert ndarray row to fixed-size array
                    let point = data.row(i);
                    let mut point_arr = [0.0; $dim];
                    for j in 0..$dim {
                        point_arr[j] = point[j];
                    }
                    tree.add(&point_arr, i as u64);
                    
                    if i % 1000 == 0 {
                        pb.set_position(i as u64);
                    }
                }
                pb.finish_with_message("kd-tree built");
                
                // Neighbor cache with LRU-like functionality
                struct NeighborCache {
                    cache: HashMap<usize, Vec<usize>>,
                    max_size: usize,
                    recently_used: VecDeque<usize>,
                }
                
                impl NeighborCache {
                    fn new(max_size: usize) -> Self {
                        Self {
                            cache: HashMap::with_capacity(max_size),
                            max_size,
                            recently_used: VecDeque::with_capacity(max_size),
                        }
                    }
                    
                    fn get(&mut self, key: usize) -> Option<&Vec<usize>> {
                        if self.cache.contains_key(&key) {
                            // Move to end of recently used
                            if let Some(pos) = self.recently_used.iter().position(|&x| x == key) {
                                self.recently_used.remove(pos);
                            }
                            self.recently_used.push_back(key);
                            Some(&self.cache[&key])
                        } else {
                            None
                        }
                    }
                    
                    fn insert(&mut self, key: usize, value: Vec<usize>) {
                        // Evict least recently used item if full
                        if self.cache.len() >= self.max_size && !self.cache.contains_key(&key) {
                            if let Some(oldest) = self.recently_used.pop_front() {
                                self.cache.remove(&oldest);
                            }
                        }
                        
                        self.cache.insert(key, value);
                        self.recently_used.push_back(key);
                    }
                }
                
                // Initialize DBSCAN structures
                let mut labels = vec![-1isize; n_points];
                let mut cluster_id: isize = 0;
                let mut neighbor_cache = NeighborCache::new(min(10000, n_points / 10)); // Adaptive cache size
                
                // Progress bar for DBSCAN
                let pb = ProgressBar::new(n_points as u64);
                pb.set_style(
                    ProgressStyle::with_template(
                        "[{elapsed_precise}] [{bar:40.green/black}] {pos}/{len} ({percent}%) - {msg}"
                    ).unwrap()
                    .progress_chars("##-")
                );
                pb.set_message("Running DBSCAN");
                
                // Optimized single-pass DBSCAN implementation
                for i in 0..n_points {
                    pb.set_position(i as u64);
                    
                    // Skip already classified points
                    if labels[i] != -1 {
                        continue;
                    }
                    
                    // Get neighbors using the cache or compute fresh
                    let neighbors = if let Some(cached) = neighbor_cache.get(i) {
                        cached.clone()
                    } else {
                        let point = data.row(i);
                        let mut point_arr = [0.0; $dim];
                        for j in 0..$dim {
                            point_arr[j] = point[j];
                        }
                        
                        let new_neighbors: Vec<usize> = tree
                            .within::<SquaredEuclidean>(&point_arr, eps_sq)
                            .into_iter()
                            .map(|nn| nn.item as usize)
                            .collect();
                        
                        neighbor_cache.insert(i, new_neighbors.clone());
                        new_neighbors
                    };
                    
                    // Check if point is a core point
                    if neighbors.len() < min_samples {
                        labels[i] = -2; // Mark as noise temporarily
                        continue;
                    }
                    
                    // Start a new cluster
                    cluster_id += 1;
                    labels[i] = cluster_id;
                    
                    // Process neighbors with breadth-first expansion
                    let mut seeds = VecDeque::from(neighbors);
                    let mut processed = HashSet::new();
                    processed.insert(i);
                    
                    while let Some(current) = seeds.pop_front() {
                        // Skip already processed points
                        if processed.contains(&current) {
                            continue;
                        }
                        processed.insert(current);
                        
                        // Skip points already in a cluster
                        if labels[current] > 0 {
                            continue;
                        }
                        
                        // Add current point to cluster
                        labels[current] = cluster_id;
                        
                        // Get neighbors of current point
                        let current_neighbors = if let Some(cached) = neighbor_cache.get(current) {
                            cached.clone()
                        } else {
                            let point = data.row(current);
                            let mut point_arr = [0.0; $dim];
                            for j in 0..$dim {
                                point_arr[j] = point[j];
                            }
                            
                            let new_neighbors: Vec<usize> = tree
                                .within::<SquaredEuclidean>(&point_arr, eps_sq)
                                .into_iter()
                                .map(|nn| nn.item as usize)
                                .collect();
                            
                            neighbor_cache.insert(current, new_neighbors.clone());
                            new_neighbors
                        };
                        
                        // If current point is a core point, add its neighbors to seeds
                        if current_neighbors.len() >= min_samples {
                            for &neighbor in &current_neighbors {
                                if !processed.contains(&neighbor) {
                                    seeds.push_back(neighbor);
                                }
                            }
                        }
                    }
                }
                
                // Final pass - reclassify noise points (-2) as true noise (-1)
                labels.iter_mut().for_each(|label| {
                    if *label == -2 {
                        *label = -1;
                    }
                });
                
                pb.finish_with_message("DBSCAN completed");
                labels
            }
        }
    };
}

// Implement for common PCA dimensions
impl_dbscan_for_dim!(2);
impl_dbscan_for_dim!(3);
impl_dbscan_for_dim!(4);
impl_dbscan_for_dim!(5);
impl_dbscan_for_dim!(6);
impl_dbscan_for_dim!(7);
impl_dbscan_for_dim!(8);
impl_dbscan_for_dim!(9);
impl_dbscan_for_dim!(10);
impl_dbscan_for_dim!(20);


/// Wrapper function to call the correct implementation based on dimensions
pub fn dynamic_dbscan_clustering(data: &ndarray::Array2<f64>, eps: f64, min_samples: usize) -> Vec<isize> {
    match data.ncols() {
        2 => <[(); 2]>::dbscan_clustering(data, eps, min_samples),
        3 => <[(); 3]>::dbscan_clustering(data, eps, min_samples),
        4 => <[(); 4]>::dbscan_clustering(data, eps, min_samples),
        5 => <[(); 5]>::dbscan_clustering(data, eps, min_samples),
        6 => <[(); 6]>::dbscan_clustering(data, eps, min_samples),
        7 => <[(); 7]>::dbscan_clustering(data, eps, min_samples),
        8 => <[(); 8]>::dbscan_clustering(data, eps, min_samples),
        9 => <[(); 9]>::dbscan_clustering(data, eps, min_samples),
        10 => <[(); 10]>::dbscan_clustering(data, eps, min_samples),
        20 => <[(); 20]>::dbscan_clustering(data, eps, min_samples),
        _ => panic!("Unsupported dimension: {}. Add implementation for this dimension.", data.ncols()),
    }
}

/// Implement an improved elbow detection algorithm for k-distance data
/// using the "knee detection" method that's robust to outliers
pub fn find_optimal_eps(k_distances: &[f64]) -> f64 {
    // Sort distances
    let mut sorted_distances = k_distances.to_vec();
    sorted_distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    // Filter outliers by removing the top 2.5% that may skew the elbow detection
    let n = sorted_distances.len();
    let outlier_threshold_idx = (n as f64 * 0.975) as usize;
    let filtered_distances = &sorted_distances[0..std::cmp::min(outlier_threshold_idx, n)];
    
    if filtered_distances.len() < 3 {
        return sorted_distances[n / 2]; // Median as fallback for very small datasets
    }
    
    // Normalize both x and y to [0, 1] for consistent processing
    let x_norm: Vec<f64> = (0..filtered_distances.len())
        .map(|i| i as f64 / (filtered_distances.len() as f64 - 1.0))
        .collect();
    
    let min_dist = *filtered_distances.first().unwrap();
    let max_dist = *filtered_distances.last().unwrap();
    let range = max_dist - min_dist;
    
    if range < 1e-6 {
        // If all distances are nearly identical, return the median
        return filtered_distances[filtered_distances.len() / 2];
    }
    
    let y_norm: Vec<f64> = filtered_distances.iter()
        .map(|&d| (d - min_dist) / range)
        .collect();
    
    // Get the coordinates of the start and end points for the line
    let start_x = x_norm[0];
    let start_y = y_norm[0];
    let end_x = x_norm[x_norm.len() - 1];
    let end_y = y_norm[y_norm.len() - 1];
    
    // Find the point that maximizes the distance to the line from start to end point
    // This is the core of the "knee/elbow detection" algorithm
    let mut max_distance = 0.0;
    let mut knee_idx = 0;
    
    for i in 1..x_norm.len() - 1 {
        // Distance from point to line calculation
        let x0 = x_norm[i];
        let y0 = y_norm[i];
        
        // Line equation parameters: ax + by + c = 0
        let a = end_y - start_y;
        let b = start_x - end_x;
        let c = end_x * start_y - start_x * end_y;
        
        // Distance from point to line formula
        let distance = (a * x0 + b * y0 + c).abs() / (a * a + b * b).sqrt();
        
        if distance > max_distance {
            max_distance = distance;
            knee_idx = i;
        }
    }
    
    // Also implement a slope-based detection as a sanity check
    let mut significant_slope_change_idx = 0;
    let mut max_diff = 0.0;
    
    // Find point where slope changes most dramatically
    for i in 1..y_norm.len() - 1 {
        let prev_slope = y_norm[i] - y_norm[i-1];
        let next_slope = y_norm[i+1] - y_norm[i];
        let slope_diff = next_slope - prev_slope;
        
        if slope_diff > max_diff {
            max_diff = slope_diff;
            significant_slope_change_idx = i;
        }
    }
    
    // Choose between methods - prefer knee method but validate with slope method
    let chosen_idx = if (knee_idx as isize - significant_slope_change_idx as isize).abs() < (filtered_distances.len() as isize / 4) {
        // Both methods agree (within 25% of the data range), use knee method
        knee_idx
    } else if knee_idx < significant_slope_change_idx {
        // Prefer earlier elbow point to avoid being influenced by late outliers
        knee_idx
    } else {
        // Default to slope-based method as it tends to be more conservative
        significant_slope_change_idx
    };
    
    // Map back to original distance value
    let elbow_value = filtered_distances[chosen_idx];
    
    // Debug output
    log::info!("Elbow detection: knee_method={:.4}, slope_method={:.4}, chosen={:.4}", 
              filtered_distances[knee_idx], 
              filtered_distances[significant_slope_change_idx], 
              elbow_value);
    
    elbow_value
}

/// Compute k-distance data for dimensional-aware DBSCAN
pub fn compute_k_distance_data(data: &ndarray::Array2<f64>, k: usize) -> Vec<f64> {
    match data.ncols() {
        2 => compute_k_distance_for_dim::<2>(data, k),
        3 => compute_k_distance_for_dim::<3>(data, k),
        4 => compute_k_distance_for_dim::<4>(data, k),
        5 => compute_k_distance_for_dim::<5>(data, k),
        6 => compute_k_distance_for_dim::<6>(data, k),
        7 => compute_k_distance_for_dim::<7>(data, k),
        8 => compute_k_distance_for_dim::<8>(data, k),
        9 => compute_k_distance_for_dim::<9>(data, k),
        10 => compute_k_distance_for_dim::<10>(data, k),
        20 => compute_k_distance_for_dim::<20>(data, k),
        _ => panic!("Unsupported dimension: {}. Add implementation for this dimension.", data.ncols()),
    }
}

/// K-distance computation for a specific dimension
fn compute_k_distance_for_dim<const D: usize>(data: &ndarray::Array2<f64>, k: usize) -> Vec<f64> {
    use kiddo::float::kdtree::KdTree;
    use kiddo::float::distance::SquaredEuclidean;
    
    const B: usize = 256; // Bucket size
    let n_points = data.nrows();
    let mut tree = KdTree::<f64, u64, D, B, u32>::new();
    
    // Build the kd-tree using PCA points
    for i in 0..n_points {
        let point = data.row(i);
        let mut point_arr = [0.0; D];
        for j in 0..D {
            point_arr[j] = point[j];
        }
        tree.add(&point_arr, i as u64);
    }
    
    let mut kth_distances = Vec::with_capacity(n_points);
    
    // For each point, retrieve its k nearest neighbors
    for i in 0..n_points {
        let point = data.row(i);
        let mut point_arr = [0.0; D];
        for j in 0..D {
            point_arr[j] = point[j];
        }
        
        // Retrieve k+1 nearest neighbors (including the point itself)
        let neighbors = tree.nearest_n::<SquaredEuclidean>(&point_arr, k+1);
        
        if neighbors.len() > k {
            // k+1th neighbor (skip self) gives our desired distance
            // We skip the first one (distance 0 to itself)
            let kth_sq = neighbors[k].distance;
            kth_distances.push(kth_sq.sqrt()); // Convert squared distance to Euclidean
        } else {
            // If we don't have enough neighbors, use the furthest available
            let furthest = neighbors.last().map(|n| n.distance.sqrt()).unwrap_or(0.0);
            kth_distances.push(furthest);
        }
    }
    
    kth_distances
}

/// Main function
fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    
    // If no arguments provided, show banner and exit
    if std::env::args().len() <= 1 {
        display_banner();
        std::process::exit(0);
    }

    // Setup custom help menu
    let matches = setup_cli().get_matches();
    
    // Get input file information
    let fastq_path = matches.get_one::<String>("fastq").cloned();
    let bam_path = matches.get_one::<String>("bam").cloned();

     // Parse arguments into the Args struct
     let args = Args {
        fastq: fastq_path,
        bam: bam_path,
        bam_filter: matches.get_one::<String>("bam_filter").unwrap_or(&String::from("all")).clone(),
        min_length: matches.get_one::<String>("min_length").unwrap_or(&String::from("30")).parse().unwrap_or(30),
        k: matches.get_one::<String>("k").unwrap_or(&String::from("4")).parse().unwrap_or(4),
        output: matches.get_one::<String>("output").unwrap_or(&String::from("cluster_report.csv")).clone(),
        dashboard: matches.get_flag("dashboard"),
        min_samples: matches.get_one::<String>("min_samples").unwrap_or(&String::from("5")).parse().unwrap_or(5),
        eps: matches.get_one::<String>("eps").unwrap_or(&String::from("0.5")).parse().unwrap_or(0.5),
        auto_epsilon: matches.get_flag("auto_epsilon"),
        pca_components: matches.get_one::<String>("pca_components").unwrap_or(&String::from("5")).parse().unwrap_or(5),
        damage_window: matches.get_one::<String>("damage_window").unwrap_or(&String::from("5")).parse().unwrap_or(5),
        sim_threshold: matches.get_one::<String>("sim_threshold").unwrap_or(&String::from("0.5")).parse().unwrap_or(0.5),
        conf_threshold: matches.get_one::<String>("conf_threshold").unwrap_or(&String::from("0.1")).parse().unwrap_or(0.1),
        damage_threshold: matches.get_one::<String>("damage_threshold").unwrap_or(&String::from("0.5")).parse().unwrap_or(0.5),
        use_codon: matches.get_flag("use_codon"),
        batch_size: matches.get_one::<String>("batch_size").unwrap_or(&String::from("1000")).parse().unwrap_or(1000),
        export_clusters: matches.get_flag("export_clusters"),
        export_min_size: matches.get_one::<String>("export_min_size").unwrap_or(&String::from("0")).parse().unwrap_or(0),
        export_format: matches.get_one::<String>("export_format").unwrap_or(&String::from("same")).clone(),
        export_output: matches.get_one::<String>("export_output").unwrap_or(&String::from("exported_clusters.zip")).clone(),
        export_with_metadata: matches.get_flag("export_with_metadata"),
    };

    info!("Starting MADERA Pipeline v0.3.0");
    
    // Validate parameters
    if args.pca_components > 20 {
        error!("PCA components exceeding 20 are not supported by this implementation");
        error!("Please specify a value between 2-20 with --pca-components");
        std::process::exit(1);
    }
    
    if args.min_length < 20 {
        info!("Warning: Min read length of {} may include low-quality reads", args.min_length);
    }
    
    if args.auto_epsilon && args.eps != 0.5 {
        info!("Both --auto-epsilon and --eps provided. Automatic epsilon selection will be used.");
    }
    
    // Start timing the pipeline
    let start_time = std::time::Instant::now();
    
    // Run the pipeline
    match run_pipeline(args) {
        Ok(_) => {
            let duration = start_time.elapsed();
            info!("Pipeline completed successfully in {:.2} seconds", duration.as_secs_f64());
        },
        Err(e) => {
            error!("Pipeline failed: {}", e);
            std::process::exit(1);
        }
    }
}


fn display_banner() {
    println!("{}", r#"


     __  __    _    ____  _____ ____      _    
    |  \/  |  / \  |  _ \| ____|  _ \    / \   
    | |\/| | / _ \ | | | |  _| | |_) |  / _ \  
    | |  | |/ ___ \| |_| | |___|  _ <  / ___ \ 
    |_|  |_/_/   \_\____/|_____|_| \_\/_/   \_\
             
                                        
Metagenomic Ancient DNA Evaluation and Reference-free Analysis
version 0.2.0
"#);
    println!("An advanced pipeline for ancient DNA quality control, damage pattern analysis,");
    println!("clustering, and taxonomic assignment without requiring reference genomes.");
    println!();
    println!("Supports both FASTQ and BAM input formats.");
    println!();
    println!("USAGE:");
    println!("    madera --fastq <FILE> [OPTIONS]");
    println!("    madera --bam <FILE> [OPTIONS]");
    println!();
    println!("For full documentation, run:");
    println!("    madera --help");
    println!();
}

/// Setup the help menu and command line arguments
fn setup_cli() -> Command {
    Command::new("MADERA")
        .version("0.2.0")
        .about(format!("{}\n{}",
            "MADERA Pipeline: Metagenomic Ancient DNA Evaluation and Reference-free Analysis".bright_green().bold(),
            "An advanced tool for ancient DNA quality control and clustering analysis".cyan()
        ))
        .author("MADERA Development Team <contact@example.org>")
        .arg(
            Arg::new("fastq")
                .long("fastq")
                .help("Input FASTQ file with ancient DNA reads")
                .required(false)
                .num_args(1)
                .value_name("FILE")
        )
        .arg(
            Arg::new("bam")
                .long("bam")
                .help("Input BAM file with ancient DNA reads")
                .required(false)
                .num_args(1)
                .value_name("FILE")
        )
        .arg(
            Arg::new("bam_filter")
                .long("bam-filter")
                .help("For BAM input, include only unmapped, only mapped, or all reads")
                .default_value("all")
                .value_parser(["unmapped", "mapped", "all"])
                .num_args(1)
        )
        .group(
            ArgGroup::new("input")
                .arg("fastq")
                .arg("bam")
                .required(true)
        )
        // Input/Output options group
        .arg(
            Arg::new("output")
                .long("output")
                .help("Output CSV file for cluster report")
                .default_value("cluster_report.csv")
                .num_args(1)
                .value_name("FILE")
        )
        .arg(
            Arg::new("dashboard")
                .long("dashboard")
                .help("Launch interactive dashboard (web-based)")
                .action(clap::ArgAction::SetTrue)
        )
        // Quality Control group
        .group(
            ArgGroup::new("quality_control")
                .arg("min_length")
                .multiple(true)
        )
        .arg(
            Arg::new("min_length")
                .long("min_length")
                .help(format!("{}: Minimum read length after QC", "Quality Control".bright_blue().bold()))
                .default_value("30")
                .num_args(1)
                .value_name("INT")
        )
        // Feature Extraction group
        .group(
            ArgGroup::new("feature_extraction")
                .arg("k")
                .arg("use_codon")
                .arg("damage_window")
                .multiple(true)
        )
        .arg(
            Arg::new("k")
                .long("k")
                .help(format!("{}: K-mer length for frequency analysis", "Feature Extraction".bright_blue().bold()))
                .default_value("4")
                .num_args(1)
                .value_name("INT")
        )
        .arg(
            Arg::new("use_codon")
                .long("use_codon")
                .help(format!("{}: Include codon usage features in the analysis", "Feature Extraction".bright_blue().bold()))
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("damage_window")
                .long("damage_window")
                .help(format!("{}: Window size for damage assessment", "Feature Extraction".bright_blue().bold()))
                .default_value("5")
                .num_args(1)
                .value_name("INT")
        )
        // Dimensionality Reduction group
        .group(
            ArgGroup::new("dim_reduction")
                .arg("pca_components")
                .arg("batch_size")
                .multiple(true)
        )
        .arg(
            Arg::new("pca_components")
                .long("pca_components")
                .help(format!("{}: Number of PCA components", "Dimensionality Reduction".bright_blue().bold()))
                .default_value("5")
                .num_args(1)
                .value_name("INT")
        )
        .arg(
            Arg::new("batch_size")
                .long("batch_size")
                .help(format!("{}: Batch size for incremental PCA", "Dimensionality Reduction".bright_blue().bold()))
                .default_value("1000")
                .num_args(1)
                .value_name("INT")
        )
        // Clustering group
        .group(
            ArgGroup::new("clustering")
                .arg("min_samples")
                .arg("eps")
                .arg("auto_epsilon")
                .multiple(true)
        )
        .arg(
            Arg::new("min_samples")
                .long("min_samples")
                .help(format!("{}: Minimum samples for DBSCAN clustering", "Clustering".bright_blue().bold()))
                .default_value("5")
                .num_args(1)
                .value_name("INT")
        )
        .arg(
            Arg::new("eps")
                .long("eps")
                .help(format!("{}: Epsilon radius for DBSCAN clustering", "Clustering".bright_blue().bold()))
                .default_value("0.5")
                .num_args(1)
                .value_name("FLOAT")
        )
        .arg(
            Arg::new("auto_epsilon")
                .long("auto_epsilon")
                .help(format!("{}: Automatically determine optimal epsilon using k-distance method", "Clustering".bright_blue().bold()))
                .action(clap::ArgAction::SetTrue)
        )
        // Taxonomy group
        .group(
            ArgGroup::new("taxonomy")
                .arg("sim_threshold")
                .arg("conf_threshold")
                .arg("damage_threshold")
                .multiple(true)
        )
        .arg(
            Arg::new("sim_threshold")
                .long("sim_threshold")
                .help(format!("{}: Similarity threshold for taxonomic assignment", "Taxonomy".bright_blue().bold()))
                .default_value("0.5")
                .num_args(1)
                .value_name("FLOAT")
        )
        .arg(
            Arg::new("conf_threshold")
                .long("conf_threshold")
                .help(format!("{}: Confidence threshold for taxonomic assignment", "Taxonomy".bright_blue().bold()))
                .default_value("0.1")
                .num_args(1)
                .value_name("FLOAT")
        )
        .arg(
            Arg::new("damage_threshold")
                .long("damage_threshold")
                .help(format!("{}: Damage threshold for authenticity", "Taxonomy".bright_blue().bold()))
                .default_value("0.5")
                .num_args(1)
                .value_name("FLOAT")
        )
        // Export options
        .group(
            ArgGroup::new("cluster_export")
                .arg("export_clusters")
                .arg("export_min_size")
                .arg("export_format")
                .arg("export_output")
                .arg("export_with_metadata")
                .multiple(true)
        )
        .arg(
            Arg::new("export_clusters")
                .long("export-clusters")
                .help(format!("{}: Export sequence clusters to new files", "Cluster Export".bright_blue().bold()))
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("export_min_size")
                .long("export-min-size")
                .help(format!("{}: Minimum cluster size to export (smaller clusters will be skipped)", "Cluster Export".bright_blue().bold()))
                .default_value("0")
                .num_args(1)
                .value_name("INT")
        )
        .arg(
            Arg::new("export_format")
                .long("export-format")
                .help(format!("{}: Export format (same=original format, fasta=convert to FASTA)", "Cluster Export".bright_blue().bold()))
                .default_value("same")
                .value_parser(["same", "fasta"])
                .num_args(1)
        )
        .arg(
            Arg::new("export_output")
                .long("export-output")
                .help(format!("{}: Output filename for exported clusters (will be a zip file)", "Cluster Export".bright_blue().bold()))
                .default_value("exported_clusters.zip")
                .num_args(1)
                .value_name("FILE")
        )
        .arg(
            Arg::new("export_with_metadata")
                .long("export-with-metadata")
                .help(format!("{}: Include cluster metadata in exported files", "Cluster Export".bright_blue().bold()))
                .action(clap::ArgAction::SetTrue)
        )        
        // Add examples section
        .after_help(format!("{}:\n  {} --fastq samples.fastq --min_length 50 --k 3 --auto_epsilon\n  {} --fastq samples.fastq --dashboard --batch_size 5000 --pca_components 3\n  {} --bam samples.bam --bam-filter unmapped --min_length 35\n  {} --bam samples.bam --bam-filter mapped --damage_threshold 0.7\n  {} --fastq samples.fastq --export-clusters --export-min-size 50 --export-format fasta\n  {} --bam samples.bam --export-clusters --export-min-size 100 --export-output clusters.zip",
            "Examples".underline().cyan(),
            "madera_pipeline".green(),
            "madera_pipeline".green(),
            "madera_pipeline".green(),
            "madera_pipeline".green(),
            "madera_pipeline".green(),
            "madera_pipeline".green()
        ))
}




fn run_pipeline(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    // Handle the case where neither or both options are provided
    match (&args.fastq, &args.bam) {
        (None, None) => return Err("Either FASTQ or BAM file must be specified".into()),
        (Some(_), Some(_)) => return Err("Please specify either FASTQ or BAM file, not both".into()),
        _ => {}
    }
    
    // Determine file path and attempt to detect file type for validation
    let file_path = args.fastq.clone().unwrap_or_else(|| args.bam.clone().unwrap());
    
    if !std::path::Path::new(&file_path).exists() {
        return Err(format!("Input file not found: {}", file_path).into());
    }
    
    // Auto-detect file type as a validation step
    let detected_type = match detect_file_type(&file_path) {
        Ok(file_type) => file_type,
        Err(e) => return Err(format!("Error validating file: {}", e).into()),
    };
    
    // Validate that the detected type matches the argument type
    let expected_type = if args.fastq.is_some() { "fastq" } else { "bam" };
    if detected_type != expected_type {
        return Err(format!("File type mismatch: detected '{}' but argument specified '{}'", 
                          detected_type, expected_type).into());
    }
    
    info!("Starting MADERA Pipeline v0.3.0");
    info!("Input file: {} ({})", file_path, detected_type);
    
    // Quality control
    info!("Performing quality control (min length: {})...", args.min_length);
    let reads = match detected_type {
        "fastq" => quality_control(&file_path, args.min_length),
        "bam" => {
            info!("Using BAM filter: {}", args.bam_filter);
            match args.bam_filter.as_str() {
                "unmapped" => {
                    info!("Extracting only unmapped reads from BAM file");
                    extract_unmapped_bam(&file_path, args.min_length)
                },
                "mapped" => {
                    info!("Extracting only mapped reads from BAM file");
                    extract_mapped_bam(&file_path, args.min_length)
                },
                _ => {
                    info!("Extracting all reads from BAM file");
                    quality_control_bam(&file_path, args.min_length, &args.bam_filter)
                }
            }
        },
        _ => return Err("Unsupported file type".into()),
    };
    info!("Total reads after QC: {}", reads.len());
    
    if reads.is_empty() {
        return Err("No reads passed quality control. Check input file or lower min_length parameter.".into());
    }
    
    // Feature extraction
    info!("Computing GC content and damage scores...");
    
    // Debug the computation for one read before doing the full dataset
    if !reads.is_empty() {
        info!("Benchmarking damage computation on a sample read...");
        let sample_read = &reads[0];
        let _sample_score = debug_damage_computation(sample_read, args.damage_window);
        info!("Sample damage computation complete");
    }
    
    // Continue with normal processing
    let gc_scores = compute_gc_for_reads(&reads);
    info!("GC content calculation finished for {} reads", gc_scores.len());
    
    let damage_scores = damage_assessment(&reads, args.damage_window);
    info!("Damage assessment finished for {} reads", damage_scores.len());

    info!("Processed GC content for {} reads", gc_scores.len());
    info!("Processed damage scores for {} reads", damage_scores.len());
    
    // Ensure data consistency before proceeding
    let consistent_reads = ensure_data_consistency(&reads, &gc_scores, &damage_scores);
    
    if consistent_reads.is_empty() {
        return Err("No reads have complete data (GC + damage scores). Cannot proceed.".into());
    }
    
    info!("Using {} reads with complete data for further analysis", consistent_reads.len());

    // Find common reads between GC and damage scores
    let gc_read_ids: HashSet<&String> = gc_scores.keys().collect();
    let damage_read_ids: HashSet<&String> = damage_scores.keys().collect();
    let common_reads = gc_read_ids.intersection(&damage_read_ids).count();
    
    info!("Reads with both GC content and damage scores: {}", common_reads);
    
    if common_reads == 0 {
        return Err("No reads have both GC content and damage scores. Check data processing.".into());
    }
    
    // Precompute feature lists
    let kmers_list = get_all_kmers(args.k);
    info!("Using {}-mer frequency analysis ({} features)", args.k, kmers_list.len());
    
    let codon_list = if args.use_codon { 
        info!("Including codon usage analysis (64 features)");
        Some(get_all_kmers(3)) 
    } else { 
        None 
    };
    
    // Before starting feature extraction, add this debug information
    info!("Reads with both GC content and damage scores: {}", common_reads);
    
    // If there's a major discrepancy, log a warning
    if common_reads < reads.len() / 2 {
        warn!("Less than half of the reads have both GC content and damage scores. Results may be unreliable.");
        
        // Display some debugging information
        let first_few_reads: Vec<&str> = reads.iter()
            .take(5)
            .map(|r| r.id.as_str())
            .collect();
        
        let first_few_gc: Vec<&str> = gc_scores.keys()
            .take(5)
            .map(|s| s.as_str())
            .collect();
        
        let first_few_damage: Vec<&str> = damage_scores.keys()
            .take(5)
            .map(|s| s.as_str())
            .collect();
        
        info!("Sample read IDs: {:?}", first_few_reads);
        info!("Sample GC score IDs: {:?}", first_few_gc);
        info!("Sample damage score IDs: {:?}", first_few_damage);
    }
    
    // Calculate total features dimension for PCA - FIX THE COUNT HERE
    // Count all the features we're actually adding in the feature_extractor:
    // 1. k-mer frequencies: kmers_list.len()
    // 2. Optional codon usage: codon_list.as_ref().map_or(0, |v| v.len())
    // 3. GC content: 1 feature
    // 4. Damage scores (damage5, damage3, combined): 3 features
    // 5. Change point confidence: 1 feature
    // 6. Model fit quality (prime5_model_fit, prime3_model_fit): 2 features
    // 7. Damage ratio: 1 feature 
    // 8. Position-specific decay rate features (prime5_lambda, prime3_lambda): 2 features
    // 9. Read pair support: 1 feature
    // 10. Log-likelihood ratio: 1 feature
    // Total: kmers_list.len() + codon_list.as_ref().map_or(0, |v| v.len()) + 12
    
    let total_features = kmers_list.len() + codon_list.as_ref().map_or(0, |v| v.len()) + 12;
    info!("Total feature space dimension: {}", total_features);
    
    // Define feature extractor closure - ensure it adds exactly total_features features
    let feature_extractor = |read: &Read| -> Vec<f64> {
        let mut features = Vec::with_capacity(total_features);
        
        // 1. k-mer frequencies
        let freq = compute_kmer_freq(&read.seq, args.k, &kmers_list);
        for kmer in &kmers_list {
            features.push(*freq.get(kmer).unwrap_or(&0.0));
        }
        
        // 2. Optional codon usage
        if let Some(ref codon_list) = codon_list {
            let usage = compute_codon_usage(&read.seq);
            for codon in codon_list {
                features.push(*usage.get(codon).unwrap_or(&0.0));
            }
        }
        
        // 3. GC content (1 feature)
        if let Some(&gc) = gc_scores.get(&read.id) {
            features.push(gc);
        } else {
            features.push(compute_gc_content(&read.seq));
        }
        
        // 4. Damage scores (3 features)
        if let Some(damage) = damage_scores.get(&read.id) {
            features.push(damage.damage5);
            features.push(damage.damage3);
            features.push(damage.combined);
        } else {
            features.push(0.2); // Default damage5
            features.push(0.2); // Default damage3
            features.push(0.2); // Default combined
        }
        
        // 5. Change point confidence (1 feature)
        let cp_results = detect_damage_change_points(read, args.damage_window);
        features.push(1.0 - cp_results.prime5_pvalue.min(cp_results.prime3_pvalue));
        
        // 6. Model fit quality (2 features)
        let decay_results = fit_decay_damage_model(read, args.damage_window);
        features.push(decay_results.prime5_model_fit);
        features.push(decay_results.prime3_model_fit);
        
        // 7. Damage ratio (1 feature)
        let damage = damage_scores.get(&read.id).unwrap_or(&DamageScore {
            damage5: 0.2,
            damage3: 0.2,
            combined: 0.2,
        });
        
        let damage_ratio = if damage.damage3 > 0.0 {
            damage.damage5 / damage.damage3
        } else {
            2.0 // Default when no 3' damage
        };
        features.push(damage_ratio.min(3.0).max(0.33)); // Clip to reasonable range
        
        // 8. Position-specific decay rate features (2 features)
        features.push(decay_results.prime5_lambda);
        features.push(decay_results.prime3_lambda);
        
        // 9. Read pair support (1 feature)
        let read_pair_results = analyze_read_pair_overlaps(read);
        features.push(read_pair_results.pair_support_score);
        
        // 10. Log-likelihood ratio (1 feature)
        features.push((decay_results.log_likelihood_ratio.max(-5.0).min(5.0) + 5.0) / 10.0);
        
        // Safety check - ensure we have exactly the expected number of features
        assert_eq!(features.len(), total_features, 
            "Feature count mismatch for read {}: got {}, expected {}", 
            read.id, features.len(), total_features);
        
        features
    };
    
    // Incremental PCA
    info!("Performing incremental PCA (components: {}, batch size: {})...", 
          args.pca_components, args.batch_size);
          
    let (mean, eigenvectors, eigenvalues, pca_results) =
        incremental_pca(&reads, args.batch_size, feature_extractor, total_features, args.pca_components);
    
    // Check if PCA succeeded
    if pca_results.nrows() == 0 {
        return Err("PCA failed: no valid feature vectors could be extracted".into());
    }
    
    // If we have fewer dimensions than requested, adjust
    let actual_dimensions = pca_results.ncols();
    
    info!("PCA yielded {} dimensions (requested: {})", actual_dimensions, args.pca_components);

    info!("PCA complete. Explained variance:");
    let total_variance: f64 = eigenvalues.iter().sum();
    for (i, &val) in eigenvalues.iter().enumerate() {
        let explained = (val / total_variance) * 100.0;
        info!("  PC{}: {:.2}% (eigenvalue: {:.4})", i+1, explained, val);
    }
    // Ensure we have enough dimensions for clustering
    let actual_dimensions = pca_results.ncols();
    if actual_dimensions == 0 {
        return Err("PCA resulted in zero dimensions. Cannot perform clustering.".into());
    }
    
    // Determine how many dimensions to use for clustering
    let cluster_dimensions = std::cmp::min(actual_dimensions, args.pca_components);
    
    // Epsilon selection
    let eps = if args.auto_epsilon {
        info!("Calculating optimal epsilon using k-distance method...");
        // Adapt compute_k_distance_data to handle varying dimensions
        let k_distances = match cluster_dimensions {
            1 => compute_k_distance_for_dim::<1>(&pca_results, args.min_samples),
            2 => compute_k_distance_for_dim::<2>(&pca_results, args.min_samples),
            3 => compute_k_distance_for_dim::<3>(&pca_results, args.min_samples),
            4 => compute_k_distance_for_dim::<4>(&pca_results, args.min_samples),
            5 => compute_k_distance_for_dim::<5>(&pca_results, args.min_samples),
            _ => {
                warn!("K-distance calculation not implemented for {} dimensions. Using first 5 dimensions.", actual_dimensions);
                // Truncate to 5 dimensions for k-distance calculation
                let truncated = truncate_pca_dimensions(&pca_results, 5);
                compute_k_distance_for_dim::<5>(&truncated, args.min_samples)
            }
        };
        
        // Find the elbow point in the k-distance graph
        let optimal_eps = find_optimal_eps(&k_distances);
        info!("Automatically determined epsilon: {:.4} (user provided: {:.4})", 
              optimal_eps, args.eps);
        optimal_eps
    } else {
        info!("Using user-specified epsilon: {:.4}", args.eps);
        args.eps
    };
    
    // Clustering with dimension safety checks
    info!("Clustering reads with DBSCAN (eps: {:.4}, min_samples: {}, dimensions: {})...", 
          eps, args.min_samples, cluster_dimensions);
    
    let clusters = if cluster_dimensions == 1 {
        // Special case for 1D data
        info!("Using 1D clustering algorithm");
        cluster_1d(&pca_results, eps, args.min_samples)
    } else {
        // Dynamic dispatch for DBSCAN based on dimensions
        match cluster_dimensions {
            2 => dbscan_clustering::<2>(&pca_results, eps, args.min_samples),
            3 => dbscan_clustering::<3>(&pca_results, eps, args.min_samples),
            4 => dbscan_clustering::<4>(&pca_results, eps, args.min_samples),
            5 => dbscan_clustering::<5>(&pca_results, eps, args.min_samples),
            _ => {
                warn!("DBSCAN not directly supported for {} dimensions. Using first 5 components.", cluster_dimensions);
                let truncated = truncate_pca_dimensions(&pca_results, 5);
                dbscan_clustering::<5>(&truncated, eps, args.min_samples)
            }
        }
    };
    
    // Compute cluster statistics
    let unique_clusters: std::collections::HashSet<isize> = clusters.iter().cloned().collect();
    let noise_count = clusters.iter().filter(|&&c| c == -1).count();
    let cluster_count = unique_clusters.len() - (if noise_count > 0 { 1 } else { 0 });
    
    info!("DBSCAN complete. Found {} clusters and {} noise points ({:.1}%)", 
          cluster_count, noise_count, 100.0 * noise_count as f64 / clusters.len() as f64);
    
    // Process cluster results
    info!("Computing cluster statistics...");
    let read_ids: Vec<String> = reads.iter().map(|r| r.id.clone()).collect();
    let mut cluster_stats = compute_cluster_stats(&read_ids, &clusters, &gc_scores, &damage_scores);
    
    // Print cluster statistics
    for (&cluster_id, stats) in cluster_stats.iter() {
        if cluster_id == -1 {
            info!("Noise points: {} reads, Avg GC: {:.4}, Avg Damage: {:.4}",
                  stats.num_reads, stats.avg_gc, stats.avg_damage);
        } else {
            info!("Cluster {}: {} reads, Avg GC: {:.4}, Avg Damage: {:.4}",
                  cluster_id, stats.num_reads, stats.avg_gc, stats.avg_damage);
        }
    }
    
    // Assign taxonomy
    info!("Assigning taxonomy (similarity threshold: {:.2}, confidence threshold: {:.2})...", 
          args.sim_threshold, args.conf_threshold);
    let taxonomy = assign_taxonomy(
        &cluster_stats,
        &pca_results,
        &clusters,
        args.k,
        args.sim_threshold,
        args.conf_threshold,
        args.damage_threshold,
        &mean,
        &eigenvectors
    );
    
    for (cluster_id, stat) in cluster_stats.iter_mut() {
        stat.taxonomy = taxonomy.get(cluster_id).unwrap_or(&"Unassigned".to_string()).clone();
    }
    
    // Generate report
    info!("Generating cluster report: {}", args.output);
    if let Err(e) = generate_report(&cluster_stats, &args.output) {
        error!("Error generating report: {}", e);
        return Err(format!("Failed to write report: {}", e).into());
    }

    // Export clusters if requested
    if args.export_clusters {
        info!("Exporting cluster sequences...");
        let file_type = if args.fastq.is_some() { "fastq" } else { "bam" };
        
        // For BAM exports, use the specialized function
        if file_type == "bam" && args.export_format == "same" {
            export_bam_clusters(
                &reads, 
                &clusters, 
                &cluster_stats, 
                &args.bam.clone().unwrap(),
                &args
            )?;
        } else {
            // For FASTQ or FASTA exports
            export_clusters(
                &reads, 
                &clusters, 
                &cluster_stats, 
                file_type,
                &damage_scores,
                &args
            )?;
        }
    }
    
    // Launch dashboard if requested
    if args.dashboard {
        info!("Launching interactive dashboard at http://127.0.0.1:8080");
        info!("Press Ctrl+C to exit");
        
        let sys = actix_web::rt::System::new();
        sys.block_on(run_dashboard(
            pca_results, 
            clusters, 
            gc_scores, 
            damage_scores, 
            read_ids, 
            args.min_samples,
            eigenvalues,
            total_features
        ))?;
    }
    
    Ok(())
}

/// Helper function to truncate PCA results to a lower dimension
fn truncate_pca_dimensions(data: &ndarray::Array2<f64>, dimensions: usize) -> ndarray::Array2<f64> {
    if data.ncols() <= dimensions {
        return data.clone();
    }
    
    let rows = data.nrows();
    let mut result = ndarray::Array2::<f64>::zeros((rows, dimensions));
    
    for i in 0..rows {
        for j in 0..dimensions {
            result[[i, j]] = data[[i, j]];
        }
    }
    
    result
}

// Fixed function
/// Ensures data consistency by finding reads that have both GC and damage scores
/// Returns only the reads that have complete data for further analysis
fn ensure_data_consistency(
    reads: &[Read],
    gc_scores: &HashMap<String, f64>,
    damage_scores: &HashMap<String, DamageScore>
) -> Vec<Read> {
    use std::collections::HashSet;
    
    // Count various types for diagnostics
    let mut total_reads = 0;
    let mut with_gc = 0;
    let mut with_damage = 0;
    let mut with_both = 0;
    
    // Create a set of read IDs that have both types of data
    let gc_read_ids: HashSet<&String> = gc_scores.keys().collect();
    let damage_read_ids: HashSet<&String> = damage_scores.keys().collect();
    
    // Find the intersection of read IDs that have both types of data
    let common_id_set: HashSet<_> = gc_read_ids.intersection(&damage_read_ids)
        .map(|&id| id.clone())
        .collect();
    
    // Check each read for the presence of both GC and damage scores
    for read in reads {
        total_reads += 1;
        let has_gc = gc_scores.contains_key(&read.id);
        let has_damage = damage_scores.contains_key(&read.id);
        
        if has_gc { with_gc += 1; }
        if has_damage { with_damage += 1; }
        if has_gc && has_damage { with_both += 1; }
    }
    
    // Log statistics
    info!("Total reads: {}", total_reads);
    info!("Reads with GC scores: {}", with_gc);
    info!("Reads with damage scores: {}", with_damage);
    info!("Reads with all data: {}", with_both);
    
    // Find examples of missing data for diagnostics
    let mut gc_missing_examples = Vec::new();
    let mut damage_missing_examples = Vec::new();
    
    for read in reads.iter().take(100) { // Only check the first 100 reads for examples
        if !gc_scores.contains_key(&read.id) {
            gc_missing_examples.push(read.id.clone());
        }
        if !damage_scores.contains_key(&read.id) {
            damage_missing_examples.push(read.id.clone());
        }
        
        // Stop once we have enough examples
        if gc_missing_examples.len() >= 5 && damage_missing_examples.len() >= 5 {
            break;
        }
    }
    
    // Log examples
    if !gc_missing_examples.is_empty() {
        info!("Examples of reads missing GC scores: {:?}", gc_missing_examples);
    }
    if !damage_missing_examples.is_empty() {
        info!("Examples of reads missing damage scores: {:?}", damage_missing_examples);
    }
    
    // Filter reads to only include those with both GC and damage scores
    reads.iter()
        .filter(|r| common_id_set.contains(&r.id))
        .cloned()
        .collect()
}
