//! MADERA Pipeline in Rust: Metagenomic Ancient DNA Evaluation and Reference-free Analysis
//! 
//! This program performs quality control, feature extraction (including k-mer frequencies,
//! optional codon usage, GC content, and damage scores), incremental PCA with progress updates,
//! and clustering using DBSCAN with spatial indexing (via a kd-tree) and parallel neighbor precomputation.

use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use env_logger;
use log::{error, info};
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

lazy_static! {
    static ref GLOBAL_CACHE: Arc<ReadPairCache> = Arc::new(ReadPairCache::new());
}

/// Command-line arguments with additions for automatic clustering parameters
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Input FASTQ file with ancient DNA reads
    #[arg(long)]
    fastq: String,

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


/// Performs quality control on reads from a FASTQ file
fn quality_control(fastq_path: &str, min_length: usize) -> Vec<Read> {
    use bio::io::fastq;
    use std::path::Path;
    
    let path = Path::new(fastq_path);
    let reader = match fastq::Reader::from_file(path) {
        Ok(reader) => reader,
        Err(e) => {
            log::error!("Failed to open FASTQ file: {}", e);
            return Vec::new();
        }
    };
    
    let mut reads = Vec::new();
    let mut total_reads = 0;
    let mut passed_reads = 0;
    
    for record in reader.records() {
        total_reads += 1;
        
        if let Ok(record) = record {
            let seq = record.seq();
            if seq.len() >= min_length {
                // Convert the sequence and quality scores
                let read = Read {
                    id: String::from_utf8_lossy(record.id().as_bytes()).into_owned(),
                    seq: String::from_utf8_lossy(seq).to_string(),
                    quality: Some(record.qual().to_vec()),
                };
                
                reads.push(read);
                passed_reads += 1;
            }
        }
    }
    
    log::info!("QC: {} out of {} reads passed minimum length filter", passed_reads, total_reads);
    reads
}

/// Compute GC content for a collection of reads
fn compute_gc_for_reads(reads: &[Read]) -> HashMap<String, f64> {
    use rayon::prelude::*;
    use std::sync::Mutex;
    
    let gc_scores = Mutex::new(HashMap::new());
    
    reads.par_iter().for_each(|read| {
        let gc = compute_gc_content(&read.seq);
        
        let mut scores = gc_scores.lock().unwrap();
        scores.insert(read.id.clone(), gc);
    });
    
    gc_scores.into_inner().unwrap()
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
    use std::sync::Mutex;
    
    let scores = Mutex::new(HashMap::new());
    
    reads.par_iter().for_each(|read| {
        let damage = compute_damage_scores(read, window_size);
        
        let mut damage_scores = scores.lock().unwrap();
        damage_scores.insert(read.id.clone(), damage);
    });
    
    scores.into_inner().unwrap()
}

/// Enhanced structs needed for the improved damage assessment
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

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ReadPairInfo {
    id: String,
    overlap_region: Option<(usize, usize)>,
    overlap_mismatches: Vec<(usize, char, char)>, // position, base1, base2
    is_ancient_pattern: bool,
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
    
    // Return the comprehensive damage score
    DamageScore {
        damage5: combined_score.prime5_score,
        damage3: combined_score.prime3_score,
        combined: combined_score.authenticity_score,
    }
}

/// APPROACH 1: Change-point analysis for damage detection
/// This method finds where damage patterns significantly change along the read
fn detect_damage_change_points(read: &Read, window_size: usize) -> ChangePointResults {
    let seq = &read.seq;
    let seq_len = seq.len();
    
    // Skip very short reads
    if seq_len < 15 {
        return ChangePointResults {
            prime5_change_point: None,
            prime3_change_point: None,
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
        prime5_change_point: five_change.position,
        prime3_change_point: three_change.position,
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
            statistic: 0.0,
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
    // This is a simplified approximation
    let pvalue = if max_statistic > 0.0 {
        let z_score = max_statistic / background_rate.sqrt();
        approximate_pvalue(z_score)
    } else {
        1.0
    };
    
    ChangePointResult {
        position: change_pos,
        pvalue,
        statistic: max_statistic,
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

/// Approximation of p-value from z-score
fn approximate_pvalue(z_score: f64) -> f64 {
    if z_score <= 0.0 {
        return 1.0;
    }
    
    // Simple approximation of the normal CDF
    let x = z_score / 1.414213562373095;
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989423 * (-x * x / 2.0).exp();
    let p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    
    if x > 0.0 { 1.0 - p } else { p }
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
                    // Enhanced context analysis
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
                    // Enhanced context analysis for 3' end
                    let (_, prob) = analyze_three_prime_context(i, &bases);
                    (prob, true)
                },
                'G' => {
                    // G that could deaminate but hasn't yet
                    // Enhanced susceptibility analysis
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
/// Enhanced read pair overlap analysis system for ancient DNA authentication
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
    /// Add a read to the cache and try to find its pair
    pub fn add_read(&self, read: &Read) -> bool {
        let mut cache = self.cache.write().unwrap();
        let mut stats = self.stats.lock().unwrap();
        let mut unprocessed = self.unprocessed_pairs.lock().unwrap();
        
        // Extract the base ID without /1 or /2 suffix
        let (base_id, read_number) = parse_read_id(&read.id);
        
        // Skip if we can't determine read number
        if read_number == 0 {
            return false;
        }
        
        // Look for the paired read in the cache
        let pair_id = if read_number == 1 {
            format!("{}/2", base_id)
        } else {
            format!("{}/1", base_id)
        };
        
        let paired_read = if read_number == 1 {
            if let Some(pair_info) = cache.get(&pair_id) {
                // Found the paired read (read2)
                stats.pair_found += 1;
                Some(PairedReadInfo {
                    read1_id: read.id.clone(),
                    read2_id: pair_id,
                    read1_seq: read.seq.clone(),
                    read2_seq: pair_info.read1_seq.clone(),
                    read1_qual: read.quality.clone(),
                    read2_qual: pair_info.read1_qual.clone(),
                    processed: false,
                })
            } else {
                // Paired read not found yet
                stats.pair_not_found += 1;
                None
            }
        } else {
            // This is read2, look for read1
            if let Some(pair_info) = cache.get(&pair_id) {
                // Found the paired read (read1)
                stats.pair_found += 1;
                Some(PairedReadInfo {
                    read1_id: pair_id,
                    read2_id: read.id.clone(),
                    read1_seq: pair_info.read1_seq.clone(),
                    read2_seq: read.seq.clone(),
                    read1_qual: pair_info.read1_qual.clone(),
                    read2_qual: read.quality.clone(),
                    processed: false,
                })
            } else {
                // Paired read not found yet
                stats.pair_not_found += 1;
                None
            }
        };
        
        // If we found a pair, store the completed pair info and mark for processing
        if let Some(pair_info) = paired_read {
            // Add to unprocessed set for batch processing
            unprocessed.insert(base_id.clone());
            // Update the cache
            cache.insert(base_id, pair_info);
            stats.inserts += 1;
            true
        } else {
            // Store this read in case its pair comes later
            let self_info = PairedReadInfo {
                read1_id: read.id.clone(),
                read2_id: String::new(),
                read1_seq: read.seq.clone(),
                read2_seq: String::new(),
                read1_qual: read.quality.clone(),
                read2_qual: None,
                processed: false,
            };
            
            // Store using the appropriate ID format
            let cache_id = if read_number == 1 {
                format!("{}/1", base_id)
            } else {
                format!("{}/2", base_id)
            };
            
            cache.insert(cache_id, self_info);
            stats.inserts += 1;
            false
        }
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
    
    if read_id.contains("/1") {
        (read_id.replace("/1", ""), 1)
    } else if read_id.contains("/2") {
        (read_id.replace("/2", ""), 2)
    } else if read_id.contains(".1") && read_id.ends_with(".1") {
        (read_id.replace(".1", ""), 1)
    } else if read_id.contains(".2") && read_id.ends_with(".2") {
        (read_id.replace(".2", ""), 2)
    } else if read_id.contains("_1") && read_id.ends_with("_1") {
        (read_id.replace("_1", ""), 1)
    } else if read_id.contains("_2") && read_id.ends_with("_2") {
        (read_id.replace("_2", ""), 2)
    } else if read_id.contains("-1") && read_id.ends_with("-1") {
        (read_id.replace("-1", ""), 1)
    } else if read_id.contains("-2") && read_id.ends_with("-2") {
        (read_id.replace("-2", ""), 2)
    } else if read_id.contains(":1") && read_id.ends_with(":1") {
        (read_id.replace(":1", ""), 1)
    } else if read_id.contains(":2") && read_id.ends_with(":2") {
        (read_id.replace(":2", ""), 2)
    } else if read_id.contains(" R1") && read_id.ends_with(" R1") {
        (read_id.replace(" R1", ""), 1)
    } else if read_id.contains(" R2") && read_id.ends_with(" R2") {
        (read_id.replace(" R2", ""), 2)
    } else {
        // Can't determine read number
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
    statistic: f64,
}

#[derive(Debug)]
struct ChangePointResults {
    prime5_change_point: Option<usize>,
    prime3_change_point: Option<usize>,
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
    feature_extractor: impl Fn(&Read) -> Vec<f64>,
    total_features: usize,
    n_components: usize
) -> (Array1<f64>, Array2<f64>, Vec<f64>, Array2<f64>) {
    use ndarray::Array1;
    use ndarray_linalg::Eig;
    use indicatif::{ProgressBar, ProgressStyle};
    
    // Initial parameters
    let n_samples = reads.len();
    let mut mean = Array1::<f64>::zeros(total_features);
    let mut components = Array2::<f64>::zeros((total_features, total_features));
    let mut explained_variance = Vec::with_capacity(n_components);
    let mut n_samples_seen = 0;
    
    // Progress bar
    let pb = ProgressBar::new(n_samples as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} reads processed"
        )
        .unwrap()
        .progress_chars("##-")
    );
    
    // Process in batches
    for batch_start in (0..n_samples).step_by(batch_size) {
        let batch_end = std::cmp::min(batch_start + batch_size, n_samples);
        let batch_size = batch_end - batch_start;
        
        // Extract features for this batch
        let mut batch_data = Array2::<f64>::zeros((batch_size, total_features));
        
        for (i, read_idx) in (batch_start..batch_end).enumerate() {
            let features = feature_extractor(&reads[read_idx]);
            for (j, &value) in features.iter().enumerate() {
                batch_data[[i, j]] = value;
            }
        }
        
        // Incremental fit
        if n_samples_seen == 0 {
            // First batch - initialize
            for i in 0..batch_size {
                mean += &batch_data.row(i);
            }
            mean /= batch_size as f64;
            
            // Center data
            let mut centered = batch_data.clone();
            for i in 0..batch_size {
                for j in 0..mean.len() {
                    centered[[i, j]] -= mean[j];
                }
            }
            
            // Compute covariance
            let cov = centered.t().dot(&centered) / (batch_size as f64 - 1.0);
            
            // Eigendecomposition
            let eig = cov.eig().unwrap();
            let (_eigenvalues, eigenvectors) = eig;
            // Convert complex eigenvectors to real values
            for i in 0..total_features {
                for j in 0..total_features {
                    components[[i, j]] = eigenvectors[[i, j]].re;
                }
            }
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
            let mut centered = batch_data.clone();
            for i in 0..batch_size {
                for j in 0..mean.len() {
                    centered[[i, j]] -= mean[j];
                }
            }
            
            // Update covariance
            let batch_cov = centered.t().dot(&centered) / (new_sample_count - 1.0);
            let old_cov = components.dot(&components.t());
            let new_cov = (old_cov * old_sample_count + batch_cov * new_sample_count) / total_samples;
            
            // Eigendecomposition on updated covariance
            let eig = new_cov.eig().unwrap();
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
    
    eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    
    let sorted_indices: Vec<usize> = eigen_pairs.iter().map(|&(_, i)| i).collect();
    let mut sorted_components = Array2::<f64>::zeros((total_features, n_components));
    let mut sorted_variance = Vec::with_capacity(n_components);
    
    for (i, &idx) in sorted_indices.iter().take(n_components).enumerate() {
        sorted_components.column_mut(i).assign(&components.column(idx));
        sorted_variance.push(explained_variance[idx]);
    }
    
    // Transform the data
    let mut transformed = Array2::<f64>::zeros((n_samples, n_components));
    
    for (i, read) in reads.iter().enumerate() {
        let features = feature_extractor(read);
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

/// Compute statistics for each cluster
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
    pca_results: &Array2<f64>,
    clusters: &[isize],
    k: usize,
    sim_threshold: f64,
    conf_threshold: f64,
    damage_threshold: f64,
    mean: &Array1<f64>,
    eigenvectors: &Array2<f64>,
) -> HashMap<isize, String> {
    // This is a placeholder implementation since real taxonomic assignment
    // would require a reference database and more complex algorithms
    
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
        // later implementation would use reference matching, ML models, etc.
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

/// Generate report with cluster statistics
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

/// Dashboard handler function - moved to module level
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
        document.getElementById('total-reads').textContent = totalReads;
        document.getElementById('num-clusters').textContent = numClusters;
        document.getElementById('noise-points').textContent = noisePoints;
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
            // Only need to create this once as it doesn't depend on filtering
            const sortedDistances = [...data.k_distance].sort((a, b) => a - b);
            const trace = {{
                x: Array.from(Array(sortedDistances.length).keys()),
                y: sortedDistances,
                mode: 'lines',
                type: 'scatter',
                line: {{
                    color: 'blue',
                    width: 2
                }},
                name: 'K-distance'
            }};
            
            // Try to find the elbow point
            // Simple method: find point of maximum curvature
            let maxCurvature = 0;
            let elbowPoint = 0;
            
            if (sortedDistances.length > 2) {{
                for (let i = 1; i < sortedDistances.length - 1; i++) {{
                    const prevDiff = sortedDistances[i] - sortedDistances[i-1];
                    const nextDiff = sortedDistances[i+1] - sortedDistances[i];
                    const curvature = Math.abs(nextDiff - prevDiff);
                    
                    if (curvature > maxCurvature) {{
                        maxCurvature = curvature;
                        elbowPoint = i;
                    }}
                }}
            }} else {{
                // Default to the middle point if not enough points
                elbowIdx = n / 2;
            }}
            
            // Add a trace to highlight the elbow point
            const elbowTrace = {{
                x: [elbowPoint],
                y: [sortedDistances[elbowPoint]],
                mode: 'markers',
                type: 'scatter',
                marker: {{
                    color: 'red',
                    size: 10,
                    symbol: 'circle'
                }},
                name: 'Suggested Epsilon',
                hovertemplate: 
                    'Point: %{{x}}<br>' +
                    'Distance (epsilon): %{{y:.4f}}<br>' +
                    '<extra>Suggested Epsilon</extra>'
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
                    x: elbowPoint,
                    y: sortedDistances[elbowPoint],
                    text: `Suggested ε = ${{sortedDistances[elbowPoint].toFixed(4)}}`,
                    showarrow: true,
                    arrowhead: 2,
                    arrowsize: 1,
                    arrowwidth: 2,
                    arrowcolor: '#636363',
                    ax: 40,
                    ay: -40
                }}]
            }};
            
            Plotly.newPlot('k_distance_plot', [trace, elbowTrace], layout, {{responsive: true}});
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
            const clusterSizes = [];
            const avgDamage = [];
            const avgGC = [];
            const clusterLabels = [];
            
            uniqueClusters.forEach(cluster => {{
                if (cluster === -1) return; // Skip noise points for this analysis
                
                const indices = filteredData.clusters.map((c, i) => c === cluster ? i : null).filter(i => i !== null);
                clusterSizes.push(indices.length);
                
                const avgDamageScore = indices.reduce((sum, i) => sum + filteredData.damage_combined[i], 0) / indices.length;
                avgDamage.push(avgDamageScore);
                
                const avgGCContent = indices.reduce((sum, i) => sum + filteredData.gc_content[i], 0) / indices.length;
                avgGC.push(avgGCContent);
                
                clusterLabels.push(`Cluster ${{cluster}}`);
            }});
            
            // Create bubble chart
            const trace = {{
                x: avgGC,
                y: avgDamage,
                mode: 'markers',
                text: clusterLabels,
                marker: {{
                    size: clusterSizes.map(s => Math.min(Math.max(s / 5, 10), 50)), // Scale bubble sizes
                    sizemode: 'diameter',
                    sizeref: 1,
                    color: uniqueClusters.filter(c => c !== -1),
                    colorscale: 'Viridis',
                    showscale: true,
                    colorbar: {{
                        title: 'Cluster ID'
                    }},
                    line: {{
                        color: 'black',
                        width: 1
                    }}
                }},
                hovertemplate: 
                    '%{{text}}<br>' +
                    'Size: %{{marker.size:.0f}} reads<br>' +
                    'GC Content: %{{x:.4f}}<br>' +
                    'Damage Score: %{{y:.4f}}<br>' +
                    '<extra></extra>'
            }};
            
            const layout = {{
                title: 'Cluster Statistics',
                xaxis: {{
                    title: 'Average GC Content',
                    gridcolor: '#d9d9d9',
                }},
                yaxis: {{
                    title: 'Average Damage Score',
                    gridcolor: '#d9d9d9',
                }}
            }};
            
            Plotly.newPlot('cluster_stats_plot', [trace], layout, {{responsive: true}});
        }}
    </script>
</body>
</html>"#
    );
    HttpResponse::Ok().content_type("text/html").body(html)
}

/// DBSCAN clustering implementation with dynamic dimensions
/// This handles the dimensional mismatch and optimizes the clustering algorithm

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
                
                // Build the KdTree with proper dimension
                let mut tree = KdTree::<f64, u64, $dim, B, u32>::new();
                
                for i in 0..n_points {
                    // Convert ndarray row to fixed-size array with correct dimensions
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
                
                // Improved neighbor cache with LRU-like functionality
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
        10 => <[(); 10]>::dbscan_clustering(data, eps, min_samples),
        20 => <[(); 20]>::dbscan_clustering(data, eps, min_samples),
        _ => panic!("Unsupported dimension: {}. Add implementation for this dimension.", data.ncols()),
    }
}

/// Implement automatic epsilon selection using the elbow method on k-distance graph
pub fn find_optimal_eps(k_distances: &[f64]) -> f64 {
    // Sort distances
    let mut sorted_distances = k_distances.to_vec();
    sorted_distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    // Calculate curvature to find the elbow
    // Method: Approximate second derivative using finite differences
    let mut max_curvature = 0.0;
    let mut elbow_idx = 0;
    let n = sorted_distances.len();
    
    // We need at least 3 points to compute curvature
    if n > 2 {
        // Simple second derivative calculation for each interior point
        for i in 1..n-1 {
            let y_prev = sorted_distances[i-1];
            let y = sorted_distances[i];
            let y_next = sorted_distances[i+1];
            
            // Approximation of second derivative
            let second_derivative = y_next - 2.0 * y + y_prev;
            
            // Find the point with maximum curvature
            if second_derivative.abs() > max_curvature {
                max_curvature = second_derivative.abs();
                elbow_idx = i;
            }
        }
    } else {
        // Default to the middle point if not enough points
        elbow_idx = n / 2;
    }
    
    // Return the distance at the elbow point
    sorted_distances[elbow_idx]
}

/// Compute k-distance data for dimensional-aware DBSCAN
pub fn compute_k_distance_data(data: &ndarray::Array2<f64>, k: usize) -> Vec<f64> {
    match data.ncols() {
        2 => compute_k_distance_for_dim::<2>(data, k),
        3 => compute_k_distance_for_dim::<3>(data, k),
        4 => compute_k_distance_for_dim::<4>(data, k),
        5 => compute_k_distance_for_dim::<5>(data, k),
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
    
    // Build the kd-tree using PCA points with correct dimensions
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
    

    // Parse arguments into the Args struct
    let args = Args {
        fastq: matches.get_one::<String>("fastq").unwrap().clone(),
        min_length: matches.get_one::<String>("min_length").unwrap().parse().unwrap_or(30),
        k: matches.get_one::<String>("k").unwrap().parse().unwrap_or(4),
        output: matches.get_one::<String>("output").unwrap().clone(),
        dashboard: matches.get_flag("dashboard"),
        min_samples: matches.get_one::<String>("min_samples").unwrap().parse().unwrap_or(5),
        eps: matches.get_one::<String>("eps").unwrap().parse().unwrap_or(0.5),
        auto_epsilon: matches.get_flag("auto_epsilon"),
        pca_components: matches.get_one::<String>("pca_components").unwrap().parse().unwrap_or(5),
        damage_window: matches.get_one::<String>("damage_window").unwrap().parse().unwrap_or(5),
        sim_threshold: matches.get_one::<String>("sim_threshold").unwrap().parse().unwrap_or(0.5),
        conf_threshold: matches.get_one::<String>("conf_threshold").unwrap().parse().unwrap_or(0.1),
        damage_threshold: matches.get_one::<String>("damage_threshold").unwrap().parse().unwrap_or(0.5),
        use_codon: matches.get_flag("use_codon"),
        batch_size: matches.get_one::<String>("batch_size").unwrap().parse().unwrap_or(1000),
    };


    info!("Starting MADERA Pipeline v0.2.0");
    info!("Input file: {}", args.fastq);
    
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
    println!("USAGE:");
    println!("    madera-pipeline --fastq <FILE> [OPTIONS]");
    println!();
    println!("For full documentation, run:");
    println!("    madera-pipeline --help");
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
                .required(true)
                .num_args(1)
                .value_name("FILE")
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
        // Add examples section
        .after_help(format!("{}:\n  {} --fastq samples.fastq --min_length 50 --k 3 --auto_epsilon\n  {} --fastq samples.fastq --dashboard --batch_size 5000 --pca_components 3",
            "Examples".underline().cyan(),
            "madera_pipeline".green(),
            "madera_pipeline".green()
        ))
}


/// Update run_pipeline to ensure all parameters are correctly passed
fn run_pipeline(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    // Validate input file
    if !std::path::Path::new(&args.fastq).exists() {
        return Err(format!("Input file not found: {}", args.fastq).into());
    }
    
    // Quality control
    info!("Performing quality control (min length: {})...", args.min_length);
    let reads = quality_control(&args.fastq, args.min_length);
    info!("Total reads after QC: {}", reads.len());
    
    if reads.is_empty() {
        return Err("No reads passed quality control. Check input file or lower min_length parameter.".into());
    }
    
    // Feature extraction
    info!("Computing GC content and damage scores...");
    let gc_scores = compute_gc_for_reads(&reads);
    let damage_scores = damage_assessment(&reads, args.damage_window);
    
    // Precompute feature lists
    let kmers_list = get_all_kmers(args.k);
    info!("Using {}-mer frequency analysis ({} features)", args.k, kmers_list.len());
    
    let codon_list = if args.use_codon { 
        info!("Including codon usage analysis (64 features)");
        Some(get_all_kmers(3)) 
    } else { 
        None 
    };
    
    // Calculate total features dimension for PCA
    let total_features = kmers_list.len() + codon_list.as_ref().map_or(0, |v| v.len()) + 7;  // Updated: 4 basic features + 3 new model-based features
    info!("Total feature space dimension: {}", total_features);
    
    // Define feature extractor closure
    let feature_extractor = |read: &Read| -> Vec<f64> {
        let mut features = Vec::with_capacity(total_features);
        
        // k-mer frequencies
        let freq = compute_kmer_freq(&read.seq, args.k, &kmers_list);
        features.extend(kmers_list.iter().map(|kmer| *freq.get(kmer).unwrap_or(&0.0)));
        
        // Optional codon usage
        if let Some(ref codon_list) = codon_list {
            let usage = compute_codon_usage(&read.seq);
            features.extend(codon_list.iter().map(|codon| *usage.get(codon).unwrap_or(&0.0)));
        }
        
        // Additional intrinsic features
        features.push(compute_gc_content(&read.seq));
        
        // Use our enhanced damage assessment
        let damage = compute_damage_scores(read, args.damage_window);
        
        // Add both individual end scores and the combined score
        features.push(damage.damage5);
        features.push(damage.damage3);
        features.push(damage.combined);
        
        // Add additional derived features from our models
        let cp_results = detect_damage_change_points(read, args.damage_window);
        let decay_results = fit_decay_damage_model(read, args.damage_window);
        let read_pair_results = analyze_read_pair_overlaps(read);
        
        // Add change point confidence
        features.push(1.0 - cp_results.prime5_pvalue.min(cp_results.prime3_pvalue));
        
        // Add model fit quality
        features.push(decay_results.prime5_model_fit);
        features.push(decay_results.prime3_model_fit);
        
        // Add ratio between 5' and 3' damage (damage symmetry feature)
        let damage_ratio = if damage.damage3 > 0.0 {
            damage.damage5 / damage.damage3
        } else {
            2.0 // Default when no 3' damage
        };
        features.push(damage_ratio.min(3.0).max(0.33)); // Clip to reasonable range
        
        // Add position-specific decay rate features
        features.push(decay_results.prime5_lambda);
        features.push(decay_results.prime3_lambda);
        
        // Add read pair support as a feature
        features.push(read_pair_results.pair_support_score);
        
        // Add log-likelihood ratio as a feature (normalized to [0,1] range)
        features.push((decay_results.log_likelihood_ratio.max(-5.0).min(5.0) + 5.0) / 10.0);
        
        features
    };
    
    // Incremental PCA
    info!("Performing incremental PCA (components: {}, batch size: {})...", 
          args.pca_components, args.batch_size);
    let (mean, eigenvectors, eigenvalues, pca_results) =
        incremental_pca(&reads, args.batch_size, feature_extractor, total_features, args.pca_components);
    
    info!("PCA complete. Explained variance:");
    let total_variance: f64 = eigenvalues.iter().sum();
    for (i, &val) in eigenvalues.iter().enumerate() {
        let explained = (val / total_variance) * 100.0;
        info!("  PC{}: {:.2}% (eigenvalue: {:.4})", i+1, explained, val);
    }
    
    // Epsilon selection
    let eps = if args.auto_epsilon {
        info!("Calculating optimal epsilon using k-distance method...");
        let k_distances = compute_k_distance_data(&pca_results, args.min_samples);
        // Find the elbow point in the k-distance graph
        let optimal_eps = find_optimal_eps(&k_distances);
        info!("Automatically determined epsilon: {:.4} (user provided: {:.4})", 
              optimal_eps, args.eps);
        optimal_eps
    } else {
        info!("Using user-specified epsilon: {:.4}", args.eps);
        args.eps
    };
    
    // Dynamic dispatch for DBSCAN based on dimensions
    info!("Clustering reads with DBSCAN (eps: {:.4}, min_samples: {})...", 
          eps, args.min_samples);
    
          let clusters = match args.pca_components {
            2 => dbscan_clustering::<2>(&pca_results, eps, args.min_samples),
            3 => dbscan_clustering::<3>(&pca_results, eps, args.min_samples),
            4 => dbscan_clustering::<4>(&pca_results, eps, args.min_samples),
            5 => dbscan_clustering::<5>(&pca_results, eps, args.min_samples),
            10 => dbscan_clustering::<10>(&pca_results, eps, args.min_samples),
            n => {
                // Default: use first 5 or 10 components based on available dimension
                let max_dim = if n > 10 { 10 } else { 5 };
                info!("PCA dimension {} not directly supported for clustering. Using first {} components.", n, max_dim);
                let truncated_data = truncate_pca_dimensions(&pca_results, max_dim);
                if max_dim == 10 {
                    dbscan_clustering::<10>(&truncated_data, eps, args.min_samples)
                } else {
                    dbscan_clustering::<5>(&truncated_data, eps, args.min_samples)
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

