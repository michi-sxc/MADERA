//! EM Iterative Clustering Module for MADERA v0.4.1
//!
//! Implements expectation-maximization based clustering using multinomial
//! k-mer profiles. The EM approach correctly handles the noise inherent in
//! short-read k-mer counts by treating each read as a sample from a
//! multinomial distribution parameterized by the cluster's k-mer profile.
//!
//! Key components:
//! - `ClusterProfile`: Normalized k-mer probability vector for a cluster
//! - `EMClusterer`: Orchestrates the EM loop (E-step, M-step, merge, converge)
//! - Multinomial log-likelihood for read-to-cluster assignment
//! - Jensen-Shannon divergence for cluster merging

use log::{info, warn};
use rayon::prelude::*;

/// K-mer profile for a single cluster.
///
/// Stores the normalized multinomial probability vector derived from
/// aggregated k-mer counts across all reads assigned to this cluster.
/// Log-probabilities are pre-computed for fast likelihood evaluation.
#[derive(Debug, Clone)]
pub struct ClusterProfile {
    pub cluster_id: usize,
    /// Normalized probabilities (sum to 1.0), length = num_kmers
    pub kmer_probs: Vec<f64>,
    /// Pre-computed ln(p_i) for fast log-likelihood, length = num_kmers
    pub log_probs: Vec<f64>,
    /// Number of reads currently assigned to this cluster
    pub read_count: usize,
    /// Mean GC content of assigned reads
    pub gc_mean: f64,
}

/// EM clustering state machine.
///
/// Holds the current cluster profiles, read assignments, and convergence
/// history. The `run` method executes the full EM loop.
pub struct EMClusterer {
    pub profiles: Vec<ClusterProfile>,
    /// Read index → cluster index (indexes into `profiles`)
    pub assignments: Vec<usize>,
    /// Total log-likelihood at each iteration (for convergence monitoring)
    pub log_likelihood_history: Vec<f64>,
    /// K-mer size used for profiles (typically 4). Stored for diagnostics.
    #[allow(dead_code)]
    pub k: usize,
    /// Number of distinct k-mers = 4^k
    pub num_kmers: usize,
    /// Laplace smoothing pseudocount (default 0.5 = Jeffreys prior)
    pub pseudocount: f64,
}

/// Compute raw k-mer counts for a sequence using direct index encoding.
///
/// Instead of HashMap-based counting, encodes each k-mer as an integer index:
/// index = Σ base_value[i] × 4^(k-1-i), where A=0, C=1, G=2, T=3.
///
/// Returns a Vec<u16> of length 4^k. For k=4, max possible count per k-mer
/// in a read of length L is L-3 (e.g., 147 for L=150), which fits in u16.
///
/// Bases not in {A,C,G,T} (e.g., N) cause the current k-mer window to be
/// skipped, matching the behavior of the existing `compute_kmer_freq`.
pub fn compute_kmer_counts(seq: &[u8], k: usize, num_kmers: usize) -> Vec<u16> {
    let mut counts = vec![0u16; num_kmers];

    if seq.len() < k {
        return counts;
    }

    let total_windows = seq.len() - k + 1;
    for i in 0..total_windows {
        let mut index: usize = 0;
        let mut valid = true;

        for j in 0..k {
            let base_val = match seq[i + j] {
                b'A' | b'a' => 0,
                b'C' | b'c' => 1,
                b'G' | b'g' => 2,
                b'T' | b't' => 3,
                _ => {
                    valid = false;
                    break;
                }
            };
            index = index * 4 + base_val;
        }

        if valid {
            // Saturating add to prevent overflow on extremely long sequences
            counts[index] = counts[index].saturating_add(1);
        }
    }

    counts
}

/// Multinomial log-likelihood of a read (given by its k-mer counts)
/// under a cluster profile (given by its log-probabilities).
///
/// log L(cluster | read) = Σ_i  counts[i] × log_probs[i]
///
/// This is a simple dot product. Zero counts contribute 0 (0 × log(p) = 0),
/// so sparsity is handled naturally. The log-probabilities already incorporate
/// Laplace smoothing, so log(0) never occurs.
#[inline]
pub fn multinomial_log_likelihood(counts: &[u16], log_probs: &[f64]) -> f64 {
    debug_assert_eq!(counts.len(), log_probs.len());
    let mut ll = 0.0;
    for i in 0..counts.len() {
        if counts[i] > 0 {
            ll += counts[i] as f64 * log_probs[i];
        }
    }
    ll
}

/// Jensen-Shannon divergence between two probability distributions.
///
/// JSD(p, q) = 0.5 × KL(p || m) + 0.5 × KL(q || m)  where m = 0.5(p + q)
///
/// Returns a value in [0, ln(2)]. JSD = 0 means identical distributions.
/// Used for deciding whether to merge two clusters.
pub fn jensen_shannon_divergence(p: &[f64], q: &[f64]) -> f64 {
    debug_assert_eq!(p.len(), q.len());
    let mut jsd = 0.0;

    for i in 0..p.len() {
        let m_i = 0.5 * (p[i] + q[i]);
        if m_i > 0.0 {
            if p[i] > 0.0 {
                jsd += 0.5 * p[i] * (p[i] / m_i).ln();
            }
            if q[i] > 0.0 {
                jsd += 0.5 * q[i] * (q[i] / m_i).ln();
            }
        }
    }

    // Clamp to valid range (floating-point imprecision can produce tiny negatives)
    jsd.max(0.0)
}

pub fn compute_bic(total_ll: f64, n_clusters: usize, num_kmers: usize, n_reads: usize) -> f64 {
    let n_params = n_clusters * (num_kmers - 1) + (n_clusters - 1);
    -2.0 * total_ll + (n_params as f64) * (n_reads as f64).ln()
}

impl EMClusterer {
    /// Create a new EMClusterer from initial cluster assignments.
    ///
    /// `initial_assignments`: cluster label per read (from Phase 1 K-means++).
    ///   Noise points (label -1) are assigned to the nearest non-noise cluster
    ///   based on GC content proximity.
    /// `read_kmer_counts`: pre-computed k=4 count vectors per read.
    /// `gc_values`: GC content per read (parallel to read_kmer_counts).
    /// `k`: k-mer size (typically 4).
    /// `min_cluster_size`: clusters smaller than this are dissolved into noise.
    pub fn new(
        initial_assignments: &[isize],
        read_kmer_counts: &[Vec<u16>],
        gc_values: &[f64],
        k: usize,
        min_cluster_size: usize,
        pseudocount: f64,
    ) -> Self {
        let num_kmers = 4usize.pow(k as u32);
        let n_reads = initial_assignments.len();

        // Collect unique positive cluster IDs and filter small clusters
        let mut cluster_sizes: std::collections::HashMap<isize, usize> = std::collections::HashMap::new();
        for &label in initial_assignments {
            if label >= 0 {
                *cluster_sizes.entry(label).or_insert(0) += 1;
            }
        }

        // Keep only clusters above min_cluster_size
        let init_min_size = std::cmp::max(3, min_cluster_size / 10);
        let valid_clusters: Vec<isize> = cluster_sizes.iter()
            .filter(|(_, &size)| size >= init_min_size)
            .map(|(&id, _)| id)
            .collect();
        if valid_clusters.is_empty() {
            warn!("No clusters above min_cluster_size {}. Using all non-noise clusters.", min_cluster_size);
        }

        // Create mapping from old cluster IDs to new contiguous indices
        let mut id_map: std::collections::HashMap<isize, usize> = std::collections::HashMap::new();
        let mut sorted_clusters = valid_clusters.clone();
        sorted_clusters.sort();
        for (new_idx, &old_id) in sorted_clusters.iter().enumerate() {
            id_map.insert(old_id, new_idx);
        }

        let n_clusters = id_map.len().max(1); // At least 1 cluster

        // If no valid clusters, put all reads in cluster 0
        if id_map.is_empty() {
            let assignments = vec![0usize; n_reads];
            let profiles = vec![Self::build_profile_from_reads(
                0, read_kmer_counts, gc_values, &assignments, 0, num_kmers, pseudocount,
            )];
            return EMClusterer {
                profiles,
                assignments,
                log_likelihood_history: Vec::new(),
                k,
                num_kmers,
                pseudocount,
            };
        }

        // Map assignments: valid cluster → new index, noise/small → usize::MAX (unassigned)
        let mut assignments = vec![usize::MAX; n_reads];
        for (i, &label) in initial_assignments.iter().enumerate() {
            if let Some(&new_idx) = id_map.get(&label) {
                assignments[i] = new_idx;
            }
        }

        // Build initial profiles from confidently-assigned reads only
        let profiles: Vec<ClusterProfile> = (0..n_clusters)
            .map(|c| {
                Self::build_profile_from_reads(
                    c, read_kmer_counts, gc_values, &assignments, c, num_kmers, pseudocount,
                )
            })
            .collect();

        // Assign unassigned reads (noise + dissolved small clusters) using multinomial
        // log-likelihood against the initial profiles — same logic as the E-step.
        // This uses the full k-mer profile instead of GC-only distance, correctly
        // distinguishing organisms with similar GC but different compositional signatures.
        for i in 0..n_reads {
            if assignments[i] == usize::MAX {
                let mut best_cluster = 0;
                let mut best_ll = f64::NEG_INFINITY;
                for (c, profile) in profiles.iter().enumerate() {
                    let ll = multinomial_log_likelihood(&read_kmer_counts[i], &profile.log_probs);
                    if ll > best_ll {
                        best_ll = ll;
                        best_cluster = c;
                    }
                }
                assignments[i] = best_cluster;
            }
        }

        info!("EM initialized with {} clusters from {} initial assignments",
              n_clusters, n_reads);

        EMClusterer {
            profiles,
            assignments,
            log_likelihood_history: Vec::new(),
            k,
            num_kmers,
            pseudocount,
        }
    }

    /// Build a ClusterProfile by aggregating k-mer counts from all reads
    /// assigned to a given cluster.
    fn build_profile_from_reads(
        cluster_id: usize,
        read_kmer_counts: &[Vec<u16>],
        gc_values: &[f64],
        assignments: &[usize],
        target_cluster: usize,
        num_kmers: usize,
        pseudocount: f64,
    ) -> ClusterProfile {
        let mut aggregated = vec![0u64; num_kmers];
        let mut gc_sum = 0.0;
        let mut count = 0usize;

        for (i, &a) in assignments.iter().enumerate() {
            if a == target_cluster {
                for (j, &c) in read_kmer_counts[i].iter().enumerate() {
                    aggregated[j] += c as u64;
                }
                gc_sum += gc_values[i];
                count += 1;
            }
        }

        // Normalize with Laplace smoothing: p_i = (count_i + α) / (total + α × K)
        let total: u64 = aggregated.iter().sum();
        let denom = total as f64 + pseudocount * num_kmers as f64;

        let kmer_probs: Vec<f64> = aggregated.iter()
            .map(|&c| (c as f64 + pseudocount) / denom)
            .collect();

        let log_probs: Vec<f64> = kmer_probs.iter()
            .map(|&p| p.ln())
            .collect();

        let gc_mean = if count > 0 { gc_sum / count as f64 } else { 0.5 };

        ClusterProfile {
            cluster_id,
            kmer_probs,
            log_probs,
            read_count: count,
            gc_mean,
        }
    }

    fn try_bic_merge(
        &mut self,
        read_kmer_counts: &[Vec<u16>],
        gc_values: &[f64],
    ) -> bool {
        let n_clusters = self.profiles.len();
        if n_clusters <= 1 {
            return false;
        }

        let n_reads = read_kmer_counts.len();

        // Current total LL and BIC
        let current_ll = self.compute_total_log_likelihood(read_kmer_counts);
        let current_bic = compute_bic(current_ll, n_clusters, self.num_kmers, n_reads);

        // Find the most similar pair (lowest JSD)
        let mut best_i = 0;
        let mut best_j = 1;
        let mut min_jsd = f64::INFINITY;

        for i in 0..n_clusters {
            for j in (i + 1)..n_clusters {
                let jsd = jensen_shannon_divergence(
                    &self.profiles[i].kmer_probs,
                    &self.profiles[j].kmer_probs,
                );
                if jsd < min_jsd {
                    min_jsd = jsd;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        // Collect read indices for both clusters
        let reads_i: Vec<usize> = self.assignments.iter()
            .enumerate()
            .filter(|(_, &a)| a == best_i)
            .map(|(idx, _)| idx)
            .collect();
        let reads_j: Vec<usize> = self.assignments.iter()
            .enumerate()
            .filter(|(_, &a)| a == best_j)
            .map(|(idx, _)| idx)
            .collect();

        // Build merged profile from combined reads
        let mut reads_merged = reads_i.clone();
        reads_merged.extend(&reads_j);
        let merged_profile = self.build_profile_for_subset(
            &reads_merged, read_kmer_counts, gc_values,
        );

        // Compute LL change for affected reads only:
        // new_total_ll = current_ll
        //   - LL(reads_i under profile_i)  - LL(reads_j under profile_j)
        //   + LL(reads_i ∪ reads_j under merged_profile)
        let old_ll_i: f64 = reads_i.iter()
            .map(|&r| multinomial_log_likelihood(&read_kmer_counts[r], &self.profiles[best_i].log_probs))
            .sum();
        let old_ll_j: f64 = reads_j.iter()
            .map(|&r| multinomial_log_likelihood(&read_kmer_counts[r], &self.profiles[best_j].log_probs))
            .sum();
        let new_ll_merged: f64 = reads_merged.iter()
            .map(|&r| multinomial_log_likelihood(&read_kmer_counts[r], &merged_profile.log_probs))
            .sum();

        let new_total_ll = current_ll - old_ll_i - old_ll_j + new_ll_merged;
        let new_bic = compute_bic(new_total_ll, n_clusters - 1, self.num_kmers, n_reads);

        let delta_bic = new_bic - current_bic;

        if new_bic < current_bic {
            // Accept merge: reassign reads from best_j to best_i, collapse indices
            info!("BIC merge: clusters {} (n={}, GC={:.3}) + {} (n={}, GC={:.3}) → merged \
                   (JSD={:.4}, ΔBIC={:.0}, {} → {} clusters)",
                  best_i, reads_i.len(), self.profiles[best_i].gc_mean,
                  best_j, reads_j.len(), self.profiles[best_j].gc_mean,
                  min_jsd, delta_bic, n_clusters, n_clusters - 1);

            // Reassign: best_j → best_i, shift indices above best_j down by 1
            for a in self.assignments.iter_mut() {
                if *a == best_j {
                    *a = best_i;
                }
                if *a > best_j {
                    *a -= 1;
                }
            }

            // Remove the dissolved profile
            self.profiles.remove(best_j);

            // Rebuild all profiles from updated assignments
            self.profiles = (0..self.profiles.len())
                .map(|c| {
                    Self::build_profile_from_reads(
                        c, read_kmer_counts, gc_values, &self.assignments,
                        c, self.num_kmers, self.pseudocount,
                    )
                })
                .collect();

            true
        } else {
            info!("BIC merge stopped at {} clusters (best pair: {} + {}, JSD={:.4}, ΔBIC=+{:.0})",
                  n_clusters, best_i, best_j, min_jsd, delta_bic);
            false
        }
    }

    /// E-step: Assign each read to the cluster with highest multinomial
    /// log-likelihood.
    ///
    /// This is embarrassingly parallel and is the main computational cost
    /// of each EM iteration.
    fn e_step(&self, read_kmer_counts: &[Vec<u16>]) -> Vec<usize> {
        read_kmer_counts
            .par_iter()
            .map(|counts| {
                let mut best_cluster = 0;
                let mut best_ll = f64::NEG_INFINITY;

                for (c, profile) in self.profiles.iter().enumerate() {
                    let ll = multinomial_log_likelihood(counts, &profile.log_probs);
                    if ll > best_ll {
                        best_ll = ll;
                        best_cluster = c;
                    }
                }

                best_cluster
            })
            .collect()
    }

    /// M-step: Recompute cluster profiles from current read assignments.
    ///
    /// Aggregates raw k-mer counts per cluster, normalizes with Laplace
    /// smoothing, and updates log-probabilities and GC statistics.
    /// Empty clusters are removed.
    fn m_step(
        &mut self,
        read_kmer_counts: &[Vec<u16>],
        gc_values: &[f64],
        new_assignments: &[usize],
    ) {
        let n_clusters = self.profiles.len();

        // Aggregate counts per cluster
        let mut aggregated: Vec<Vec<u64>> = vec![vec![0u64; self.num_kmers]; n_clusters];
        let mut gc_sums = vec![0.0f64; n_clusters];
        let mut counts = vec![0usize; n_clusters];

        for (i, &cluster) in new_assignments.iter().enumerate() {
            if cluster < n_clusters {
                for (j, &c) in read_kmer_counts[i].iter().enumerate() {
                    aggregated[cluster][j] += c as u64;
                }
                gc_sums[cluster] += gc_values[i];
                counts[cluster] += 1;
            }
        }

        // Rebuild profiles, removing empty clusters
        let mut new_profiles = Vec::new();
        let mut old_to_new: Vec<Option<usize>> = vec![None; n_clusters];

        for c in 0..n_clusters {
            if counts[c] == 0 {
                warn!("EM: Cluster {} became empty during M-step, removing.", c);
                continue;
            }

            let new_idx = new_profiles.len();
            old_to_new[c] = Some(new_idx);

            let total: u64 = aggregated[c].iter().sum();
            let denom = total as f64 + self.pseudocount * self.num_kmers as f64;

            let kmer_probs: Vec<f64> = aggregated[c].iter()
                .map(|&v| (v as f64 + self.pseudocount) / denom)
                .collect();

            let log_probs: Vec<f64> = kmer_probs.iter()
                .map(|&p| p.ln())
                .collect();

            let gc_mean = gc_sums[c] / counts[c] as f64;

            new_profiles.push(ClusterProfile {
                cluster_id: new_idx,
                kmer_probs,
                log_probs,
                read_count: counts[c],
                gc_mean,
            });
        }

        // Remap assignments to new indices
        self.assignments = new_assignments.iter()
            .map(|&c| {
                old_to_new.get(c)
                    .and_then(|o| *o)
                    .unwrap_or(0) // Fallback: shouldn't happen since empty clusters had no reads
            })
            .collect();

        self.profiles = new_profiles;
    }


    /// Build a ClusterProfile from a subset of read indices.
    fn build_profile_for_subset(
        &self,
        read_indices: &[usize],
        read_kmer_counts: &[Vec<u16>],
        gc_values: &[f64],
    ) -> ClusterProfile {
        let mut aggregated = vec![0u64; self.num_kmers];
        let mut gc_sum = 0.0f64;

        for &i in read_indices {
            for (j, &c) in read_kmer_counts[i].iter().enumerate() {
                aggregated[j] += c as u64;
            }
            gc_sum += gc_values[i];
        }

        let total: u64 = aggregated.iter().sum();
        let denom = total as f64 + self.pseudocount * self.num_kmers as f64;

        let kmer_probs: Vec<f64> = aggregated.iter()
            .map(|&v| (v as f64 + self.pseudocount) / denom)
            .collect();
        let log_probs: Vec<f64> = kmer_probs.iter()
            .map(|&p| p.ln())
            .collect();

        ClusterProfile {
            cluster_id: 0, // Will be set by caller
            kmer_probs,
            log_probs,
            read_count: read_indices.len(),
            gc_mean: if read_indices.is_empty() { 0.0 } else { gc_sum / read_indices.len() as f64 },
        }
    }

    
    /// Compute the total log-likelihood of all reads under their current
    /// cluster assignments.
    fn compute_total_log_likelihood(&self, read_kmer_counts: &[Vec<u16>]) -> f64 {
        read_kmer_counts
            .par_iter()
            .enumerate()
            .map(|(i, counts)| {
                let cluster = self.assignments[i];
                if cluster < self.profiles.len() {
                    multinomial_log_likelihood(counts, &self.profiles[cluster].log_probs)
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Run the full EM loop.
    ///
    /// 1. E-step: Reassign reads to highest-likelihood cluster
    /// 2. M-step: Recompute cluster profiles from assignments
    /// 3. Merge similar clusters (JSD < merge_threshold)
    /// 4. Compute total log-likelihood
    /// 5. Check convergence: |ΔLL / LL| < convergence_threshold
    ///
    /// Returns final cluster assignments as Vec<isize> (with noise as -1,
    /// though EM typically doesn't produce noise points).
    pub fn run(
        &mut self,
        read_kmer_counts: &[Vec<u16>],
        gc_values: &[f64],
        max_iterations: usize,
        convergence_threshold: f64,
        _merge_threshold: f64, // Deprecated: BIC replaces threshold-based merge
    ) -> Vec<isize> {
        let min_iterations: usize = 5;

        // Compute initial BIC for logging
        let init_ll = self.compute_total_log_likelihood(read_kmer_counts);
        let init_bic = compute_bic(init_ll, self.profiles.len(), self.num_kmers, read_kmer_counts.len());
        info!("Starting EM with BIC-guided merge (max_iter={}, conv={}, min_iter={})",
              max_iterations, convergence_threshold, min_iterations);
        info!("Initial state: {} clusters, LL={:.2}, BIC={:.0}",
              self.profiles.len(), init_ll, init_bic);

        for iteration in 0..max_iterations {
            // === E-step ===
            let new_assignments = self.e_step(read_kmer_counts);

            let changes: usize = new_assignments.iter()
                .zip(self.assignments.iter())
                .filter(|(&new, &old)| new != old)
                .count();

            // === M-step ===
            self.m_step(read_kmer_counts, gc_values, &new_assignments);

            // === BIC-guided merge: greedily merge until no improvement ===
            let pre_merge_count = self.profiles.len();
            let mut merges_this_iter = 0;
            while self.try_bic_merge(read_kmer_counts, gc_values) {
                merges_this_iter += 1;
            }
            let post_merge_count = self.profiles.len();

            // === Compute log-likelihood and BIC ===
            let total_ll = self.compute_total_log_likelihood(read_kmer_counts);
            let bic = compute_bic(total_ll, self.profiles.len(), self.num_kmers, read_kmer_counts.len());
            self.log_likelihood_history.push(total_ll);

            // === Log progress ===
            let merge_note = if merges_this_iter > 0 {
                format!(" (merged {} → {})", pre_merge_count, post_merge_count)
            } else {
                String::new()
            };
            info!("EM iteration {}: {} clusters{}, {} reads changed, LL={:.2}, BIC={:.0}",
                  iteration + 1, self.profiles.len(), merge_note, changes, total_ll, bic);

            for profile in &self.profiles {
                info!("  Cluster {}: {} reads, GC={:.3}",
                      profile.cluster_id, profile.read_count, profile.gc_mean);
            }

            // === Convergence check (only after min_iterations and no merges) ===
            if merges_this_iter == 0 && iteration + 1 >= min_iterations {
                if changes == 0 {
                    info!("EM converged after {} iterations (no assignment changes)", iteration + 1);
                    break;
                }

                if self.log_likelihood_history.len() >= 2 {
                    let prev_ll = self.log_likelihood_history[self.log_likelihood_history.len() - 2];
                    let delta = (total_ll - prev_ll).abs();
                    let relative_delta = if total_ll.abs() > 1e-10 {
                        delta / total_ll.abs()
                    } else {
                        delta
                    };

                    if relative_delta < convergence_threshold {
                        info!("EM converged after {} iterations (relative ΔLL = {:.6})",
                              iteration + 1, relative_delta);
                        break;
                    }
                }
            }
        }

        // Final summary
        let final_ll = self.compute_total_log_likelihood(read_kmer_counts);
        let final_bic = compute_bic(final_ll, self.profiles.len(), self.num_kmers, read_kmer_counts.len());
        info!("EM complete: {} clusters, LL={:.2}, BIC={:.0} (started at BIC={:.0})",
              self.profiles.len(), final_ll, final_bic, init_bic);

        self.assignments.iter().map(|&a| a as isize).collect()
    }
}

/// Run 2-means (K-means with k=2) on a subset of k-mer count vectors.
///
/// Uses K-means++ initialization: first centroid chosen as read 0,
/// second chosen as the farthest point from the first.
/// Returns assignments: 0 or 1 for each read.
pub fn two_means_split(kmer_counts: &[Vec<u16>], max_iter: usize) -> Vec<usize> {
    let n = kmer_counts.len();
    if n < 2 {
        return vec![0; n];
    }

    let dim = kmer_counts[0].len();

    // K-means++ initialization
    // Centroid 0: first read's counts (converted to f64)
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(2);
    centroids.push(kmer_counts[0].iter().map(|&c| c as f64).collect());

    // Centroid 1: farthest point from centroid 0 (deterministic K-means++ variant)
    let mut max_dist = 0.0f64;
    let mut farthest_idx = 1;
    for i in 1..n {
        let dist: f64 = kmer_counts[i].iter().zip(centroids[0].iter())
            .map(|(&a, &b)| (a as f64 - b).powi(2))
            .sum();
        if dist > max_dist {
            max_dist = dist;
            farthest_idx = i;
        }
    }
    centroids.push(kmer_counts[farthest_idx].iter().map(|&c| c as f64).collect());

    // K-means iterations
    let mut assignments = vec![0usize; n];

    for _iter in 0..max_iter {
        // Assign each point to nearest centroid
        let new_assignments: Vec<usize> = (0..n)
            .map(|i| {
                let d0: f64 = kmer_counts[i].iter().zip(centroids[0].iter())
                    .map(|(&a, &b)| (a as f64 - b).powi(2))
                    .sum();
                let d1: f64 = kmer_counts[i].iter().zip(centroids[1].iter())
                    .map(|(&a, &b)| (a as f64 - b).powi(2))
                    .sum();
                if d0 <= d1 { 0 } else { 1 }
            })
            .collect();

        if new_assignments == assignments {
            break;
        }
        assignments = new_assignments;

        // Recompute centroids
        for c in 0..2 {
            let mut sum = vec![0.0f64; dim];
            let mut count = 0usize;
            for i in 0..n {
                if assignments[i] == c {
                    for j in 0..dim {
                        sum[j] += kmer_counts[i][j] as f64;
                    }
                    count += 1;
                }
            }
            if count > 0 {
                centroids[c] = sum.iter().map(|&s| s / count as f64).collect();
            }
        }
    }

    assignments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_kmer_counts_basic() {
        let seq = b"ACGT";
        let counts = compute_kmer_counts(seq, 2, 16);
        // AC=0*4+1=1, CG=1*4+2=6, GT=2*4+3=11
        assert_eq!(counts[1], 1);  // AC
        assert_eq!(counts[6], 1);  // CG
        assert_eq!(counts[11], 1); // GT
        assert_eq!(counts.iter().map(|&c| c as u32).sum::<u32>(), 3);
    }

    #[test]
    fn test_compute_kmer_counts_with_n() {
        let seq = b"ACNGT";
        let counts = compute_kmer_counts(seq, 2, 16);
        // AC=1, CN=skip, NG=skip, GT=1
        assert_eq!(counts[1], 1);  // AC
        assert_eq!(counts[11], 1); // GT
        assert_eq!(counts.iter().map(|&c| c as u32).sum::<u32>(), 2);
    }

    #[test]
    fn test_multinomial_log_likelihood() {
        let counts = vec![10u16, 5, 0, 3];
        // let log_probs = vec![(-1.0f64).ln(), (-2.0f64).ln(), (-3.0f64).ln(), (-1.5f64).ln()];
        // log_probs should always be ln of probabilities, which are negative since p < 1
        let probs: Vec<f64> = vec![0.4, 0.3, 0.1, 0.2];
        let log_probs: Vec<f64> = probs.iter().map(|&p| p.ln()).collect();      
        let ll = multinomial_log_likelihood(&counts, &log_probs);
        let expected = 10.0 * (0.4f64).ln() + 5.0 * (0.3f64).ln() + 0.0 + 3.0 * (0.2f64).ln();
        assert!((ll - expected).abs() < 1e-10);
    }

    #[test]
    fn test_jensen_shannon_divergence_identical() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let q = vec![0.25, 0.25, 0.25, 0.25];
        let jsd = jensen_shannon_divergence(&p, &q);
        assert!(jsd < 1e-10);
    }

    #[test]
    fn test_jensen_shannon_divergence_different() {
        let p = vec![0.9, 0.1, 0.0, 0.0];
        let q = vec![0.0, 0.0, 0.1, 0.9];
        let jsd = jensen_shannon_divergence(&p, &q);
        // Should be close to ln(2) ≈ 0.693 for maximally different distributions
        assert!(jsd > 0.5);
        assert!(jsd <= 2.0f64.ln() + 1e-10);
    }

    #[test]
    fn test_jensen_shannon_divergence_symmetric() {
        let p = vec![0.6, 0.2, 0.1, 0.1];
        let q = vec![0.1, 0.1, 0.2, 0.6];
        let jsd_pq = jensen_shannon_divergence(&p, &q);
        let jsd_qp = jensen_shannon_divergence(&q, &p);
        assert!((jsd_pq - jsd_qp).abs() < 1e-10);
    }

    #[test]
    fn test_gc_variance_detects_mixed_cluster() {
        // Simulate a cluster mixing M. tuberculosis (GC=0.66) with E. coli (GC=0.51)
        let gc_values: Vec<f64> = (0..100).map(|i| {
            if i < 50 { 0.66 } else { 0.51 }
        }).collect();
        let mean: f64 = gc_values.iter().sum::<f64>() / gc_values.len() as f64;
        let variance = gc_values.iter().map(|&gc| (gc - mean).powi(2)).sum::<f64>() / gc_values.len() as f64;
        let std_dev = variance.sqrt();
        // std_dev should be ~0.075, well above the 0.03 threshold
        assert!(std_dev > 0.03, "GC std_dev {} should exceed 0.03 split threshold", std_dev);
    }

    #[test]
    fn test_gc_variance_homogeneous_cluster() {
        // Simulate a pure E. coli cluster (GC ≈ 0.51 with natural variation)
        let gc_values: Vec<f64> = (0..100).map(|i| {
            0.51 + (i as f64 * 0.0001) - 0.005 // tiny variation around 0.51
        }).collect();
        let mean: f64 = gc_values.iter().sum::<f64>() / gc_values.len() as f64;
        let variance = gc_values.iter().map(|&gc| (gc - mean).powi(2)).sum::<f64>() / gc_values.len() as f64;
        let std_dev = variance.sqrt();
        // std_dev should be << 0.03
        assert!(std_dev < 0.03, "GC std_dev {} should be below 0.03 split threshold", std_dev);
    }

    #[test]
    fn test_two_means_split_separates_distinct_profiles() {
        // Create two clearly different k-mer profiles
        let mut counts_a = vec![0u16; 16]; // k=2, 16 k-mers
        counts_a[0] = 20; counts_a[1] = 15; counts_a[2] = 10; counts_a[3] = 5;
        let mut counts_b = vec![0u16; 16];
        counts_b[12] = 20; counts_b[13] = 15; counts_b[14] = 10; counts_b[15] = 5;

        // Mix into one cluster
        let all_counts: Vec<Vec<u16>> = (0..20).map(|i| {
            if i < 10 { counts_a.clone() } else { counts_b.clone() }
        }).collect();

        // Run 2-means
        let assignments = two_means_split(&all_counts, 20);

        // First 10 should be in one cluster, last 10 in another
        let cluster_0 = assignments[0];
        for i in 1..10 {
            assert_eq!(assignments[i], cluster_0, "Read {} should be same cluster as read 0", i);
        }
        for i in 10..20 {
            assert_ne!(assignments[i], cluster_0, "Read {} should be different cluster from read 0", i);
        }
    }
}
