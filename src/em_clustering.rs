//! EM Iterative Clustering Module for MADERA v0.4.0
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

impl EMClusterer {
    /// Create a new EMClusterer from initial cluster assignments.
    ///
    /// `initial_assignments`: cluster label per read (from Phase 1 DBSCAN).
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
        let valid_clusters: Vec<isize> = cluster_sizes.iter()
            .filter(|(_, &size)| size >= min_cluster_size)
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

    /// Merge clusters whose k-mer profiles are too similar.
    ///
    /// Uses Jensen-Shannon divergence as the similarity metric.
    /// When two clusters are below the threshold, the smaller one is
    /// absorbed into the larger one. Profiles and assignments are
    /// updated accordingly.
    fn merge_similar_clusters(
        &mut self,
        read_kmer_counts: &[Vec<u16>],
        gc_values: &[f64],
        threshold: f64,
    ) {
        if self.profiles.len() <= 1 {
            return;
        }

        let n = self.profiles.len();
        let mut merge_into: Vec<usize> = (0..n).collect(); // Union-find parent

        // Compute pairwise JSD and mark merges
        for i in 0..n {
            for j in (i + 1)..n {
                let jsd = jensen_shannon_divergence(
                    &self.profiles[i].kmer_probs,
                    &self.profiles[j].kmer_probs,
                );

                if jsd < threshold {
                    // Merge smaller into larger
                    let (keep, dissolve) = if self.profiles[i].read_count >= self.profiles[j].read_count {
                        (i, j)
                    } else {
                        (j, i)
                    };

                    // Follow chain to find root
                    let root_keep = Self::find_root(&merge_into, keep);
                    let root_dissolve = Self::find_root(&merge_into, dissolve);
                    if root_keep != root_dissolve {
                        merge_into[root_dissolve] = root_keep;
                    }
                }
            }
        }

        // Check if any merges happened
        let any_merges = merge_into.iter().enumerate().any(|(i, &p)| p != i);
        if !any_merges {
            return;
        }

        // Resolve all roots
        let roots: Vec<usize> = (0..n).map(|i| Self::find_root(&merge_into, i)).collect();

        // Remap to contiguous indices
        let unique_roots: Vec<usize> = {
            let mut seen = std::collections::HashSet::new();
            roots.iter().filter(|&&r| seen.insert(r)).copied().collect()
        };
        let root_to_new: std::collections::HashMap<usize, usize> = unique_roots.iter()
            .enumerate()
            .map(|(new_idx, &root)| (root, new_idx))
            .collect();

        let merged_count = n - unique_roots.len();
        if merged_count > 0 {
            info!("EM: Merged {} clusters (JSD < {:.4}), {} → {} clusters",
                  merged_count, threshold, n, unique_roots.len());
        }

        // Remap assignments
        self.assignments = self.assignments.iter()
            .map(|&c| {
                let root = roots[c];
                *root_to_new.get(&root).unwrap_or(&0)
            })
            .collect();

        // Rebuild profiles from merged assignments
        let new_n = unique_roots.len();
        self.profiles = (0..new_n)
            .map(|c| {
                Self::build_profile_from_reads(
                    c, read_kmer_counts, gc_values, &self.assignments,
                    c, self.num_kmers, self.pseudocount,
                )
            })
            .collect();
    }

    // TODO(v0.5.0): Implement cluster splitting for bimodal detection.
    // After convergence, check each cluster for bimodality by computing the variance
    // of multinomial log-likelihoods within the cluster. If variance is significantly
    // higher than expected (estimable from the multinomial model), split the cluster
    // into two via k-means on the k=4 count vectors, then continue EM.
    // This is the split-merge EM variant (Ueda et al. 2000) and is the main thing
    // limiting clustering resolution when DBSCAN merges two organisms into one cluster.

    /// Union-find: follow parent chain to root.
    fn find_root(parents: &[usize], mut node: usize) -> usize {
        while parents[node] != node {
            node = parents[node];
        }
        node
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
        merge_threshold: f64,
    ) -> Vec<isize> {
        info!("Starting EM iterative clustering (max_iter={}, conv={}, merge_jsd={})",
              max_iterations, convergence_threshold, merge_threshold);

        for iteration in 0..max_iterations {
            // E-step
            let new_assignments = self.e_step(read_kmer_counts);

            // Count how many reads changed cluster
            let changes: usize = new_assignments.iter()
                .zip(self.assignments.iter())
                .filter(|(&new, &old)| new != old)
                .count();

            // M-step
            self.m_step(read_kmer_counts, gc_values, &new_assignments);

            // Merge similar clusters (only after iteration 3 to let profiles stabilize;
            // merging too early when profiles are noisy can permanently collapse distinct clusters)
            if iteration >= 3 {
                self.merge_similar_clusters(read_kmer_counts, gc_values, merge_threshold);
            }

            // Compute total log-likelihood
            let total_ll = self.compute_total_log_likelihood(read_kmer_counts);
            self.log_likelihood_history.push(total_ll);

            // Log progress
            info!("EM iteration {}: {} clusters, {} reads changed, total LL = {:.2}",
                  iteration + 1, self.profiles.len(), changes, total_ll);

            for profile in &self.profiles {
                info!("  Cluster {}: {} reads, GC={:.3}",
                      profile.cluster_id, profile.read_count, profile.gc_mean);
            }

            // Check convergence
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

            // Also check if no reads changed (hard convergence)
            if changes == 0 {
                info!("EM converged after {} iterations (no assignment changes)", iteration + 1);
                break;
            }
        }

        // Convert to isize for compatibility with existing pipeline
        self.assignments.iter().map(|&a| a as isize).collect()
    }
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
}
