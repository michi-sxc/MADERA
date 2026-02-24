//! Optional MEGAHIT Assembly Module for MADERA v0.4.0
//!
//! Provides lightweight assembly as a pre-step for improved clustering.
//! When enabled via `--assemble`, reads are assembled with MEGAHIT to
//! produce contigs. K=4 features on contigs (≥500bp) have much higher
//! SNR than individual short reads, enabling better cluster initialization.
//!
//! MEGAHIT is an external dependency — the user must have it installed.
//! This module shells out via `std::process::Command`.

use crate::Read;
use log::{info, error};
use std::io::{BufRead, BufReader, Write};

/// Check if MEGAHIT is available on the system.
///
/// Runs `megahit --version` and returns the version string on success,
/// or an error message if the binary is not found or fails to execute.
pub fn check_megahit_available(megahit_path: &str) -> Result<String, String> {
    match std::process::Command::new(megahit_path)
        .arg("--version")
        .output()
    {
        Ok(output) => {
            if output.status.success() {
                let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
                let version = if version.is_empty() {
                    String::from_utf8_lossy(&output.stderr).trim().to_string()
                } else {
                    version
                };
                Ok(version)
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                Err(format!("megahit returned error: {}", stderr.trim()))
            }
        }
        Err(e) => Err(format!(
            "Could not find megahit at '{}': {}. \
             Install MEGAHIT (https://github.com/voutcn/megahit) or \
             specify path with --megahit-path",
            megahit_path, e
        )),
    }
}

/// Run MEGAHIT assembly on a set of reads.
///
/// 1. Writes reads to a temporary FASTQ file
/// 2. Invokes MEGAHIT with parameters suitable for short aDNA reads:
///    `--min-count 1 --k-min 21 --k-step 10`
/// 3. Parses output contigs from `final.contigs.fa`
/// 4. Returns contigs as `Read` objects (id = contig name, seq = contig sequence)
///
/// The temporary directory is cleaned up automatically when the returned
/// `TempDir` guard is dropped — but we clean up explicitly on success.
pub fn run_megahit(
    reads: &[Read],
    megahit_path: &str,
) -> Result<Vec<Read>, Box<dyn std::error::Error>> {
    // Verify megahit is available
    let version = check_megahit_available(megahit_path)?;
    info!("Using MEGAHIT: {}", version);

    // Create temp directory for input/output
    let tmp_dir = tempfile::Builder::new()
        .prefix("madera_assembly_")
        .tempdir()?;
    let tmp_path = tmp_dir.path();

    // Write reads to temp FASTQ
    let input_fq = tmp_path.join("input.fastq");
    {
        let mut fq_file = std::fs::File::create(&input_fq)?;
        for read in reads {
            writeln!(fq_file, "@{}", read.id)?;
            writeln!(fq_file, "{}", read.seq)?;
            writeln!(fq_file, "+")?;
            // Use quality if available, otherwise generate dummy quality
            if let Some(ref qual) = read.quality {
                let qual_str: String = qual.iter().map(|&q| (q + 33) as char).collect();
                writeln!(fq_file, "{}", qual_str)?;
            } else {
                let dummy_qual: String = std::iter::repeat('I').take(read.seq.len()).collect();
                writeln!(fq_file, "{}", dummy_qual)?;
            }
        }
    }
    info!("Wrote {} reads to temporary FASTQ for assembly", reads.len());

    // MEGAHIT output directory (must not exist beforehand)
    let megahit_out = tmp_path.join("megahit_out");

    // Run MEGAHIT
    info!("Running MEGAHIT assembly (this may take a while)...");
    let output = std::process::Command::new(megahit_path)
        .arg("--read")
        .arg(&input_fq)
        .arg("--min-count")
        .arg("1")
        .arg("--k-min")
        .arg("21")
        .arg("--k-step")
        .arg("10")
        .arg("--out-dir")
        .arg(&megahit_out)
        .arg("--num-cpu-threads")
        .arg(rayon::current_num_threads().to_string())
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        error!("MEGAHIT failed: {}", stderr);
        return Err(format!("MEGAHIT assembly failed: {}", stderr).into());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    info!("MEGAHIT stdout: {}", stdout.lines().last().unwrap_or("(empty)"));

    // Parse output contigs
    let contigs_path = megahit_out.join("final.contigs.fa");
    if !contigs_path.exists() {
        return Err("MEGAHIT completed but final.contigs.fa not found".into());
    }

    let contigs = parse_fasta(&contigs_path)?;
    info!("Assembly produced {} contigs", contigs.len());

    if !contigs.is_empty() {
        let lengths: Vec<usize> = contigs.iter().map(|c| c.seq.len()).collect();
        let total_bases: usize = lengths.iter().sum();
        let max_len = lengths.iter().max().unwrap_or(&0);
        let min_len = lengths.iter().min().unwrap_or(&0);
        let mean_len = total_bases as f64 / lengths.len() as f64;

        // Compute N50
        let mut sorted_lengths = lengths.clone();
        sorted_lengths.sort_unstable_by(|a, b| b.cmp(a));
        let half_total = total_bases / 2;
        let mut cumulative = 0;
        let mut n50 = 0;
        for &l in &sorted_lengths {
            cumulative += l;
            if cumulative >= half_total {
                n50 = l;
                break;
            }
        }

        info!("Contig stats: {} contigs, {}-{} bp (mean {:.0}), N50={}, total {} bp",
              contigs.len(), min_len, max_len, mean_len, n50, total_bases);
    }

    // tmp_dir is cleaned up when dropped
    Ok(contigs)
}

/// Parse a FASTA file into a vector of Read objects.
fn parse_fasta(path: &std::path::Path) -> Result<Vec<Read>, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);

    let mut contigs = Vec::new();
    let mut current_id = String::new();
    let mut current_seq = String::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if let Some(header) = line.strip_prefix('>') {
            // Save previous contig
            if !current_id.is_empty() && !current_seq.is_empty() {
                contigs.push(Read {
                    id: current_id.clone(),
                    seq: current_seq.clone(),
                    quality: None,
                });
            }
            current_id = header.split_whitespace().next().unwrap_or(header).to_string();
            current_seq.clear();
        } else {
            current_seq.push_str(line);
        }
    }

    // Don't forget the last contig
    if !current_id.is_empty() && !current_seq.is_empty() {
        contigs.push(Read {
            id: current_id,
            seq: current_seq,
            quality: None,
        });
    }

    Ok(contigs)
}
