use log::{error, info, warn};
use rust_htslib::bam::{self, Read as BamRead};
use std::path::Path;
use crate::Read;

/// Read BAM file and perform qc
pub fn quality_control_bam(bam_path: &str, min_length: usize, read_filter: &str) -> Vec<Read> {
    let path = Path::new(bam_path);
    let mut reader = match bam::Reader::from_path(path) {
        Ok(reader) => reader,
        Err(e) => {
            error!("Failed to open BAM file: {}", e);
            return Vec::new();
        }
    };
    
    let mut reads = Vec::new();
    let mut total_reads = 0;
    let mut mapped_count = 0;
    let mut unmapped_count = 0;
    let mut secondary_count = 0;
    let mut supplementary_count = 0;
    let mut short_count = 0;
    let mut passed_reads = 0;
    
    info!("Reading BAM file with filter: {}", read_filter);
    
    for record_result in reader.records() {
        total_reads += 1;
        
        if let Ok(record) = record_result {
            // Track read statistics
            if record.is_unmapped() {
                unmapped_count += 1;
            } else {
                mapped_count += 1;
            }
            
            if record.is_secondary() {
                secondary_count += 1;
                continue;
            }
            
            if record.is_supplementary() {
                supplementary_count += 1;
                continue;
            }
            
            // Apply filtering based on user preference
            let is_mapped = !record.is_unmapped();
            let include_read = match read_filter {
                "unmapped" => !is_mapped,
                "mapped" => is_mapped,
                _ => true, // "all"
            };
            
            if !include_read {
                continue;
            }
            
            // Check minimum length requirement
            let seq = record.seq().as_bytes();
            if seq.len() < min_length {
                short_count += 1;
                continue;
            }
            
            // Convert BAM record to our internal Read structure
            let read = Read {
                id: String::from_utf8_lossy(record.qname()).into_owned(),
                seq: String::from_utf8_lossy(&seq).to_string(),
                quality: Some(record.qual().to_vec()),
            };
            
            reads.push(read);
            passed_reads += 1;
            
            // Log progress periodically
            if total_reads % 100000 == 0 {
                info!("Processed {} BAM records...", total_reads);
            }
        }
    }
    
    // Log detailed statistics for debugging
    info!("BAM statistics:");
    info!("  Total records: {}", total_reads);
    info!("  Mapped: {}", mapped_count);
    info!("  Unmapped: {}", unmapped_count);
    info!("  Secondary alignments: {}", secondary_count);
    info!("  Supplementary alignments: {}", supplementary_count);
    info!("  Too short: {}", short_count);
    info!("BAM QC: {} out of {} reads passed filtering and minimum length (filter: {})", 
           passed_reads, total_reads, read_filter);
    
    reads
}

/// Extract unmapped reads only from a BAM file
pub fn extract_unmapped_bam(bam_path: &str, min_length: usize) -> Vec<Read> {
    quality_control_bam(bam_path, min_length, "unmapped")
}

/// Extract mapped reads only from a BAM file
pub fn extract_mapped_bam(bam_path: &str, min_length: usize) -> Vec<Read> {
    quality_control_bam(bam_path, min_length, "mapped")
}

/// Check if a file is a valid BAM file
pub fn is_valid_bam(file_path: &str) -> bool {
    match bam::Reader::from_path(file_path) {
        Ok(_) => true,
        Err(e) => {
            warn!("BAM validation failed: {}", e);
            false
        }
    }
}

/// Detect file type based on extension and content
pub fn detect_file_type(file_path: &str) -> Result<&'static str, String> {
    use std::path::Path;
    
    let path = Path::new(file_path);
    
    // First check by extension
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("fastq") | Some("fq") => Ok("fastq"),
        Some("bam") => {
            // Verify it's a valid BAM file
            if is_valid_bam(file_path) {
                Ok("bam")
            } else {
                Err(format!("File has .bam extension but isn't a valid BAM file: {}", file_path))
            }
        },
        Some(ext) => Err(format!("Unsupported file extension: .{}", ext)),
        None => Err("File has no extension".to_string()),
    }
}