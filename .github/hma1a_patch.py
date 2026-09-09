from pathlib import Path

BASE_SCHEMA = "mer.gpu-native-source-to-upload-copy-elision-production.v2"
NEW_SCHEMA = "mer.gpu-native-source-to-upload-copy-elision-production.v3"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected 1 match, found {count}")
    return text.replace(old, new, 1)


io = Path("rust-engine/src/io_provider.rs")
s = io.read_text()

s = replace_once(
    s,
    "use std::sync::Arc;",
    "use std::sync::{Arc, Weak};",
    "Arc/Weak import",
)

s = replace_once(
    s,
    """    files: Mutex<LruCache<u32, Arc<File>>>,
    /// Optional multi-drive layout.""",
    """    files: Mutex<LruCache<u32, Arc<File>>>,
    /// Source/upload-only proof cache. Each proof belongs to one exact
    /// opened `Arc<File>` object, never merely an expert id or raw fd number.
    /// An fd-cache eviction/reopen makes the Weak identity stale and forces
    /// the replacement file to prove O_DIRECT + full-file length again.
    source_upload_fd_proofs: Mutex<HashMap<u32, Weak<File>>>,
    source_upload_fd_proof_hits: AtomicU64,
    source_upload_fd_proof_misses: AtomicU64,
    /// Optional multi-drive layout.""",
    "NvmeStorage proof fields",
)

s = replace_once(
    s,
    """            files: Mutex::new(LruCache::new(
                NonZeroUsize::new(default_fd_cache_cap())
                    .expect("default_fd_cache_cap() is clamped to >= 64"),
            )),
            striped_paths: Vec::new(),""",
    """            files: Mutex::new(LruCache::new(
                NonZeroUsize::new(default_fd_cache_cap())
                    .expect("default_fd_cache_cap() is clamped to >= 64"),
            )),
            source_upload_fd_proofs: Mutex::new(HashMap::new()),
            source_upload_fd_proof_hits: AtomicU64::new(0),
            source_upload_fd_proof_misses: AtomicU64::new(0),
            striped_paths: Vec::new(),""",
    "NvmeStorage proof init",
)

marker = """    /// Per-expert circuit-breaker state (gist Task 3). Lazily
"""
if s.count(marker) != 1:
    raise SystemExit(f"source-upload proof helper marker matches={s.count(marker)}")

helper = r'''    /// Resolve the exact cached file used by source/upload and prove its
    /// direct-I/O contract once per opened `Arc<File>` object. A Weak proof
    /// cannot survive fd-cache eviction/reopen, and `Arc::ptr_eq` prevents
    /// raw-fd-number reuse from aliasing a stale proof.
    fn source_upload_fd_for(&self, id: u32) -> io::Result<Arc<File>> {
        let file = self.fd_for(id)?;
        let cached = self
            .source_upload_fd_proofs
            .lock()
            .get(&id)
            .and_then(Weak::upgrade)
            .is_some_and(|proven| Arc::ptr_eq(&proven, &file));
        if cached {
            self.source_upload_fd_proof_hits
                .fetch_add(1, Ordering::Relaxed);
            return Ok(file);
        }

        #[cfg(target_os = "linux")]
        {
            // SAFETY: `file` is the live Arc<File> handed to the read/retry
            // scheduler below; F_GETFL has no third argument.
            let flags = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETFL) };
            if flags < 0 {
                return Err(io::Error::last_os_error());
            }
            if flags & libc::O_DIRECT == 0 || file.metadata()?.len() != self.cfg.expert_size as u64
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "source/upload source fd must be O_DIRECT and exactly one full expert",
                ));
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "source/upload direct source requires Linux O_DIRECT",
            ));
        }

        self.source_upload_fd_proofs
            .lock()
            .insert(id, Arc::downgrade(&file));
        self.source_upload_fd_proof_misses
            .fetch_add(1, Ordering::Relaxed);
        Ok(file)
    }

    pub(crate) fn source_upload_fd_proof_snapshot(&self) -> (u64, u64) {
        (
            self.source_upload_fd_proof_hits.load(Ordering::Relaxed),
            self.source_upload_fd_proof_misses.load(Ordering::Relaxed),
        )
    }

'''
s = s.replace(marker, helper + marker, 1)

fn_start = s.index("    pub(crate) async fn read_experts_batch_into_aligned_slices(")
loop_start = s.index(
    "        let mut files: Vec<Arc<File>> = Vec::with_capacity(ids.len());",
    fn_start,
)
id_vec = s.index("        let id_vec: Vec<u32> = ids.to_vec();", loop_start)
old_block = s[loop_start:id_vec]
expected_block = '''        let mut files: Vec<Arc<File>> = Vec::with_capacity(ids.len());
        for &id in ids {
            files.push(self.fd_for(id)?);
        }
        // Prove O_DIRECT on the actual Arc<File> passed to the retry helper;
        // cache churn cannot substitute a different fd after this check.
        for file in &files {
            #[cfg(target_os = "linux")]
            {
                // SAFETY: files owns this live fd; F_GETFL has no third argument.
                let flags = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETFL) };
                if flags < 0 {
                    return Err(io::Error::last_os_error());
                }
                if flags & libc::O_DIRECT == 0
                    || file.metadata()?.len() != self.cfg.expert_size as u64
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "source/upload source fd must be O_DIRECT and exactly one full expert",
                    ));
                }
            }
            #[cfg(not(target_os = "linux"))]
            {
                let _ = file;
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "source/upload direct source requires Linux O_DIRECT",
                ));
            }
        }
'''
if old_block != expected_block:
    raise SystemExit("source-upload fd proof hot block changed from frozen main")
replacement = '''        let mut files: Vec<Arc<File>> = Vec::with_capacity(ids.len());
        for &id in ids {
            files.push(self.source_upload_fd_for(id)?);
        }
'''
s = s[:loop_start] + replacement + s[id_vec:]

if "mod hma1a_source_upload_fd_proof_tests" in s:
    raise SystemExit("HMA-1A structural test already present")
s += r'''
#[cfg(test)]
mod hma1a_source_upload_fd_proof_tests {
    #[test]
    fn source_upload_fd_proof_is_object_identity_cached_and_hot_read_has_no_reproof_syscalls() {
        let source = include_str!("io_provider.rs");
        let helper = source
            .split("fn source_upload_fd_for")
            .nth(1)
            .expect("source-upload fd proof helper")
            .split("/// Per-expert circuit-breaker state")
            .next()
            .unwrap();
        assert!(helper.contains("Weak::upgrade"));
        assert!(helper.contains("Arc::ptr_eq"));
        assert!(helper.contains("libc::F_GETFL"));
        assert!(helper.contains("file.metadata()?.len()"));

        let hot = source
            .split("pub(crate) async fn read_experts_batch_into_aligned_slices")
            .nth(1)
            .expect("source-upload batch reader")
            .split("/// **Tier 2.** Packed-blob sibling")
            .next()
            .unwrap();
        assert!(hot.contains("self.source_upload_fd_for(id)?"));
        assert!(!hot.contains("libc::F_GETFL"));
        assert!(!hot.contains("file.metadata()?.len()"));

        let control = source
            .split("pub async fn read_experts_batch(")
            .nth(1)
            .expect("ordinary batch reader")
            .split("/// Source/upload external destinations")
            .next()
            .unwrap();
        assert!(!control.contains("source_upload_fd_for"));
    }
}
'''
io.write_text(s)

upload = Path("rust-engine/src/gpu_native_source_upload.rs")
u = upload.read_text()

u = replace_once(
    u,
    """    pub(crate) direct_payload_bytes: u64,
    pub(crate) source_failures: u64,""",
    """    pub(crate) direct_payload_bytes: u64,
    pub(crate) fd_proof_cache_hits: u64,
    pub(crate) fd_proof_cache_misses: u64,
    pub(crate) source_failures: u64,""",
    "upload proof metrics fields",
)

u = replace_once(
    u,
    """        let started = Instant::now();
        let result = storage
            .read_experts_batch_into_aligned_slices(ids, &mut destinations)
            .await;
        self.add(|m| &mut m.fused_source_us, elapsed(started));""",
    """        let (proof_hits_before, proof_misses_before) = storage.source_upload_fd_proof_snapshot();
        let started = Instant::now();
        let result = storage
            .read_experts_batch_into_aligned_slices(ids, &mut destinations)
            .await;
        self.add(|m| &mut m.fused_source_us, elapsed(started));
        let (proof_hits_after, proof_misses_after) = storage.source_upload_fd_proof_snapshot();
        self.add(
            |m| &mut m.fd_proof_cache_hits,
            proof_hits_after.saturating_sub(proof_hits_before),
        );
        self.add(
            |m| &mut m.fd_proof_cache_misses,
            proof_misses_after.saturating_sub(proof_misses_before),
        );""",
    "read_source proof accounting",
)
upload.write_text(u)

qual = Path("rust-engine/src/gpu_native_source_to_upload_copy_elision_production.rs")
q = qual.read_text()

q = replace_once(
    q,
    f'pub(crate) const SCHEMA: &str = "{BASE_SCHEMA}";',
    f'pub(crate) const SCHEMA: &str = "{NEW_SCHEMA}";',
    "qualifier schema",
)

q = replace_once(
    q,
    """    treatment_every_nvme_read_fused: bool,
    ring_and_submission_accounting_exact: bool,""",
    """    treatment_every_nvme_read_fused: bool,
    source_upload_fd_proof_cache_exact: bool,
    ring_and_submission_accounting_exact: bool,""",
    "qualifier proof gate field",
)

q = replace_once(
    q,
    """        c.fallback_installs,
        c.fallback_payload_copy_bytes,
    ]""",
    """        c.fallback_installs,
        c.fallback_payload_copy_bytes,
        c.fd_proof_cache_hits,
        c.fd_proof_cache_misses,
    ]""",
    "control proof counters zero",
)

q = replace_once(
    q,
    """    let ring_and_submission_accounting_exact = ring_exact(cu, tu);
    let production_upload_ownership_exact = !cu.production_owned && tu.production_owned;""",
    """    let source_upload_fd_proof_cache_exact = cm.fd_proof_cache_hits == 0
        && cm.fd_proof_cache_misses == 0
        && tm.fd_proof_cache_hits.checked_add(tm.fd_proof_cache_misses)
            == Some(tm.direct_source_reads);
    let ring_and_submission_accounting_exact = ring_exact(cu, tu);
    let production_upload_ownership_exact = !cu.production_owned && tu.production_owned;""",
    "qualifier proof gate calculation",
)

q = replace_once(
    q,
    """        && treatment_every_nvme_read_fused
        && ring_and_submission_accounting_exact""",
    """        && treatment_every_nvme_read_fused
        && source_upload_fd_proof_cache_exact
        && ring_and_submission_accounting_exact""",
    "qualifier proof gate pass chain",
)

q = replace_once(
    q,
    """        treatment_every_nvme_read_fused,
        ring_and_submission_accounting_exact,""",
    """        treatment_every_nvme_read_fused,
        source_upload_fd_proof_cache_exact,
        ring_and_submission_accounting_exact,""",
    "qualifier proof gate construction",
)

qual.write_text(q)
print("HMA1A_PATCH_RESULT=PASS")
