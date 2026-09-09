from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected 1 match, found {count}")
    return text.replace(old, new, 1)


# Update the existing source/upload scheduler-shape test to encode the intentional
# asymmetry: ordinary control resolves through fd_for; HMA-1A treatment resolves
# through source_upload_fd_for, then both retain the same block_in_place/scoped
# thread/read_at_with_retries/order shape.
io = Path("rust-engine/src/io_provider.rs")
s = io.read_text()
old = '''        for body in [production, treatment] {
            assert_eq!(body.matches("tokio::task::block_in_place(").count(), 1);
            assert_eq!(body.matches("std::thread::scope(").count(), 1);
            assert!(body.contains("scope.spawn("));
            assert!(body.contains("read_at_with_retries("));
            assert!(
                body.find("files.push(self.fd_for(id)?)").unwrap()
                    < body.find("tokio::task::block_in_place(").unwrap()
            );
            assert!(body.contains(".zip(id_vec.iter())"));
            assert!(body.contains("h.join()"));
        }
        assert!(!treatment.contains("read_expert("));
        assert!(!treatment.contains("spawn_blocking"));'''
new = '''        for (body, resolver) in [
            (production, "files.push(self.fd_for(id)?)"),
            (treatment, "files.push(self.source_upload_fd_for(id)?)"),
        ] {
            assert_eq!(body.matches("tokio::task::block_in_place(").count(), 1);
            assert_eq!(body.matches("std::thread::scope(").count(), 1);
            assert!(body.contains("scope.spawn("));
            assert!(body.contains("read_at_with_retries("));
            assert!(
                body.find(resolver).unwrap() < body.find("tokio::task::block_in_place(").unwrap()
            );
            assert!(body.contains(".zip(id_vec.iter())"));
            assert!(body.contains("h.join()"));
        }
        assert!(!production.contains("source_upload_fd_for"));
        assert!(!treatment.contains("libc::F_GETFL"));
        assert!(!treatment.contains("file.metadata()?.len()"));
        assert!(!treatment.contains("read_expert("));
        assert!(!treatment.contains("spawn_blocking"));'''
s = replace_once(s, old, new, "source-upload scheduler-shape test")
io.write_text(s)


qual = Path("rust-engine/src/gpu_native_source_to_upload_copy_elision_production.rs")
q = qual.read_text()

# The synthetic exact-treatment fixture models two direct source reads. Under
# HMA-1A those can be represented as two first-use proof misses; the gate only
# requires hits+misses to reconcile exactly with direct_source_reads because a
# measured arm after warmup can legitimately be all hits.
q = replace_once(
    q,
    '''            m.direct_source_reads = 2;
            m.direct_source_bytes = 2 * FULL as u64;''',
    '''            m.direct_source_reads = 2;
            m.fd_proof_cache_misses = 2;
            m.direct_source_bytes = 2 * FULL as u64;''',
    "treatment fixture proof misses",
)

# Explicitly prove that corrupting proof-cache accounting fails the mechanism
# gate rather than merely relying on the positive fixture.
q = replace_once(
    q,
    '''            |s| s.metrics.direct_source_reads += 1,
            |s| s.metrics.direct_source_bytes += 1,''',
    '''            |s| s.metrics.direct_source_reads += 1,
            |s| s.metrics.fd_proof_cache_hits += 1,
            |s| s.metrics.direct_source_bytes += 1,''',
    "proof counter negative mutation",
)

q = replace_once(
    q,
    '''        assert_eq!(
            SCHEMA,
            "mer.gpu-native-source-to-upload-copy-elision-production.v2"
        );''',
    '''        assert_eq!(
            SCHEMA,
            "mer.gpu-native-source-to-upload-copy-elision-production.v3"
        );''',
    "v3 schema fixture",
)
qual.write_text(q)

print("HMA1A_TEST_CONTRACT_FIX_RESULT=PASS")
