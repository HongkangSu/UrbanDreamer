# Standalone corpus performance plan

Scope: `tools/build_standalone_corpus.py` and its dedicated tests. Do not stop or
modify the currently running frozen builder; the leader owns the handover.

1. Preserve all existing receipt/manifest identities and every byte, provenance,
   raw-field, cache-hash, summary, and real-batch validation. Existing receipts
   must be fully verified on resume rather than recomputed or trusted blindly.
2. Replace Python threads with a spawn-based process pool of at most four
   workers. Set Torch, OpenMP, MKL, OpenBLAS, and NumExpr threads to one per
   worker. Submit only a bounded number of jobs at once to limit memory and
   cancel unstarted work promptly on failure.
3. Replace the compression-wide disk lock with a short reservation-ledger
   critical section. Reserve array bytes plus headers and a 16 MiB margin;
   require physical free space of at least 1 GiB plus the new reservation and
   all live outstanding reservations. Compression and full verification run
   outside the lock. Keep reservations until the temporary archive is removed
   or atomically published and any associated writes finish.
4. Persist reservation tokens with PID and Linux process-start identity to
   prevent PID reuse from retaining/releasing the wrong reservation. Recover
   only ledger records belonging to dead/replaced processes. Never delete
   another process's temporary files or any raw archive. Release own
   reservations in `finally`, including conversion and verification failures.
5. Test multiprocess admission, capacity accounting, exception release,
   stale-PID recovery, and resume validation. Retain the real three-city smoke
   evidence; run a bounded fresh process-pool smoke before recommending the
   leader resume the full corpus using a new frozen source snapshot.
