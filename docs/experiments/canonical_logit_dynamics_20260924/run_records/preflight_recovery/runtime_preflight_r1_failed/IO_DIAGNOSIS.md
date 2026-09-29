# Administrative diagnosis of preflight 924786 timeout

No scientific results were produced. The job's final state was TIMEOUT,
September 24 09:23:32–09:33:40 Israel, elapsed 608 GPU-allocation seconds.
The batch step terminated at09:33:41. No active queue entry remained.

Two read-only process inspections were performed inside the existing allocation
using Slurm overlap steps, not a new allocation or login-node computation:

| Inspection | Slurm step start (Israel) | Python rchar | Python read_bytes |
|---|---|---:|---:|
| 924786.1 | 09:31:32 | 911409164 | 1353195520 |
| 924786.2 | 09:32:43 | 1333197568 | 1774489600 |

The second inspection found PID2397417, the scientific preflight module,
with state `Dl`, wait channel `folio_wait_bit_common`, elapsed07:20,
CPU00:00:37, and an open descriptor for the downloaded pinned
`data/cifar100c/jpeg_compression/severity_1.parquet`.

The96-file manifest totals2,151,903,646 source bytes. Static sealed extractor
lines113–118 verify each full file and then separately hash the same files
for input_provenance, before creating preflight_cls. This explains the absence
of an output directory during those reads. Observed process read growth of
421,788,404 bytes over71seconds is roughly5.94MB/s. This is an administrative
I/O observation, not a benchmark of ViT inference or an experiment result.
At that sample rate, two full source passes (4.30GB combined) alone need about12minutes;
model import/load, cold parquet decoding, inference and gates add time.

Root first approved20min then25min extension of the running job. The first
attempt was denied by Slurm, and was not repeated; the exact receipt is in
ops/amendments/preflight_r1_924786_timelimit20.txt. The original10min cap
remained effective. Root subsequently approved a45min config-only retry
under configv4, preserving the source/method and all original gates.
