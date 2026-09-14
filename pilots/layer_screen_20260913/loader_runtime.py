"""Optional CPU-only spawn workers; preserve logical order and parent RNG use."""
import torch
from torch_geometric.loader import DataLoader as GraphLoader


def settings(workers=0):
    if workers not in (0, 2, 4):
        raise ValueError("Supported worker counts are 0, 2 and 4")
    return {"num_workers": workers, "multiprocessing_context": "spawn" if workers else None,
            "prefetch_factor": 1 if workers else None, "pin_memory": False,
            "training_persistent_workers": bool(workers), "evaluation_persistent_workers": False,
            "ordering": "FIFO; original BlockShuffleSampler is sole training-order source",
            "generator_policy": "one dedicated int64 generator draw per successful iterator creation; no global/dropout RNG"}


def cpu_worker(worker_id):
    if torch.cuda.is_initialized():
        raise RuntimeError("A data worker must never initialize CUDA")
    # Torch's worker loop also sets this. Make the intended CPU allocation explicit.
    torch.set_num_threads(1)


class EpochSeedGraphLoader(GraphLoader):
    """Normalize generator accounting for persistent versus recreated iterators.

    Torch's ordinary iterator draws one base seed, while reuse of a persistent
    iterator does not. Materialization is deterministic and has no worker RNG
    operations. Charge the same one dedicated draw to each logical iterator, so
    saving/recreating at an epoch boundary preserves the loader-generator cursor.
    The diagnostic verifies this against the pinned Torch implementation.
    """
    def __iter__(self):
        if self.generator is None:
            raise RuntimeError("Parallel loader requires an isolated generator")
        state = self.generator.get_state()
        try:
            iterator = super().__iter__()
        except BaseException:
            self.generator.set_state(state)
            raise
        self.generator.set_state(state)
        torch.empty((), dtype=torch.int64, device="cpu").random_(generator=self.generator)
        return iterator


def build(dataset, *, batch_size, sampler, generator, workers):
    if workers == 0:
        return GraphLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False,
                           num_workers=0, generator=generator)
    settings(workers)
    return EpochSeedGraphLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False,
        num_workers=workers, generator=generator, multiprocessing_context="spawn",
        persistent_workers=sampler is not None, prefetch_factor=1, pin_memory=False,
        worker_init_fn=cpu_worker)
