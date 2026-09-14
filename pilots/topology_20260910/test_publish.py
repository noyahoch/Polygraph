"""Offline Hub mocks; execution is restricted to a Slurm allocation."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import publish
from .protocol import atomic_json, digest, file_sha256, require_slurm

REAL_BUNDLE_SOURCE = publish.bundle_source


class MockHub:
    def __init__(self):
        self.private = True
        self.head = "0" * 40
        self.revisions = {self.head: {}}
        self.uploads = 0
        self.fail_after_document_commit = False

    def repo_info(self, repo_id, repo_type, revision=None):
        return SimpleNamespace(private=self.private, sha=revision or self.head)

    def model_info(self, repo_id, expand):
        return SimpleNamespace(usedStorage=123)

    def upload_folder(self, repo_id, repo_type, folder_path, allow_patterns, commit_message):
        files = dict(self.revisions[self.head])
        files.update({name: (Path(folder_path) / name).read_bytes() for name in allow_patterns})
        if files != self.revisions[self.head]:
            self.uploads += 1
            self.head = f"{self.uploads:040x}"
            self.revisions[self.head] = files
        if "publication.json" in allow_patterns and self.fail_after_document_commit:
            self.fail_after_document_commit = False
            raise ConnectionError("Simulated lost commit response")
        return SimpleNamespace(oid=self.head)

    def download(self, repo_id, repo_type, revision, filename, local_dir):
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.revisions[revision][filename])
        return str(target)


class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.args = SimpleNamespace(run_root=self.root / "runs", receipt_dir=self.root / "receipts",
                                    code_root=self.root / "code", cache_metadata_root=None, freeze=None,
                                    results_root=None, repo_id=publish.DEFAULT_REPO, quota_available_bytes=None)
        self.args.receipt_dir.mkdir()
        self.source_path = "pilots/topology_20260910/train.py"
        source = self.args.code_root / self.source_path
        source.parent.mkdir(parents=True)
        source.write_text("# exact restoration source\n")
        self.implementation = {self.source_path: file_sha256(source)}
        self.run = self.args.run_root / "full_graph" / "seed1"
        self.run.mkdir(parents=True)
        config = {"arm": "full_graph", "seed": 1, "preprocessing": {"kind": "none"},
                  "implementation": self.implementation, "implementation_sha256": digest(self.implementation)}
        atomic_json(self.run / "config.json", config)
        atomic_json(self.run / "history.json", [{"epoch": 1}])
        (self.run / "best.safetensors").write_bytes(b"selected checkpoint")
        (self.run / "latest.pt").write_bytes(b"trusted resume checkpoint")
        (self.run / "validation.npz").write_bytes(b"validation reference fixture")
        atomic_json(self.run / "validation.json", {"threshold": 0.5, "preprocessing": config["preprocessing"],
                    "config_sha256": file_sha256(self.run / "config.json"),
                    "best_sha256": file_sha256(self.run / "best.safetensors")})
        completion = {"complete": True, "selection_split": "val"}
        completion.update({key: file_sha256(self.run / name) for key, name in publish.BOUND_FILES.items()
                           if key != "complete_sha256"})
        completion.update(latest_sha256=file_sha256(self.run / "latest.pt"),
                          history_sha256=file_sha256(self.run / "history.json"))
        atomic_json(self.run / "complete.json", completion)
        self.bundle_patch = patch.object(publish, "bundle_source", self.fake_bundle)
        self.bundle_patch.start()
        decision = self.args.code_root / publish.DECISION_PATH
        decision.parent.mkdir(parents=True)
        decision.write_bytes((Path(publish.__file__).resolve().parents[2] / publish.DECISION_PATH).read_bytes())

    def tearDown(self):
        self.bundle_patch.stop()
        patch.stopall()
        self.temp.cleanup()

    def fake_bundle(self, code_root, destination):
        metadata = publish.copy_stable(code_root / self.source_path, destination / "source" / self.source_path)
        files = {self.source_path: metadata}
        files[publish.DECISION_PATH] = publish.copy_stable(code_root / publish.DECISION_PATH,
                                                          destination / "source" / publish.DECISION_PATH)
        bundle = {"source_files": files, "source_sha256": digest(files)}
        atomic_json(destination / "bundle.json", bundle)
        return bundle

    def make_freeze(self):
        """Full matrix and bound metadata, without loading a numerical model."""
        cache = self.root / "cache_metadata"
        cache.mkdir()
        for name in ("protocol.json", "cohort.json", "manifest.json", "validation.json", "rewire/manifest.json"):
            atomic_json(cache / name, {"name": name, "complete": True, "passed": True})
        self.rewiring_status = {"mixing_quality_passed": False, "decision": publish.approved_decision(),
                               "development_diagnostics": {"mean_changed_fraction": .54}}
        atomic_json(cache / "rewire/admission.json", self.rewiring_status)
        # Numerical construction validation has dedicated Slurm tests; this mock
        # isolates publisher inventory, hash binding and qualification retention.
        patch.object(publish, "read_admission", return_value=self.rewiring_status).start()
        self.args.cache_metadata_root = cache
        for arm in publish.ARMS:
            for seed in publish.SEEDS:
                target = self.args.run_root / arm / f"seed{seed}"
                if target != self.run:
                    shutil.copytree(self.run, target)
                    config = json.loads((target / "config.json").read_text())
                    config.update(arm=arm, seed=seed)
                    if arm == "full_rewired":
                        config.update(rewire_admission_sha256=file_sha256(cache / "rewire/admission.json"),
                                      rewire_decision_sha256=publish.approved_decision()["sha256"])
                    atomic_json(target / "config.json", config)
                    validation = json.loads((target / "validation.json").read_text())
                    validation["config_sha256"] = file_sha256(target / "config.json")
                    atomic_json(target / "validation.json", validation)
                    complete = json.loads((target / "complete.json").read_text())
                    complete.update({key: file_sha256(target / name) for key, name in publish.BOUND_FILES.items()
                                     if key != "complete_sha256"})
                    atomic_json(target / "complete.json", complete)
        references = {}
        for extension in ("npz", "json"):
            name = "freeze.validation_references." + extension
            (self.root / name).write_bytes(b"reference calibration bytes")
            references[extension + "_path"] = name
            references[extension + "_sha256"] = file_sha256(self.root / name)
        value = {"schema_version": 1, "frozen": True, "matrix": "full", "references": references,
                 "runs": {f"{arm}/seed{seed}": publish.run_bindings(self.args.run_root / arm / f"seed{seed}")
                          for arm in publish.ARMS for seed in publish.SEEDS},
                 "implementation": self.implementation, "implementation_sha256": digest(self.implementation),
                 "protocol_sha256": digest(json.loads((cache / "protocol.json").read_text())),
                 "cohort_sha256": digest(json.loads((cache / "cohort.json").read_text())),
                 "cache_manifest_sha256": file_sha256(cache / "manifest.json"),
                 "cache_validation_sha256": file_sha256(cache / "validation.json"),
                 "rewire_manifest_sha256": file_sha256(cache / "rewire/manifest.json"),
                 "rewire_admission_sha256": file_sha256(cache / "rewire/admission.json"),
                 "rewire_decision_sha256": publish.approved_decision()["sha256"],
                 "rewiring_status": self.rewiring_status}
        self.args.freeze = self.root / "freeze.json"
        atomic_json(self.args.freeze, value)
        return value

    def make_results(self, freeze):
        results = self.root / "results"
        results.mkdir()
        self.args.results_root = results
        identity = file_sha256(self.args.freeze)
        for name in set(publish.FINAL_FILES) - {"evaluation_complete.json"}:
            if name.endswith(".json"):
                atomic_json(results / name, {"freeze_sha256": identity})
            else:
                (results / name).write_bytes(b"complete result fixture")
        for run in freeze["runs"]:
            stem = "predictions/" + run.replace("/", "__")
            for extension in (".npz", ".receipt.json"):
                path = results / (stem + extension)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"bound model prediction fixture")
        for name in ("msp", "entropy"):
            (results / "predictions" / (name + ".npz")).write_bytes(b"reference prediction fixture")
        artifacts = {path.relative_to(results).as_posix(): file_sha256(path) for path in results.rglob("*") if path.is_file()}
        atomic_json(results / "evaluation_complete.json", {"status": "complete", "freeze_sha256": identity, "artifacts": artifacts})
        return results

    def test_allowlist_and_all_seed_rows(self):
        (self.run / "private_chat.txt").write_text("must not upload")
        (self.run / "old_checkpoint.pt").write_bytes(b"must not upload")
        before = {p.name: p.read_bytes() for p in self.run.iterdir()}
        stage, manifest = publish.stage_snapshot(self.args)
        self.assertEqual(len(manifest["runs"]), 35)
        self.assertEqual(manifest["runs"]["raw_set/seed27"]["status"], "not_started")
        self.assertFalse(any("private_chat" in name or "old_checkpoint" in name for name in manifest["files"]))
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.run.iterdir()})
        self.assertEqual(publish.verify_local(stage)["snapshot_id"], manifest["snapshot_id"])

    def test_publish_retry_is_idempotent_and_pins_model_cards(self):
        hub = MockHub()
        first = publish.upload_once(self.args, api=hub, downloader=hub.download)
        second = publish.upload_once(self.args, api=hub, downloader=hub.download)
        self.assertTrue(first["backup_complete"])
        self.assertEqual(first["revision"], second["revision"])
        self.assertEqual(hub.uploads, 2)
        readme = hub.revisions[first["revision"]]["topology_20260910/runs/full_graph/seed1/README.md"].decode()
        self.assertIn(first["artifact_revision"], readme)

    def test_lost_document_commit_response_resumes(self):
        hub = MockHub()
        hub.fail_after_document_commit = True
        with self.assertRaises(ConnectionError):
            publish.upload_once(self.args, api=hub, downloader=hub.download)
        receipt = publish.upload_once(self.args, api=hub, downloader=hub.download)
        self.assertTrue(receipt["backup_complete"])
        self.assertEqual(hub.uploads, 2)

    def test_public_repository_rejected_without_mutation(self):
        hub = MockHub()
        hub.private = False
        with self.assertRaisesRegex(RuntimeError, "private model"):
            publish.upload_once(self.args, api=hub, downloader=hub.download)
        self.assertEqual(hub.uploads, 0)

    def test_tampered_selected_weights_rejected(self):
        (self.run / "best.safetensors").write_bytes(b"different")
        with self.assertRaisesRegex(RuntimeError, "Completion hash mismatch"):
            publish.stage_snapshot(self.args)

    def test_source_and_freeze_bindings_enforced(self):
        freeze = self.make_freeze()
        publish.stage_snapshot(self.args)
        freeze["runs"]["full_graph/seed1"]["config_sha256"] = "f" * 64
        atomic_json(self.args.freeze, freeze)
        with self.assertRaisesRegex(RuntimeError, "Frozen run mismatch"):
            publish.stage_snapshot(self.args)

    def test_changed_implementation_and_missing_completed_run_rejected(self):
        previous = {"completed_runs": {"full_graph/seed1": publish.run_bindings(self.run)}}
        (self.args.code_root / self.source_path).write_text("# changed implementation\n")
        with self.assertRaisesRegex(RuntimeError, "Bundled code does not match"):
            publish.stage_snapshot(self.args)
        (self.run / "complete.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "disappeared or changed"):
            publish.stage_snapshot(self.args, previous)

    def test_final_metrics_require_freeze(self):
        atomic_json(self.run / "final_metrics.json", {"status": "done"})
        with self.assertRaisesRegex(RuntimeError, "requires --freeze"):
            publish.stage_snapshot(self.args)

    def test_operator_storage_ceiling_keeps_backup_incomplete(self):
        hub = MockHub()
        self.args.quota_available_bytes = 0
        with self.assertRaisesRegex(RuntimeError, "storage ceiling"):
            publish.upload_once(self.args, api=hub, downloader=hub.download)
        self.assertEqual(hub.uploads, 0)
        receipts = [path for path in self.args.receipt_dir.glob("*.json") if path.name != "latest.json"]
        self.assertEqual(len(receipts), 1)
        self.assertIs(json.loads(receipts[0].read_text())["backup_complete"], False)

    def test_download_checksum_and_revision_enforced(self):
        hub = MockHub()
        receipt = publish.upload_once(self.args, api=hub, downloader=hub.download)
        with self.assertRaisesRegex(ValueError, "immutable"):
            publish.download_verified(self.args.repo_id, "main", self.root / "download", api=hub, downloader=hub.download)
        name = "topology_20260910/runs/full_graph/seed1/latest.pt"
        hub.revisions[receipt["revision"]][name] = b"tampered"
        with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            publish.download_verified(self.args.repo_id, receipt["revision"], self.root / "download",
                                      api=hub, downloader=hub.download)

    def test_symlink_and_path_escape_rejected(self):
        (self.run / "best.safetensors").unlink()
        (self.run / "best.safetensors").symlink_to(self.root / "external")
        (self.root / "external").write_bytes(b"outside allowlist")
        with self.assertRaisesRegex(RuntimeError, "Symlinks"):
            publish.stage_snapshot(self.args)
        with self.assertRaises(ValueError):
            publish.safe_relative("../Context/private.txt")
        self.assertFalse(publish.source_allowed("runs/old/model.pt"))
        self.assertFalse(publish.source_allowed("Context/chat.txt"))

    def test_single_writer_lock(self):
        with publish.publisher_lock(self.args.receipt_dir):
            with self.assertRaisesRegex(RuntimeError, "Another centralized"):
                with publish.publisher_lock(self.args.receipt_dir):
                    self.fail("Second publisher acquired the lock")

    def test_gitless_source_release_is_verified_and_has_no_false_patch_claim(self):
        private = self.args.code_root / "Context/private.md"
        private.parent.mkdir()
        private.write_text("private context must not be copied")
        files = {self.source_path: file_sha256(self.args.code_root / self.source_path),
                 "Context/private.md": file_sha256(private)}
        atomic_json(self.args.code_root / "source_manifest.json", {"base_commit": publish.SOURCE_COMMIT, "files": files})
        destination = self.root / "real_bundle"
        with patch.object(publish, "git", side_effect=AssertionError("Git must not be required by immutable release")):
            bundle = REAL_BUNDLE_SOURCE(self.args.code_root, destination)
        self.assertIsNone(bundle["patch_sha256"])
        self.assertEqual(bundle["provenance"]["kind"], "immutable_slurm_release")
        self.assertFalse((destination / "working-tree.patch").exists())
        self.assertFalse((destination / "source/Context/private.md").exists())
        self.assertTrue((destination / "requirements.lock.txt").is_file())
        (self.args.code_root / self.source_path).write_text("changed after immutable manifest")
        with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            REAL_BUNDLE_SOURCE(self.args.code_root, self.root / "changed_bundle")

    def test_final_artifact_contract_and_calibration_are_complete(self):
        freeze = self.make_freeze()
        results = self.make_results(freeze)
        stage, payload = publish.stage_snapshot(self.args)
        self.assertEqual(payload["test_results_status"], "complete_evaluation_available")
        self.assertTrue(payload["runs"]["full_graph/seed1"]["final_metrics_available"])
        for name in ("results.csv", "bootstrap_draws.npz", "report.he.md", "evaluation_complete.json",
                     "predictions/raw_set__seed27.npz", "predictions/msp.npz"):
            self.assertIn(publish.PREFIX + "/results/" + name, payload["files"])
        self.assertIn(publish.PREFIX + "/freeze.validation_references.json", payload["files"])
        self.assertIn(publish.PREFIX + "/cache_metadata/rewire/manifest.json", payload["files"])
        self.assertIn(publish.PREFIX + "/cache_metadata/rewire/admission.json", payload["files"])
        self.assertFalse(payload["rewiring_status"]["mixing_quality_passed"])
        publish.verify_local(stage)
        (results / "results.csv").unlink()
        with self.assertRaisesRegex(RuntimeError, "Missing or unsafe"):
            publish.stage_snapshot(self.args)

    def test_final_freeze_rejects_missing_models_and_calibration(self):
        freeze = self.make_freeze()
        reference = self.root / freeze["references"]["npz_path"]
        reference.unlink()
        with self.assertRaisesRegex(RuntimeError, "Missing or unsafe"):
            publish.stage_snapshot(self.args)
        del freeze["runs"]["raw_set/seed27"]
        atomic_json(self.args.freeze, freeze)
        with self.assertRaisesRegex(RuntimeError, "complete approved"):
            publish.stage_snapshot(self.args)

    def test_terminal_wait_preserves_interval_without_real_sleep(self):
        clock = {"now": 0.0, "sleeps": []}

        def sleep(seconds):
            clock["sleeps"].append(seconds)
            clock["now"] += seconds

        with patch.object(publish.time, "monotonic", side_effect=lambda: clock["now"]), \
                patch.object(publish.time, "sleep", side_effect=sleep):
            self.assertEqual(publish.wait_for_terminal_or_interval(65, None), "interval")
        self.assertEqual(clock["sleeps"], [30.0, 30.0, 5.0])
        self.assertEqual(clock["now"], 65.0)

    def test_terminal_wakes_final_upload_and_retains_single_publisher_lock(self):
        terminal = self.root / "terminal.json"
        clock = {"now": 0.0, "sleeps": [], "uploads": []}

        def assert_locked():
            with self.assertRaisesRegex(RuntimeError, "Another centralized"):
                with publish.publisher_lock(self.args.receipt_dir):
                    self.fail("Central publisher lock was released before final upload")

        def sleep(seconds):
            assert_locked()
            clock["sleeps"].append(seconds)
            clock["now"] += seconds
            if clock["now"] >= 45.0:
                terminal.write_text('{"status":"terminal"}')

        def upload(args):
            assert_locked()
            clock["uploads"].append(clock["now"])
            return {"revision": "a" * 40, "snapshot_id": "mock_snapshot"}

        argv = ["publish", "--run-root", str(self.args.run_root), "--receipt-dir", str(self.args.receipt_dir),
                "--code-root", str(self.args.code_root), "--interval", "3600", "--max-cycles", "3",
                "--terminal-marker", str(terminal)]
        with patch.object(publish.sys, "argv", argv), \
                patch.object(publish.time, "monotonic", side_effect=lambda: clock["now"]), \
                patch.object(publish.time, "sleep", side_effect=sleep), \
                patch.object(publish, "upload_once", side_effect=upload):
            publish.main()
        self.assertEqual(clock["uploads"], [0.0, 60.0])
        self.assertLessEqual(max(clock["sleeps"]), 30.0)
        self.assertLessEqual(clock["uploads"][-1] - 45.0, 30.0)
        with publish.publisher_lock(self.args.receipt_dir):
            pass  # Final upload completed and the central lock is now released.

    def test_completion_wakes_once_and_does_not_consume_hourly_cycle_budget(self):
        clock = {"now": 0.0, "sleeps": [], "uploads": []}
        marker = self.args.run_root / "raw_set/seed27/complete.json"

        def assert_locked():
            with self.assertRaisesRegex(RuntimeError, "Another centralized"):
                with publish.publisher_lock(self.args.receipt_dir):
                    self.fail("Fit completion started an independent uploader")

        def sleep(seconds):
            assert_locked()
            clock["sleeps"].append(seconds)
            clock["now"] += seconds
            if clock["now"] >= 45.0 and not marker.exists():
                marker.parent.mkdir(parents=True)
                marker.write_text('{"complete":true}')

        def upload(args):
            assert_locked()
            clock["uploads"].append(clock["now"])
            return {"revision": "a" * 40, "snapshot_id": "same_verified_snapshot"}

        argv = ["publish", "--run-root", str(self.args.run_root), "--receipt-dir", str(self.args.receipt_dir),
                "--code-root", str(self.args.code_root), "--interval", "3600", "--max-cycles", "2"]
        with patch.object(publish.sys, "argv", argv), \
                patch.object(publish.time, "monotonic", side_effect=lambda: clock["now"]), \
                patch.object(publish.time, "sleep", side_effect=sleep), \
                patch.object(publish, "upload_once", side_effect=upload):
            publish.main()
        self.assertEqual(clock["uploads"], [0.0, 60.0, 3660.0])
        self.assertLessEqual(max(clock["sleeps"]), 30.0)
        receipt = json.loads((self.args.receipt_dir / "last_cycle.json").read_text())
        self.assertEqual(receipt["cycle_counts"], {"progress": 2, "completion": 1, "terminal": 0})
        self.assertIn("raw_set/seed27", receipt["completion_baseline"])
        self.assertEqual(receipt["completion_baseline"], receipt["completion_after_upload"])
        self.assertEqual(receipt["snapshot_id"], "same_verified_snapshot")

    def test_completion_during_upload_and_terminal_requires_catch_up(self):
        terminal = self.root / "terminal.json"
        marker = self.args.run_root / "logit/seed27/complete.json"
        observed = []

        def upload(args):
            observed.append(publish.completed_run_markers(args.run_root))
            with self.assertRaisesRegex(RuntimeError, "Another centralized"):
                with publish.publisher_lock(args.receipt_dir):
                    self.fail("The central uploader lock was not retained")
            if len(observed) == 1:
                marker.parent.mkdir(parents=True)
                marker.write_text('{"complete":true}')
                terminal.write_text('{"status":"terminal"}')
            return {"revision": "a" * 40, "snapshot_id": "verified_snapshot"}

        argv = ["publish", "--run-root", str(self.args.run_root), "--receipt-dir", str(self.args.receipt_dir),
                "--code-root", str(self.args.code_root), "--interval", "3600", "--max-cycles", "1",
                "--terminal-marker", str(terminal)]
        with patch.object(publish.sys, "argv", argv), \
                patch.object(publish.time, "monotonic", return_value=0.0), \
                patch.object(publish.time, "sleep", side_effect=AssertionError("Catch-up must be immediate")), \
                patch.object(publish, "upload_once", side_effect=upload):
            publish.main()
        self.assertEqual(len(observed), 2)
        self.assertNotIn("logit/seed27", observed[0])
        self.assertIn("logit/seed27", observed[1])
        receipt = json.loads((self.args.receipt_dir / "last_cycle.json").read_text())
        self.assertEqual(receipt["wake_reason"], "completion")
        self.assertEqual(receipt["cycle_counts"], {"progress": 1, "completion": 1, "terminal": 0})
        self.assertEqual(receipt["completion_baseline"], receipt["completion_after_upload"])

    def test_only_known_completion_paths_wake_the_publisher(self):
        baseline = publish.completed_run_markers(self.args.run_root)
        clock = {"now": 0.0}

        def sleep(seconds):
            clock["now"] += seconds
            unrelated = self.args.run_root / "unregistered_arm/seed99/complete.json"
            unrelated.parent.mkdir(parents=True, exist_ok=True)
            unrelated.write_text('{"complete":true}')

        with patch.object(publish.time, "monotonic", side_effect=lambda: clock["now"]), \
                patch.object(publish.time, "sleep", side_effect=sleep):
            reason = publish.wait_for_terminal_or_interval(60, None, self.args.run_root, baseline)
        self.assertEqual(reason, "interval")
        self.assertEqual(publish.completed_run_markers(self.args.run_root), baseline)

    def test_terminal_observation_rechecks_completion_after_cycle_receipt(self):
        terminal = self.root / "terminal.json"
        marker = self.args.run_root / "raw_graph/seed17/complete.json"
        write_json = publish.atomic_json
        uploads = []

        def write_receipt(path, value):
            write_json(path, value)
            if path.name == "last_cycle.json" and value["cycle"] == 1:
                # Arrive after the first post-upload set read, immediately
                # before the publisher observes the terminal marker.
                marker.parent.mkdir(parents=True)
                marker.write_text('{"complete":true}')
                terminal.write_text('{"status":"terminal"}')

        def upload(args):
            uploads.append(publish.completed_run_markers(args.run_root))
            return {"revision": "a" * 40, "snapshot_id": "verified_snapshot"}

        argv = ["publish", "--run-root", str(self.args.run_root), "--receipt-dir", str(self.args.receipt_dir),
                "--code-root", str(self.args.code_root), "--interval", "3600", "--max-cycles", "1",
                "--terminal-marker", str(terminal)]
        with patch.object(publish.sys, "argv", argv), \
                patch.object(publish.time, "monotonic", return_value=0.0), \
                patch.object(publish.time, "sleep", side_effect=AssertionError("Catch-up must be immediate")), \
                patch.object(publish, "atomic_json", side_effect=write_receipt), \
                patch.object(publish, "upload_once", side_effect=upload):
            publish.main()
        self.assertEqual(len(uploads), 2)
        self.assertNotIn("raw_graph/seed17", uploads[0])
        self.assertIn("raw_graph/seed17", uploads[1])


if __name__ == "__main__":
    require_slurm()
    unittest.main()
