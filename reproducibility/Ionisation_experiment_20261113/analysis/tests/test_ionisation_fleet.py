import hashlib
import importlib
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

ClientError = pytest.importorskip("botocore.exceptions").ClientError


@pytest.fixture
def fleet(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "aws"))
    return importlib.import_module("ionisation_fleet")


def client_error(code):
    return ClientError({"Error": {"Code": code, "Message": code}}, "mock-s3")


class ConditionalS3:
    def __init__(self):
        self.objects = {}
        self.rejections = []

    def get_object(self, *, Bucket, Key):
        try:
            body, etag = self.objects[Bucket, Key]
        except KeyError:
            raise client_error("NoSuchKey") from None
        return {"Body": io.BytesIO(body), "ETag": etag}

    def put_object(self, *, Bucket, Key, Body, IfMatch=None, IfNoneMatch=None, **kwargs):
        assert kwargs["ServerSideEncryption"] == "AES256"
        assert (IfMatch is not None) != (IfNoneMatch is not None)
        current = self.objects.get((Bucket, Key))
        if (IfNoneMatch == "*" and current) or (IfMatch is not None and (not current or current[1] != IfMatch)):
            self.rejections.append((Key, IfMatch, IfNoneMatch))
            raise client_error("PreconditionFailed")
        etag = f'"{hashlib.sha256(Body).hexdigest()}"'
        self.objects[Bucket, Key] = Body, etag
        return {"ETag": etag}

    def download_file(self, bucket, key, destination):
        Path(destination).write_bytes(self.objects[bucket, key][0])


@pytest.fixture
def queue(fleet, monkeypatch):
    clock = SimpleNamespace(now=20_000.)
    monkeypatch.setattr(fleet, "time", SimpleNamespace(time=lambda: clock.now))
    s3 = ConditionalS3()
    queue = fleet.LeaseQueue(s3, "test", "campaign-prefix", "campaign", ttl=20, max_attempts=2)
    return queue, s3, clock


def test_concurrent_first_claim_only_one_wins(fleet, queue, monkeypatch):
    first, s3, _ = queue
    second = fleet.LeaseQueue(s3, "test", "campaign-prefix", "campaign", ttl=20, max_attempts=2)
    observed_missing = second.get("u1")
    assert first.claim("u1", "one") is not None
    monkeypatch.setattr(second, "get", lambda unit: observed_missing)
    assert second.claim("u1", "two") is None
    assert first.get("u1")[0]["owner"] == "one"
    assert s3.rejections == [(first.key("u1"), None, "*")]


def test_stale_etag_completion_is_rejected(fleet, queue):
    queue, s3, clock = queue
    lease = queue.claim("u1", "same-worker")
    # An in-flight commit retains the prior state even if a heartbeat wins CAS.
    stale = fleet.Lease(queue, "u1", lease.state.copy(), lease.etag)
    clock.now += 5
    lease.heartbeat()
    assert lease.etag != stale.etag
    with pytest.raises(RuntimeError, match="replaced"):
        stale.update(status="complete", expires=0)
    assert stale.lost
    assert queue.get("u1") == (lease.state, lease.etag)
    assert s3.rejections == [(queue.key("u1"), stale.etag, None)]


def test_expiry_preserves_checkpoint_and_fences_prior_attempt(queue):
    queue, _, clock = queue
    first = queue.claim("u1", "first")
    checkpoint = {"key": "campaign-prefix/attempts/u1/previous.tar.gz", "sha256": "a" * 64}
    first.update(checkpoint=checkpoint)
    assert queue.claim("u1", "second") is None
    clock.now = first.state["expires"]
    second = queue.claim("u1", "second")
    assert second.state["checkpoint"] == checkpoint
    assert second.state["attempt"] == 2
    assert second.state["token"] != first.state["token"]
    with pytest.raises(RuntimeError, match="expired"):
        first.update(status="complete", expires=0)
    assert queue.get("u1")[0]["owner"] == "second"
    second.update(status="complete", expires=0)
    assert queue.claim("u1", "third") is None


def test_max_attempts_applies_to_failed_and_expired_leases(queue):
    queue, _, clock = queue
    first = queue.claim("u1", "first")
    first.update(status="failed", expires=0)
    retry = queue.claim("u1", "second")
    clock.now = retry.state["expires"] + 1
    assert queue.claim("u1", "third") is None
    assert queue.get("u1")[0]["attempt"] == 2


def test_claim_refuses_reused_prefix_from_another_campaign(fleet, queue):
    current, s3, _ = queue
    lease = current.claim("u1", "original")
    other = fleet.LeaseQueue(s3, "test", "campaign-prefix", "wrong-campaign")
    with pytest.raises(ValueError, match="another campaign"):
        other.claim("u1", "different")
    assert current.get("u1") == (lease.state, lease.etag)


def add_tar_file(archive, name, data=b"ok"):
    member = tarfile.TarInfo(name)
    member.size = len(data)
    archive.addfile(member, io.BytesIO(data))


@pytest.mark.parametrize("kind", ["traversal", "absolute", "symlink", "hardlink", "fifo"])
def test_archive_rejects_escape_and_unsupported_members(fleet, tmp_path, kind):
    archive = tmp_path / "attempt.tar.gz"
    destination = tmp_path / "restore"
    outside = tmp_path / "outside.txt"
    with tarfile.open(archive, "w:gz") as stream:
        add_tar_file(stream, "good/data.txt")
        if kind == "traversal":
            add_tar_file(stream, "../outside.txt")
        elif kind == "absolute":
            add_tar_file(stream, outside.as_posix())
        else:
            member = tarfile.TarInfo("bad")
            member.type = {"symlink": tarfile.SYMTYPE, "hardlink": tarfile.LNKTYPE, "fifo": tarfile.FIFOTYPE}[kind]
            member.linkname = "../outside.txt"
            stream.addfile(member)
    with pytest.raises(ValueError, match="Unsafe archive path|Unsupported archive member"):
        fleet.safe_extract(archive, destination)
    assert not outside.exists()
    assert not (destination / "good/data.txt").exists()


def test_archive_roundtrip(fleet, tmp_path):
    source = tmp_path / "unit"
    source.mkdir()
    (source / "complete_unit.json").write_text('{"unit": "u1"}', encoding="utf-8")
    (source / ".fit.lock").touch()
    archive = tmp_path / "attempt.tar.gz"
    digest = fleet.pack_tree(source, archive)
    assert digest == hashlib.sha256(archive.read_bytes()).hexdigest()
    restored = tmp_path / "restored"
    fleet.safe_extract(archive, restored)
    assert json.loads((restored / "complete_unit.json").read_text(encoding="utf-8")) == {"unit": "u1"}
    assert not (restored / ".fit.lock").exists()


def save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_worker_keeps_checkpoint_when_previous_archive_is_corrupt(fleet, tmp_path, monkeypatch):
    import boto3

    unit = "class-GraphConv-outer1"
    bucket = "test"
    prefix = "campaign-prefix"
    s3 = ConditionalS3()
    queue = fleet.LeaseQueue(s3, bucket, prefix, "campaign")
    first = queue.claim(unit, "first")
    checkpoint = {"key": f"{prefix}/attempts/{unit}/previous.tar.gz", "sha256": "a" * 64}
    s3.objects[bucket, checkpoint["key"]] = (b"corrupt", "mock-checkpoint-etag")
    first.update(status="failed", expires=0, checkpoint=checkpoint)
    root = tmp_path / "bundle"
    (root / "runtime").mkdir(parents=True)
    save_json(root / "runtime/plan.json", {"campaign_id": "campaign", "units": [{"unit_id": unit}]})
    monkeypatch.setattr(fleet, "verify_plan", lambda root, plan: None)
    monkeypatch.setattr(fleet, "run_unit", lambda *args: pytest.fail("Corrupt resume ran a fit"))
    monkeypatch.setattr(boto3, "client", lambda service, region_name: s3)
    args = SimpleNamespace(root=root, work=tmp_path / "work", bucket=bucket, prefix=prefix,
                           region="us-east-1", max_hours=1., workers=1)
    fleet.worker(args)
    state, _ = queue.get(unit)
    assert state["attempt"] == 2 and state["status"] == "failed"
    assert state["checkpoint"] == checkpoint
    assert state["error"] == "ValueError: Resume checksum mismatch"
    assert (args.work / "attempts" / state["token"] / f"{unit}-resume.tar.gz").is_file()


@pytest.fixture
def launch_setup(fleet, tmp_path, monkeypatch):
    import boto3

    request_path, approval_path, bundle_path = [tmp_path / name for name in ("request.json", "approval.json", "bundle.json")]
    args = SimpleNamespace(request=request_path, approval=approval_path, bundle_receipt=bundle_path,
                           out=tmp_path / "receipt.json", region="us-east-1", execute=True)
    bundle = {"training_ready": True, "sha256": "b" * 64, "campaign_id": "campaign"}
    save_json(bundle_path, bundle)
    settings = {
        "bucket": "mock-bucket", "bundle_key": f"bundles/{bundle['sha256']}.tar.gz",
        "bundle_sha256": bundle["sha256"], "prefix": "campaign-prefix", "region": args.region,
        "container_image": f"123456789012.dkr.ecr.us-east-1.amazonaws.com/ionisation@sha256:{'c' * 64}",
        "workers": 2,
    }
    request = {
        "ImageId": "ami-fake", "InstanceType": "g4dn.xlarge", "MinCount": 2, "MaxCount": 2,
        "ClientToken": "approved-test-campaign", "InstanceInitiatedShutdownBehavior": "terminate",
        "MetadataOptions": {"HttpTokens": "required"},
        "BlockDeviceMappings": [{"DeviceName": "/dev/xvda", "Ebs": {"DeleteOnTermination": True}}],
        "UserData": fleet.render_user_data(settings, 2.),
    }
    save_json(request_path, request)
    approval = {
        "approved": True, "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
        "expires_utc_epoch": 30_000., "all_in_usd_per_instance_hour": .5,
        "hard_stop_hours": 2., "approved_cost_ceiling_usd": 2.25, "bundle_sha256": bundle["sha256"],
        "region": args.region, "campaign_id": bundle["campaign_id"],
        "client_token": request["ClientToken"], "worker_settings": settings,
    }
    save_json(approval_path, approval)
    monkeypatch.setattr(fleet, "time", SimpleNamespace(time=lambda: 20_000.))
    calls = []

    def run_instances(**kwargs):
        calls.append(kwargs)
        return {"Instances": [{"InstanceId": f"i-mock-{i}"} for i in range(kwargs["MaxCount"])]}

    def client(service, *, region_name):
        assert (service, region_name) == ("ec2", "us-east-1")
        return SimpleNamespace(run_instances=run_instances)

    monkeypatch.setattr(boto3, "client", client)
    return args, request, approval, calls


def test_launch_rejects_request_byte_change_and_over_budget(fleet, launch_setup):
    args, request, approval, calls = launch_setup
    original = args.request.read_bytes()
    args.request.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="exact request"):
        fleet.launch(args)
    assert calls == []
    args.request.write_bytes(original)
    approval["approved_cost_ceiling_usd"] = 2.24  # Below the boot/termination-inclusive envelope.
    save_json(args.approval, approval)
    with pytest.raises(ValueError, match="cost ceiling"):
        fleet.launch(args)
    assert calls == []
    assert not args.out.exists()


def test_launch_dry_run_and_exact_approved_request(fleet, launch_setup):
    args, request, _, calls = launch_setup
    args.execute = False
    fleet.launch(args)
    assert calls == []
    assert not args.out.exists()
    args.execute = True
    fleet.launch(args)
    assert calls == [request]
    receipt = json.loads(args.out.read_text(encoding="utf-8"))
    assert receipt["region"] == "us-east-1"
    assert receipt["request_sha256"] == hashlib.sha256(args.request.read_bytes()).hexdigest()
    assert receipt["instance_ids"] == ["i-mock-0", "i-mock-1"]


@pytest.mark.parametrize("field, value", [
    ("all_in_usd_per_instance_hour", -1),
    ("all_in_usd_per_instance_hour", float("nan")),
    ("all_in_usd_per_instance_hour", True),
    ("hard_stop_hours", 0),
    ("hard_stop_hours", float("inf")),
    ("hard_stop_hours", 25),
    ("approved_cost_ceiling_usd", float("nan")),
    ("approved_cost_ceiling_usd", float("inf")),
    ("expires_utc_epoch", float("nan")),
    ("expires_utc_epoch", 20_000.),
    ("region", "us-west-2"),
])
def test_launch_rejects_invalid_budget_terms_and_region(fleet, launch_setup, field, value):
    args, _, approval, calls = launch_setup
    approval[field] = value
    save_json(args.approval, approval)
    with pytest.raises(ValueError):
        fleet.launch(args)
    assert calls == []
    assert not args.out.exists()


@pytest.mark.parametrize("field, value", [
    ("ClientToken", "unapproved-token"),
    ("InstanceInitiatedShutdownBehavior", "stop"),
    ("DisableApiTermination", True),
    ("DisableApiStop", True),
    ("LaunchTemplate", {"LaunchTemplateId": "lt-mock"}),
    ("InstanceMarketOptions", {"MarketType": "spot"}),
    ("MetadataOptions", {"HttpTokens": "optional"}),
    ("BlockDeviceMappings", [{"Ebs": {"DeleteOnTermination": False}}]),
    ("UserData", "#!/bin/bash\nsleep infinity\n"),
])
def test_launch_rejects_rehashed_request_without_operational_guards(fleet, launch_setup, field, value):
    args, request, approval, calls = launch_setup
    request[field] = value
    save_json(args.request, request)
    approval["request_sha256"] = hashlib.sha256(args.request.read_bytes()).hexdigest()
    save_json(args.approval, approval)
    with pytest.raises(ValueError):
        fleet.launch(args)
    assert calls == []
    assert not args.out.exists()
