"""Local tests for categorical pH worker lifecycle and checkpoint fencing."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import threading
import time
from types import SimpleNamespace

import pytest


@pytest.fixture
def fleet(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'aws'))
    return importlib.import_module('ph_ionisation_fleet')


def test_dag_and_retry_readiness(fleet):
    core = {'unit_id':'core','depends_on':None}
    tune = {'unit_id':'tune','depends_on':'core'}
    states = {'core':{'status':'pending','attempt':0,'expires':0},
              'tune':{'status':'pending','attempt':0,'expires':0}}
    assert fleet.eligible(core,states,50)
    assert not fleet.eligible(tune,states,50)
    states['core'].update(status='running',attempt=1,expires=51)
    assert not fleet.eligible(core,states,50)
    assert fleet.eligible(core,states,51)
    assert not fleet.eligible(tune,states,51)
    states['core'].update(status='complete',expires=0)
    assert fleet.eligible(tune,states,51)
    states['tune'].update(status='failed',attempt=2)
    assert not fleet.eligible(tune,states,60)


@pytest.fixture
def snapshot_fixture(fleet,tmp_path):
    unit = tmp_path/'core-u'
    fits = unit/'fits'
    complete, incomplete = (fits/name for name in ('fit-good','fit-in-progress'))
    complete.mkdir(parents=True)
    incomplete.mkdir()
    (incomplete/'checkpoint.pt').write_bytes(b'in-progress bytes')
    (complete/'.checkpoint.pt.tmp').write_bytes(b'not a committed artifact')
    for name in fleet.FIT_FILES:
        (complete/name).write_text(name,encoding='utf-8')
    fleet.write(complete/'request.json',{'request':{'campaign_id':'campaign','fit_id':complete.name},'resolved_hash':'run'})
    fleet.write(complete/'complete.json',{'fit_id':complete.name,'resolved_hash':'run',
                                         'files':{name:fleet.sha(complete/name) for name in fleet.FIT_FILES}})
    (unit/'selection.json').write_text('uncommitted selection',encoding='utf-8')
    lease = SimpleNamespace(state={'token':'lease','checkpoint':None},unit='core-u',queue=SimpleNamespace(campaign='campaign'))
    uploads = {}

    def upload(path,bucket,key,ExtraArgs):
        assert ExtraArgs=={'ServerSideEncryption':'AES256'}
        assert key not in uploads
        uploads[key] = Path(path).read_bytes()

    s3 = SimpleNamespace(upload_file=upload)
    return unit, complete, lease, uploads, s3


def test_snapshot_is_checksum_verified_and_omits_in_progress_files(fleet,snapshot_fixture,tmp_path):
    unit, complete, lease, uploads, s3 = snapshot_fixture
    stop = threading.Event()
    result = fleet._publish_checkpoint(s3,'bucket','prefix',lease,unit,stop)
    assert result['fit_markers'] == {complete.name:fleet.sha(complete/'complete.json')}
    assert result['key'].startswith('prefix/attempts/core-u/snapshot-lease-')
    assert lease.state['checkpoint'] is None  # Publishing alone does not update the lease.
    archived = tmp_path/'result.tar.gz'
    archived.write_bytes(uploads[result['key']])
    assert fleet.sha(archived) == result['sha256'] and archived.stat().st_size == result['bytes']
    with tarfile.open(archived) as archive:
        assert set(archive.getnames()) == {f'fits/{complete.name}/{name}' for name in (*fleet.FIT_FILES,'complete.json')}
    fleet.safe_extract(archived,tmp_path/'restored')
    assert fleet.sha(tmp_path/'restored/fits/fit-good/checkpoint.pt') == fleet.sha(complete/'checkpoint.pt')
    lease.state['checkpoint'] = result
    assert fleet._publish_checkpoint(s3,'bucket','prefix',lease,unit,stop) is None
    assert len(uploads) == 1
    (complete/'checkpoint.pt').write_bytes(b'tampered checkpoint')
    with pytest.raises(ValueError,match='differs from completion marker'):
        fleet._publish_checkpoint(s3,'bucket','prefix',lease,unit,stop)


def test_final_archive_binds_selection_and_all_fit_markers(fleet,snapshot_fixture,tmp_path):
    unit, complete, lease, uploads, s3 = snapshot_fixture
    marker = {'campaign_id':'campaign','unit':{'unit_id':'core-u'},'fit_markers':{complete.name:fleet.sha(complete/'complete.json')},
              'selection_sha256':'incorrect'}
    fleet.write(unit/'complete_unit.json',marker)
    with pytest.raises(ValueError,match='does not bind'):
        fleet._publish_checkpoint(s3,'bucket','prefix',lease,unit,threading.Event(),complete=True)
    marker['selection_sha256'] = fleet.sha(unit/'selection.json')
    fleet.write(unit/'complete_unit.json',marker)
    result = fleet._publish_checkpoint(s3,'bucket','prefix',lease,unit,threading.Event(),complete=True)
    assert result['key'].startswith('prefix/attempts/core-u/complete-lease-')
    archived = tmp_path/'final.tar.gz'
    archived.write_bytes(uploads[result['key']])
    with tarfile.open(archived) as archive:
        assert 'complete_unit.json' in archive.getnames() and 'selection.json' in archive.getnames()
        assert not any('fit-in-progress' in name or name.endswith('.tmp') for name in archive.getnames())


def test_upload_racing_lease_loss_cannot_return_commit(fleet,snapshot_fixture):
    unit, _, lease, _, s3 = snapshot_fixture
    stop = threading.Event()
    s3.upload_file = lambda *args,**kwargs:stop.set()
    assert fleet._publish_checkpoint(s3,'bucket','prefix',lease,unit,stop) is None
    assert lease.state['checkpoint'] is None


@pytest.mark.parametrize('mode', ['heartbeat','deadline'])
def test_cancel_managed_process(fleet,monkeypatch,mode):
    stop = threading.Event()
    killed = []
    started = []
    fake = SimpleNamespace(poll=lambda:None)
    monotonic = iter((0,0,0,0,0))
    deadline = iter((1,2 if mode=='heartbeat' else 100))
    monkeypatch.setattr(fleet,'boot_uptime',lambda:next(deadline))
    monkeypatch.setattr(fleet,'time',SimpleNamespace(monotonic=lambda:next(monotonic)))
    monkeypatch.setattr(fleet,'_terminate_process',killed.append)

    def popen(command,**kwargs):
        assert kwargs == {'start_new_session':True}
        started.append(command)
        if mode == 'heartbeat':
            stop.set()
        return fake

    monkeypatch.setattr(fleet,'subprocess',SimpleNamespace(Popen=popen))
    error = RuntimeError if mode=='heartbeat' else TimeoutError
    with pytest.raises(error):
        fleet._run_managed(['runner'],stop,100,lambda cancel:None)
    assert started == [['runner']] and killed == [fake]


@pytest.mark.skipif(os.name!='posix',reason='AWS worker uses a Linux process group')
def test_cancel_process_group_stops_grandchild(fleet,tmp_path,monkeypatch):
    stop = threading.Event()
    marker = tmp_path/'descendant-pulse'
    child = f"import time; from pathlib import Path; p=Path({str(marker)!r});\nwhile True: p.touch(); time.sleep(.05)"
    parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(60)"
    monkeypatch.setattr(fleet,'boot_uptime',lambda:0)

    def cancel():
        until = time.monotonic()+5
        while not marker.exists() and time.monotonic()<until:
            time.sleep(.02)
        stop.set()

    thread = threading.Thread(target=cancel,daemon=True)
    thread.start()
    with pytest.raises(RuntimeError,match='lease lost'):
        fleet._run_managed([sys.executable,'-c',parent],stop,60,lambda cancel:None)
    assert marker.exists()
    stamp = marker.stat().st_mtime_ns
    time.sleep(.2)
    assert marker.stat().st_mtime_ns==stamp


@pytest.mark.parametrize('overrides', [{'count':8},{'count':True},{'region':'us-west-2'},
                                      {'instance_type':'g6.xlarge'},{'hours':4},{'workers':1}])
def test_launch_scope_is_strict(fleet,overrides):
    arguments = dict(region='us-east-1',instance_type='g4dn.xlarge',count=1,hours=3,workers=2)
    fleet._approved_scope(**arguments)
    fleet._approved_scope(**(arguments|{'count':7}))
    with pytest.raises(ValueError,match='Reviewed launch scope'):
        fleet._approved_scope(**(arguments|overrides))
