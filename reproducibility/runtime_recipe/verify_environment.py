"""Fail closed unless the image has the captured package inventory and conda records."""
import argparse
import importlib.metadata as md
import json
from pathlib import Path
import re
import subprocess
import sys

HERE=Path(__file__).resolve().parent
def key(s):return re.sub(r'[-_.]+','-',s).lower()
def urls(s):return {x.strip() for x in s.splitlines() if x.strip() and not x.startswith(('#','@'))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--base',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    expected=json.loads((HERE/('base_inventory.json' if a.base else 'target-pip-list.json')).read_text())
    expected=expected['packages'] if isinstance(expected,dict) else expected
    target={key(x['name']):x['version'] for x in expected}
    actual={key(d.metadata['Name']):d.version for d in md.distributions()}
    difference={n:{'expected':target.get(n),'actual':actual.get(n)} for n in sorted(set(target)|set(actual)) if target.get(n)!=actual.get(n)}
    if difference:raise ValueError(difference)
    conda=subprocess.check_output(['conda','list','--explicit'],text=True)
    if urls(conda)!=urls((HERE/'conda-explicit.txt').read_text()):raise ValueError('Conda artifact list changed')
    report={'status':'PASS','mode':'public_base' if a.base else 'rebuilt_target','python':sys.version,
            'distributions':len(actual),'exact_version_inventory_match':True,'conda_artifact_match':True,
            'conda_artifacts':len(urls(conda)),'binary_equivalence':'not asserted; installed names/versions and conda URLs checked'}
    if not a.base:
        apt={}
        for line in (HERE/'apt-runtime.lock').read_text().splitlines():
            name,version=line.split('=',1)
            actual_version=subprocess.check_output(['dpkg-query','-W','-f=${Version}',name],text=True).strip()
            if actual_version!=version:raise ValueError((name,actual_version,version))
            apt[name]=version
        report['apt_runtime_versions']=apt
        check=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
        baseline=json.loads((HERE/'base_inventory.json').read_text())['pip_check']
        # Preserve the exact observed historical environment; allow only the
        # independently reproduced public-base Ninja tag diagnostic, no new issue.
        allowed='ninja 1.11.1.1 is not supported on this platform\n'
        if check.returncode and not(check.stdout==baseline['stdout']==allowed and not check.stderr):
            raise ValueError({'pip_check':check.stdout,'stderr':check.stderr})
        ninja=subprocess.run(['ninja','--version'],capture_output=True,text=True,check=True)
        report['pip_check']={'returncode':check.returncode,'stdout':check.stdout,
                             'disposition':'inherited public-base wheel-tag diagnostic; actual ninja binary executes' if check.returncode else 'PASS'}
        report['ninja_version']=ninja.stdout.strip()
        import torch
        from rdkit import Chem
        import torch_geometric,chemprop,numpy,pandas,scipy
        assert Chem.MolFromSmiles('CCO').GetNumAtoms()==3
        x=torch.tensor([1.,2.,3.],requires_grad=True);(x.square().sum()).backward();assert torch.equal(x.grad,2*x)
        report['cpu_imports_and_autograd']='PASS'
        report['torch']=torch.__version__;report['cuda_build']=torch.version.cuda
    if a.output:a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
