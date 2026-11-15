"""Inventory retained installed-package notices; this is not a legal clearance."""
import hashlib
import importlib.metadata as md
import json
from pathlib import Path

rows=[]
for d in md.distributions():
    notices=[]
    for f in d.files or []:
        if any(s in Path(str(f)).name.lower() for s in ['license','copying','notice']):
            p=Path(d.locate_file(f))
            if p.is_file():notices.append({'installed_path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    rows.append({'name':d.metadata['Name'],'version':d.version,
                 'license_expression':d.metadata.get('License-Expression'),
                 'declared_license_files':d.metadata.get_all('License-File') or [],'retained_notice_files':notices})
out=Path('/checks/upstream_notice_inventory.json')
out.write_text(json.dumps({'scope':'Metadata/file inventory only. Upstream notices remain installed unchanged; no blanket relicensing or legal clearance is asserted.',
                           'distributions':sorted(rows,key=lambda x:x['name'].lower())},indent=2))
print(json.dumps({'distributions':len(rows),'distributions_with_notice_files':sum(bool(r['retained_notice_files']) for r in rows),
                  'notice_files':sum(len(r['retained_notice_files']) for r in rows)}))
