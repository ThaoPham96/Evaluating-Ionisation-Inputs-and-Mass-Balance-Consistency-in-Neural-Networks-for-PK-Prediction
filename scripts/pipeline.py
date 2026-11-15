"""Reconstruct frozen inputs, rescore saved predictions, render, or train locally."""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'reproducibility'
IMAGE = 'pipknet-reproduction:20261114-v1'
UNIFIED = 'Unified_ionisation_combined_20261114'

def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def hydrate():
    """Materialize only verified tracked gzip artifacts, without replacing changed files."""
    index = json.loads((DATA / 'artifacts.json').read_text())
    for record in index['files']:
        target = DATA / record['path']
        packed = DATA / record['artifact']
        if not target.resolve().is_relative_to(DATA.resolve()):
            raise ValueError('Artifact path escapes the reproducibility directory')
        if digest(packed) != record['compressed_sha256']:
            raise ValueError(f'Compressed artifact checksum mismatch: {packed}')
        if target.exists():
            if digest(target) != record['sha256']:
                raise ValueError(f'Existing table differs from frozen data: {target}')
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        content = gzip.decompress(packed.read_bytes())
        if len(content) != record['bytes'] or hashlib.sha256(content).hexdigest() != record['sha256']:
            raise ValueError(f'Expanded artifact checksum mismatch: {packed}')
        temporary = target.with_suffix(target.suffix + '.tmp')
        temporary.write_bytes(content)
        temporary.replace(target)

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(obj)
    return obj

def hydrate_mass_replay():
    """Expand only the indexed regular files after verifying every chunk and member."""
    index = json.loads((DATA / 'mass_replay.json').read_text())
    destination = DATA / 'Mass_balance_experiment_20261113'
    if all((destination / row['path']).is_file() and digest(destination / row['path']) == row['sha256'] for row in index['members']):
        return
    temporary = ROOT / 'outputs/mass-replay.tar.gz'
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open('wb') as stream:
        for row in index['chunks']:
            source = DATA / row['path']
            if source.stat().st_size != row['bytes'] or digest(source) != row['sha256']:
                raise ValueError(f'Replay chunk mismatch: {source}')
            stream.write(source.read_bytes())
    if digest(temporary) != index['archive_sha256']:
        raise ValueError('Replay archive checksum mismatch')
    expected = {row['path']: row for row in index['members']}
    seen = set()
    with tarfile.open(temporary, 'r:gz') as archive:
        for member in archive:
            target = destination / member.name
            if not member.isfile() or member.name not in expected or member.name in seen or not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f'Unexpected replay archive member: {member.name}')
            content = archive.extractfile(member).read()
            if len(content) != expected[member.name]['bytes'] or hashlib.sha256(content).hexdigest() != expected[member.name]['sha256']:
                raise ValueError(f'Replay member mismatch: {member.name}')
            if target.exists() and target.read_bytes() != content:
                raise ValueError(f'Existing replay file differs: {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError('Missing replay members')

def verify(output, metrics=False):
    checker = module('release_verifier', DATA / 'verify_release.py')
    manifest = json.loads((DATA / 'manifest.json').read_text())
    for name, expected in manifest['checksums'].items():
        if digest(DATA / name) != expected:
            raise ValueError(f'Payload checksum mismatch: {name}')
    public_identity = json.loads((DATA / 'PUBLIC_METADATA_IDENTITY.json').read_text())
    for record in public_identity['tables']:
        path = DATA / record['path']
        if digest(path) != record['public_sha256']:
            raise ValueError(f'Public metadata identity mismatch: {path}')
        with path.open(encoding='utf-8-sig', newline='') as handle:
            table = list(csv.DictReader(handle))
        columns = [name for name in record['columns'] if name != record['field']]
        scientific = {'columns': columns, 'rows': [[row[name] for name in columns] for row in table]}
        canonical = json.dumps(scientific, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        if hashlib.sha256(canonical).hexdigest() != record['scientific_columns_sha256']:
            raise ValueError(f'Scientific columns differ from the archived source: {path}')
    model, reconstruction = checker.reconstruct(DATA)
    rows = []
    report = {'status': 'PASS', 'release_version': manifest['release_version'],
              'checksum_files_verified': len(manifest['checksums']), 'data': reconstruction,
              'metadata_only_public_tables_verified': len(public_identity['tables']),
              'mass_scoring': checker.score_mass(DATA, model, rows),
              'unified_scoring': checker.score_unified(DATA, model, rows),
              'class_scoring': checker.score_class(DATA, model, rows),
              'scope': 'Historical frozen cohorts and saved-prediction point metrics. Use validate-robustness and statistics for the complete new matrix.'}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    if metrics:
        with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(report, indent=2))

def container(command, output, unit, device, dry_run, historical=False, current_base=False, draws=10000, campaign='historical'):
    output = output.resolve()
    if output == ROOT or output == DATA or DATA in output.parents:
        raise ValueError('Choose an output directory outside reproducibility/')
    output.mkdir(parents=True, exist_ok=True)
    args = ['docker', 'run', '--rm', '--network', 'none', '--read-only',
            '--tmpfs', '/tmp:rw,size=1g', '--cpus', '2', '--memory', '6g',
            '-e', 'MPLCONFIGDIR=/tmp/matplotlib', '-e', 'HOME=/tmp',
            '-v', f'{ROOT}:/repository:ro', '-v', f'{output}:/results:rw',
            '-w', '/repository']
    if device == 'cuda' and (command == 'train' or (command == 'preflight' and campaign == 'robustness')):
        args += ['--gpus', 'all']
    if hasattr(os, 'getuid'):
        args += ['--user', f'{os.getuid()}:{os.getgid()}']
    args += [IMAGE, '/repository/scripts/pipeline.py', command, '--inside-container',
             '--output', '/results', '--unit', unit, '--device', device]
    if historical:
        args += ['--historical']
    if current_base:
        args += ['--current-base']
    args += ['--draws', str(draws), '--campaign', campaign]
    if dry_run:
        print(json.dumps({'command': args, 'starts_training': False}, indent=2))
        return
    subprocess.run(args, check=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['hydrate', 'validate', 'validate-robustness', 'rescore', 'statistics', 'figures', 'train', 'preflight'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--unit', help='Campaign unit; use preflight to list units')
    parser.add_argument('--campaign', choices=['mass', 'class', 'historical', 'robustness'], default='historical', help='Training/preflight procedure: mass952, class494, unified327, or four-input T/R0/R2')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--dry-run', action='store_true', help='Print Docker command without starting computation')
    parser.add_argument('--historical', action='store_true', help='Use historical-only statistics, or archived unified figures with original intervals')
    parser.add_argument('--current-base', action='store_true', help='Render current Figures 1/2/3/S1/S2; pending new robustness figures excluded')
    parser.add_argument('--draws', type=int, default=10000, help='Bootstrap draws for statistics (use 10000 for the manuscript)')
    parser.add_argument('--inside-container', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    defaults = {'mass': 'all', 'class': 'class-GraphConv-outer1', 'historical': 'GraphConv-outer1', 'robustness': 'GraphConv-outer1-P2'}
    args.unit = args.unit or defaults[args.campaign]
    output = args.output or ROOT / 'outputs' / args.command
    if args.command == 'figures' and not (args.historical or args.current_base) and not (DATA / 'current/render_s3.py').is_file():
        raise SystemExit('Final robustness figure integration is pending. Use --current-base for current Figures 1/2/3/S1/S2, or --historical for archived Figures 2/3/S2 with original intervals.')
    if args.historical and args.current_base:
        parser.error('Choose only one figure scope')
    if not args.inside_container:
        hydrate()
        if args.command == 'statistics' and not args.dry_run:
            hydrate_mass_replay()
    if args.command == 'hydrate':
        print('All compressed tables verified and materialized.')
    elif args.command in ['validate', 'rescore']:
        verify(output, metrics=args.command == 'rescore')
    elif not args.inside_container:
        container(args.command, output, args.unit, args.device, args.dry_run, args.historical, args.current_base, args.draws, args.campaign)
    elif args.command == 'validate-robustness':
        subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/verify_robustness.py'), '--output', str(output / 'validation.json')], check=True)
    elif args.command == 'figures':
        if not args.historical:
            if any(output.iterdir()):
                raise ValueError('Figure output directory must be empty')
            subprocess.run([sys.executable, '-B', str(DATA / 'current/render_preserved_figures.py'), '--output', str(output)], check=True)
            subprocess.run([sys.executable, '-B', str(DATA / 'current/render_primary_intervals.py'), '--output', str(output / 'primary')], check=True)
            if not args.current_base:
                subprocess.run([sys.executable, '-B', str(DATA / 'current/render_s3.py'),
                                '--scores', str(DATA / 'Review_statistics_20261114/outputs_with_real_matrix'),
                                '--output', str(output / 'figS3')], check=True)
            print('Current manuscript figures regenerated.' if not args.current_base else 'Current Figures 1/2/3/S1/S2 regenerated.')
            return
        # Invoke the unchanged historical builder with a separate writable output.
        builder = module('historical_reporting', DATA / UNIFIED / 'reporting/build.py')
        if any(output.iterdir()):
            raise ValueError('Figure output directory must be empty')
        (output / 'scores').mkdir()
        builder.build(DATA / UNIFIED / 'outputs/export', output)
        print('Historical unified Figure 2, Figure 3 and S2 rebuilt with score tables.')
    elif args.command == 'statistics':
        source = DATA / 'Review_statistics_20261114'
        command = [sys.executable, '-B', str(source / 'evaluate.py'), '--draws', str(args.draws), '--output', str(output)]
        if not args.historical:
            command += ['--new-export', str(DATA / 'Review_robustness_experiment_20261114/outputs/export')]
        subprocess.run(command, check=True)
        # The archive comparison is defined for the manuscript's 10,000 draws.
        followups = ['archive_comparison.py', 'report_evidence.py'] if args.draws == 10000 else []
        for name in followups:
            subprocess.run([sys.executable, '-B', str(source / name), '--output', str(output)], check=True)
        if args.draws == 10000 and not args.historical:
            subprocess.run([sys.executable, '-B', str(source / 'report_real_matrix.py'), '--output', str(output)], check=True)
        if args.draws != 10000:
            print('Diagnostic draw count only; exact manuscript interval comparison was not requested.')
    else:
        if args.campaign == 'mass':
            source = DATA / 'Mass_balance_experiment_20261113'
            script = str(source / 'analysis/mass_balance_campaign.py')
            options = ['--release-dir', str(source / 'data'), '--feature-dir', str(source / 'features'),
                       '--plan-dir', str(source / 'plans'), '--output-dir', str(output)]
            if args.command == 'preflight' or not (output / 'campaign.json').exists():
                subprocess.run([sys.executable, '-B', script, 'init', *options], check=True)
            if args.command == 'train':
                subprocess.run([sys.executable, '-B', script, 'worker', *options, '--device', args.device], check=True)
            return
        if args.campaign == 'class':
            source = DATA / 'Ionisation_experiment_20261113'
            script = str(source / 'analysis/ionisation_campaign.py')
            plan = output / 'class_plan.json'
            if args.command == 'preflight' or not plan.exists():
                subprocess.run([sys.executable, '-B', script, '--root', str(source), 'plan',
                                '--out', str(plan), '--cohorts', 'class'], check=True)
            if args.command == 'train':
                subprocess.run([sys.executable, '-B', script, '--root', str(source), 'run-unit',
                                '--plan', str(plan), '--unit', args.unit, '--output', str(output), '--device', args.device], check=True)
            return
        if args.campaign == 'robustness':
            subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/train_robustness.py'),
                            'preflight' if args.command == 'preflight' else 'run-unit',
                            '--unit', args.unit, '--device', args.device, '--output', str(output)], check=True)
            return
        campaign = DATA / UNIFIED
        cli = ['preflight', '--root', str(campaign)] if args.command == 'preflight' else [
            'run-unit', '--root', str(campaign), '--unit', args.unit,
            '--output-dir', str(output), '--device', args.device]
        subprocess.run([sys.executable, '-B', str(campaign / 'analysis/unified_campaign.py'), *cli], check=True)

if __name__ == '__main__':
    main()
