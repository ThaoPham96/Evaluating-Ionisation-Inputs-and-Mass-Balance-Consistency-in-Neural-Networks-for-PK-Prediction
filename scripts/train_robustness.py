"""Fresh local refits with actual-environment identities and unchanged scientific procedures."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1] / 'reproducibility/Review_robustness_experiment_20261114'
sys.path.insert(0, str(ROOT / 'analysis'))
import robustness_campaign as campaign
from combined_train import _hashes
import torch

def bind_local_context(device):
    original = campaign.load_campaign(ROOT)
    environments = {}
    for model in campaign.a.BACKBONES:
        unit = f'{model}-outer1-P0'
        request = campaign.architecture_requests(original, unit)[0][0]
        _, environments[model] = _hashes(request, original.release, original.features, torch.device(device))
    identity = copy.deepcopy(original.plan['identity'])
    identity['expected_environments'] = environments
    identity['environment_image'] = 'public-source runtime; see runtime_recipe/Dockerfile'
    identity['fresh_local_refit'] = dict(
        source_public_campaign_id=original.plan['campaign_id'],
        archived_campaign_id=json.loads((ROOT / 'outputs/export/manifest.json').read_text())['campaign_id'],
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        environment=environments,
        scope='New fits under the actual runtime. Original F reference values remain archived. Exact historical GPU predictions are not promised.')
    references = copy.deepcopy(original.references)
    references['environments'] = environments
    bound = campaign.Campaign(ROOT, dict(campaign_id=campaign.a.digest(identity), identity=identity),
                              original.dataset, original.triads, references)
    original_factory = campaign.request_for
    def request_for(context, unit, *args, **kwargs):
        request = original_factory(context, unit, *args, **kwargs)
        request.pop('fit_id')
        request['reproduction_environment'] = environments[request['backbone']]
        request['reproduction_mode'] = 'fresh_local_refit'
        request['fit_id'] = 'ROB-' + campaign.a.digest(request)[:24]
        return request
    campaign.request_for = request_for
    return bound

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'run-unit'])
    parser.add_argument('--unit', choices=campaign.UNITS, default='GraphConv-outer1-P2')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable; use --device cpu for CPU refits or preflight')
    c = bind_local_context(args.device)
    result = campaign.preflight(c)
    result['runtime_gate'] = 'Actual environment bound into new campaign and every fit request; per-fit runtime consistency enforced.'
    result['environment'] = c.references['environments']
    if args.output.resolve().is_relative_to(ROOT.resolve()):
        parser.error('Refit outputs must be outside the frozen public input snapshot')
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / 'public_refit_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != c.plan:
        raise ValueError('Output already belongs to a different refit plan; choose a new directory')
    campaign.a.write_json(args.output / 'public_refit_plan.json', c.plan)
    campaign.a.write_json(args.output / 'public_refit_preflight.json', result)
    campaign.a.write_json(args.output / 'example_request.json', campaign.architecture_requests(c, args.unit)[0][0])
    if args.command == 'run-unit':
        result = campaign.execute_unit(c, args.unit, args.output, args.device)[0]
        result = {key:value for key,value in result.items() if key not in ('fit_markers', 'conditions')}
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
