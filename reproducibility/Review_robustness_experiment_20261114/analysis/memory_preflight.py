"""Diagnostic only: concurrent largest Chemprop/GraphConv configurations locally."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil
import robustness_campaign as r


def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',choices=['Chemprop','GraphConv'])
    p.add_argument('--output',type=Path,default=Path('outputs/memory_preflight'))
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    if args.model:
        import combined_train as trainer
        c=r.load_campaign()
        config=c.plan['identity']['grid'][args.model][1 if args.model=='Chemprop' else 8]
        req=r.request_for(c,f'{args.model}-outer1-P3',config,strength=.3,inner=1)
        req.pop('campaign_id'); req.update(diagnostic=True,max_epochs=2,fit_id='memory-'+args.model)
        summary=trainer.run_fit(req,c.release,c.features,args.output/args.model,'cuda')
        print(json.dumps(summary)); return
    jobs=[]; peak_rss=0
    for model in ('Chemprop','GraphConv'):
        handle=(args.output/(model+'.log')).open('w')
        jobs.append((subprocess.Popen([sys.executable,__file__,'--model',model,'--output',str(args.output)],stdout=handle,stderr=subprocess.STDOUT),handle))
    while any(job.poll() is None for job,_ in jobs):
        rss=0
        for job,_ in jobs:
            try:
                process=psutil.Process(job.pid)
                rss+=sum(child.memory_info().rss for child in [process]+process.children(recursive=True))
            except psutil.NoSuchProcess: pass
        peak_rss=max(peak_rss,rss); time.sleep(.2)
    for job,handle in jobs:
        handle.close()
        if job.returncode: raise RuntimeError('Memory diagnostic failed; inspect local logs')
    summaries={model:r.a.read_json(args.output/model/'summary.json') for model in ('Chemprop','GraphConv')}
    result=dict(passed=True,concurrent_processes=2,peak_process_rss_sum_bytes=peak_rss,
        script_sha256=r.a.sha(Path(__file__)),trainer_sha256=r.a.sha(r.ROOT/'analysis/combined_train.py'),
        summed_peak_cuda_allocated_bytes=sum(s['timing']['peak_cuda_allocated_bytes'] for s in summaries.values()),
        configurations={model:s['config'] for model,s in summaries.items()},
        environment=summaries['Chemprop']['environment'],
        limitation='Two-epoch diagnostic on local RTX 4070 SUPER, not T4 runtime benchmarking; CUDA allocator peaks exclude driver/context overhead',
        output_directory=str(args.output))
    if peak_rss>12*1024**3 or result['summed_peak_cuda_allocated_bytes']>10*1024**3:
        raise RuntimeError('Two-process memory allowance exceeded')
    r.a.write_json(r.ROOT/'runtime/memory_preflight.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
