"""Audit saved experiment invariants and add a full post-run input inventory."""
import hashlib,json
from pathlib import Path
import numpy as np
from script.merit_design_audit import VARIANTS
from script.run_paper_dataset import discover_data_paths
from script.merit_source_provenance import MERIT_SOURCE_FILES, require_merit_sources

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    root=Path('artifacts/merit-design-audit-20260908')
    manifest=json.loads((root/'manifest.json').read_text())
    assert digest(Path('RCAEval/e2e/merit.py'))==manifest['source_sha256']
    require_merit_sources(manifest)
    assert digest(Path('script/merit_design_audit.py'))==manifest['runner_sha256']
    inventory=[];errors=[];status_counts={};n=0;invariants=0
    for dataset in ['re2-ss','re2-ob','re2-tt']:
      paths=discover_data_paths(Path.cwd(),dataset);assert len(paths)==90
      for path in paths:
        case=path.parent.parent.name+'/'+path.parent.name
        saved=json.loads((root/dataset/(case.replace('/','__')+'.json')).read_text())
        records={r['variant']:r for r in saved['records']};assert set(records)==set(VARIANTS)
        files={}
        for f in [path,path.parent/'inject_time.txt',path.parent/'traces.csv',path.parent/'tracets_lat.csv',path.parent/'tracets_err.csv']:
          files[f.name]={'exists':f.exists()}
          if f.exists():
            files[f.name].update(sha256=digest(f),size=f.stat().st_size,mtime_ns=f.stat().st_mtime_ns)
            if f.name in saved.get('input_sha256',{}):assert files[f.name]['sha256']==saved['input_sha256'][f.name]
        inventory.append({'dataset':dataset,'case':case,'files':files})
        for r in records.values():
          n+=1;status_counts[r['status']]=status_counts.get(r['status'],0)+1
          if r['status']=='error':errors.append({'dataset':dataset,'case':case,**r})
          if r['status']=='ok':
            A=np.asarray(r['coverage_strength_matrix']);assert np.isfinite(A).all()
            assert ((A>=0)&(A<=1)).all()
            ranks=r['service_ranks'];assert len(ranks)==len(set(ranks))
        if records['full']['status']=='ok':
          full=records['full']
          for v in ['singleton','shapley','no_trace_evidence']:
            assert records[v]['game_candidates']==full['game_candidates']
            assert records[v]['coverage_strength_matrix']==full['coverage_strength_matrix']
            assert records[v]['metric_service_evidence']==full['metric_service_evidence']
            invariants+=1
          assert records['no_trace_evidence']['service_ranks']==full['service_ranks']
    dependencies=['RCAEval/e2e/merit.py','RCAEval/io/time_series.py','script/merit_design_audit.py','script/run_paper_dataset.py','RCAEval/benchmark/input_boundaries.py','experiment/protocol.json']
    dependencies = list(dict.fromkeys([*dependencies, *MERIT_SOURCE_FILES]))
    audit={'records':n,'status_counts':status_counts,'same_matrix_comparisons_verified':invariants,'errors':errors,
      'input_inventory_timing':'post-run inventory; initial data/spans/injection hashes verified unchanged; trace-series hashes first recorded here',
      'dependencies':{p:digest(Path(p)) for p in dependencies}}
    (root/'input_inventory.json').write_text(json.dumps(inventory,indent=2))
    (root/'verification.json').write_text(json.dumps(audit,indent=2))
    for src in dependencies:
      dest=root/'source'/src;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(src).read_bytes())
    print(json.dumps({k:v for k,v in audit.items() if k!='dependencies'},indent=2))
    assert not errors

if __name__=='__main__':main()
