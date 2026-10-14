import ast,hashlib,json,sys
from pathlib import Path
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'src'))
from paired_bootstrap import self_test
from budget_ledger import ledger
self_test();ledger();text=(r/'src/training_functions.py').read_text();tree=ast.parse(text)
for name,digest in json.loads((r/'docs/training_function_hashes.json').read_text()).items():
 n=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
 assert hashlib.sha256(ast.get_source_segment(text,n).encode()).hexdigest()==digest
for p in r.rglob('*.py'):ast.parse(p.read_text())
assert not list(r.rglob('validation_predictions_*.csv'))
assert not any(p.suffix in ('.pt','.pth','.png','.jpg','.jpeg','.tif','.xlsx') for p in r.rglob('*') if p.is_file())
print('PUBLIC_SOURCE_AND_CPU_ANALYSIS_CHECKS_PASS; GPU_ADAPTER_NOT_RERUN_LOCALLY')
