"""Attention only at 1x24x96x96. MAC=2 FLOPs; unary op=1 scalar operation.
Comparisons, rectifiers, memory and padding copies excluded. Not latency.
"""
import csv,json
from pathlib import Path
def ledger():
 c,n=24,96*96;e=c*n;channel=2*(24*10+10*24)
 mac=e*(83+2*299+6)+channel;point=38*e+2156;reference=2*mac+point
 rows=[]
 for name,m,p in [('N3_adaptive_s4',mac,point),('M1_budget_s4',e*(2*25+13*49)+98*n+channel,19*e+3*n+140),('N3_late_pooled_s4',mac,point+6*e),('N3_magnitude_basis_s4',mac,point+6*e)]:
  r={'variant':name,'attention_parameters_per_branch':687,'attention_parameters_two_branches':1374,'MACs_per_branch':m,'scalar_non_MAC_operations':p,'estimated_FLOPs_per_branch':2*m+p,'MAC_difference_percent':100*(m/mac-1),'estimated_FLOP_difference_percent':100*((2*m+p)/reference-1),'status':'ANALYTICAL_ESTIMATE; SERVER_PREFLIGHT_REQUIRED'}
  assert abs(r['MAC_difference_percent'])<=1 and abs(r['estimated_FLOP_difference_percent'])<=1;rows.append(r)
 return rows
def write(path):
 path=Path(path);path.mkdir(parents=True,exist_ok=True);rows=ledger()
 (path/'attention_budget.json').write_text(json.dumps(rows,indent=2)+'\n')
 with (path/'attention_budget.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 return rows
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);print(write(p.parse_args().output))
