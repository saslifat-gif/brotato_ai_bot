import pytest
from brotato_ai.control import native_separation as n
from brotato_ai.control.hazards import enemy_separation_diagnostics


def frame():
 return {'player':{'position':{'x':200,'y':200}},'combat':{'ranged_count':6,'weapon_range':400,'move_speed':300},'enemies':[{'runtime_id':'a','position':{'x':300,'y':300}}]}


def test_unavailable_library_falls_back(monkeypatch):
 monkeypatch.setenv('BROTATO_NATIVE_DLL','does-not-exist.dll')
 assert n.kernel() is None
 assert n.native_decision_scope(enemy_separation_diagnostics)(frame(),0)['active']


def test_cache_is_scoped_and_released_even_after_error(monkeypatch):
 monkeypatch.setattr(n,'kernel',lambda:object())
 calls=[]
 def batch(p,h,lib):
  calls.append(p['value']);return [{'value':p['value']} for _ in range(9)]
 monkeypatch.setattr(n,'_batch',batch)
 @n.native_decision_scope
 def check(p):
  a=n.diagnostic(p,0,.45);a['value']=-1
  return n.diagnostic(p,1,.45)['value']
 p={'value':1};assert check(p)==1
 p['value']=2;assert check(p)==2
 assert calls==[1,2]
 assert n._frame.get() is None
 @n.native_decision_scope
 def broken():raise RuntimeError('test')
 with pytest.raises(RuntimeError):broken()
 assert n._frame.get() is None


def test_installed_kernel_matches_python_and_refreshes_mutated_frame(monkeypatch):
 if n.kernel() is None:pytest.skip('optional library absent')
 p=frame()
 for x in (300,100,600):
  p['enemies'][0]['position']['x']=x
  expected=[enemy_separation_diagnostics(p,a) for a in range(9)]
  @n.native_decision_scope
  def native():return [enemy_separation_diagnostics(p,a) for a in range(9)]
  for a,b in zip(expected,native()):
   for key,value in a.items():
    if isinstance(value,float):assert b[key]==pytest.approx(value,abs=1e-10)
    else:assert b[key]==value


def test_expanded_geometry_and_decisions_match_python(monkeypatch):
 import random
 from brotato_ai.control.arbiter import FinalActionArbiter
 from brotato_ai.control.hazards import UnifiedHazardScorer
 from brotato_ai.control.recovery import CrowdRecoveryGuard
 from brotato_ai.control.safe_zone import SafeZonePlanner
 from brotato_ai.control.materials import MaterialTargetTracker
 if n.kernel() is None:pytest.skip('optional library absent')
 rng=random.Random(82)
 for count in (0,1,12,56):
  for _ in range(8):
   p=frame();p.update(phase='combat',arena={'width':1000,'height':1000},wave={'number':15})
   p['player'].update(health=100,max_health=100)
   p['enemies']=[{'position':{'x':rng.uniform(0,1000),'y':rng.uniform(0,1000)},'velocity':{'x':rng.uniform(-100,100),'y':rng.uniform(-100,100)},'radius':30} for i in range(count)]
   p['projectiles']=[{'position':{'x':rng.uniform(0,1000),'y':rng.uniform(0,1000)},'velocity':{'x':rng.uniform(-300,300),'y':rng.uniform(-300,300)},'radius':12} for i in range(4)]
   p['pickups']=[{'category':'material','position':{'x':rng.uniform(0,1000),'y':rng.uniform(0,1000)}} for i in range(20)]
   targets=[(200.,200.),(0.,0.),(500.,500.),(150.,250.)]
   expected=[SafeZonePlanner()._route_score(p,t) for t in targets]
   @n.native_decision_scope
   def routes():return [SafeZonePlanner()._route_score(p,t) for t in targets]
   for a,b in zip(expected,routes()):
    if a is None:assert b is None
    else:assert b==pytest.approx(a,abs=1e-10)
   traces=[]
   for mode in ('0','1'):
    monkeypatch.setenv('BROTATO_NATIVE_SEPARATION',mode)
    shield=UnifiedHazardScorer();arb=FinalActionArbiter(safety_shield=shield,crowd_recovery_guard=CrowdRecoveryGuard(shield=shield))
    traces.append(arb.apply(p,4,control_interval_ms=42))
   a,b=traces
   assert a.decision.applied_action==b.decision.applied_action
   for action in a.all_risks:
    assert a.all_risks[action].to_dict()==pytest.approx(b.all_risks[action].to_dict(),abs=1e-10)
