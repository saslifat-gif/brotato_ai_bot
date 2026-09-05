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
