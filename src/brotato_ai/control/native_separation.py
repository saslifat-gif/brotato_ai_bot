"""Optional C++ separation kernel; cache lives for one arbiter call only."""
import ctypes
import math
import os
from pathlib import Path
from contextvars import ContextVar
from functools import lru_cache, wraps

D = ctypes.c_double
P = ctypes.POINTER(D)
_frame = ContextVar("native_separation_frame", default=None)

@lru_cache(maxsize=4)
def _load(path):
 try:
  lib=ctypes.CDLL(path)
  lib.separation.argtypes=[P,ctypes.c_int,D,D,D,D,D,ctypes.c_int,D,P]
  lib.separation.restype=None
  return lib
 except (OSError, AttributeError):
  return None

def kernel():
 if os.environ.get('BROTATO_NATIVE_SEPARATION','1') != '1': return None
 path=os.environ.get('BROTATO_NATIVE_DLL',str(Path(__file__).resolve().parents[3]/'native'/'separation.dll'))
 return _load(path)

def native_decision_scope(fn):
 @wraps(fn)
 def wrapped(*args,**kwargs):
  if kernel() is None: return fn(*args,**kwargs)
  token=_frame.set({})
  try: return fn(*args,**kwargs)
  finally: _frame.reset(token)
 return wrapped

def diagnostic(payload, action, horizon):
 cache=_frame.get()
 if cache is None or not math.isfinite(horizon): return None
 lib=kernel()
 if lib is None: return None
 key=(id(payload),horizon)
 if key not in cache:
  try: rows=_batch(payload,horizon,lib)
  except (ValueError, TypeError, OverflowError): return None
  # Hold the payload reference so object IDs cannot be reused within a frame.
  cache[key]=(payload,rows)
 return dict(cache[key][1][action])

def _batch(payload, horizon, lib):
 from brotato_ai.control import hazards as h
 c=h._mapping(payload.get('combat'));p=h._mapping(payload.get('player'))
 px,py=h._xy(p.get('position'));pr=max(18.,h._number(p.get('radius'),28),h._number(p.get('width'))*.5,h._number(p.get('height'))*.5)
 speed=max(150.,h._number(c.get('move_speed'),300));wr=h._number(c.get('weapon_range'))
 ranged=h._number(c.get('ranged_count'))>0 and h._number(c.get('ranged_count'))>h._number(c.get('melee_count')) and wr>0
 enemies=[e for e in h._items(payload.get('enemies')) if not bool(e.get('dead'))]
 values=[]
 for e in enemies:values.extend((*h._xy(e.get('position')),*h._xy(e.get('velocity')),max(25.,h._number(e.get('radius'),40))))
 arr=(D*len(values))(*values);out=(D*54)()
 lib.separation(arr,len(enemies),px,py,pr,speed,wr,ranged,horizon,out)
 result=[]
 for k in range(9):
  i=int(out[k*6]);r=dict(zip(('current_distance','predicted_distance','target_distance','closing_rate','radial_dot'),out[k*6+1:k*6+6]))
  r.update(active=i>=0,ranged_active=ranged,enemy_runtime_id=str(enemies[i].get('runtime_id','')) if i>=0 else '')
  result.append(r)
 return result
