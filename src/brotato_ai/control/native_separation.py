"""Optional C++ separation kernel; cache lives for one arbiter call only."""
import ctypes
import math
import os
from pathlib import Path
from contextvars import ContextVar
from functools import lru_cache, wraps

D = ctypes.c_double
P = ctypes.POINTER(D)
_DEFAULT_PATH = str(Path(__file__).resolve().parents[3]/"native"/"separation.dll")
_frame = ContextVar("native_separation_frame", default=None)

@lru_cache(maxsize=4)
def _load(path):
 try:
  lib=ctypes.CDLL(path)
  lib.separation.argtypes=[P,ctypes.c_int,D,D,D,D,D,ctypes.c_int,D,P]
  lib.separation.restype=None
  lib.crowd.argtypes=[P,ctypes.c_int,D,D,D,D,D,D];lib.crowd.restype=D
  lib.route.argtypes=[P,ctypes.c_int,P,ctypes.c_int,D,D,D,D,D,D,D];lib.route.restype=D
  lib.coin_progress.argtypes=[P,ctypes.c_int,D,D,ctypes.c_int,P];lib.coin_progress.restype=None
  return lib
 except (OSError, AttributeError):
  return None

def kernel():
 if os.environ.get('BROTATO_NATIVE_SEPARATION','1') != '1': return None
 path=os.environ.get('BROTATO_NATIVE_DLL',_DEFAULT_PATH)
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
 enemies,arr=_packed_enemies(payload.get('enemies'))
 out=(D*54)()
 lib.separation(arr,len(enemies),px,py,pr,speed,wr,ranged,horizon,out)
 result=[]
 for k in range(9):
  i=int(out[k*6]);r=dict(zip(('current_distance','predicted_distance','target_distance','closing_rate','radial_dot'),out[k*6+1:k*6+6]))
  r.update(active=i>=0,ranged_active=ranged,enemy_runtime_id=str(enemies[i].get('runtime_id','')) if i>=0 else '')
  result.append(r)
 return result

def _packed_enemies(enemies):
 from brotato_ai.control import hazards as h
 cache=_frame.get();key=('enemies',id(enemies))
 if cache is not None and key in cache:return cache[key][1:]
 live=[e for e in h._items(enemies) if not bool(e.get('dead'))]
 values=[]
 for e in live:values.extend((*h._xy(e.get('position')),*h._xy(e.get('velocity')),max(25.,h._number(e.get('radius'),40))))
 arr=(D*len(values))(*values)
 if cache is not None:cache[key]=(enemies,live,arr)
 return live,arr

def crowd_risk(enemies,fx,fy,movement,px,py):
 if _frame.get() is None:return None
 lib=kernel()
 if lib is None:return None
 live,arr=_packed_enemies(enemies)
 return lib.crowd(arr,len(live),fx,fy,*movement,px,py)

def route_score(payload,target):
 cache=_frame.get()
 if cache is None:return None
 lib=kernel()
 if lib is None:return None
 key=('route',id(payload))
 if key not in cache:
  rows=[];coins=[]
  for kind in ('enemies','projectiles','attack_indicators'):
   for item in payload.get(kind,()):
    if item.get('dead'):continue
    p=item.get('position',{});v=item.get('velocity',{})
    rows.extend((float(p.get('x',0)),float(p.get('y',0)),float(v.get('x',0)),float(v.get('y',0)),max(12.,float(item.get('radius',40)))+48.))
  for item in payload.get('pickups',()):
   if item.get('category',item.get('kind'))=='material':
    p=item.get('position',{});coins.extend((float(p.get('x',0)),float(p.get('y',0))))
  cache[key]=(payload,(D*len(rows))(*rows),len(rows)//5,(D*len(coins))(*coins),len(coins)//2)
 _,rows,n,coins,nc=cache[key]
 p=payload.get('player',{}).get('position',{});arena=payload.get('arena',{})
 score=lib.route(rows,n,coins,nc,float(p.get('x',0)),float(p.get('y',0)),*target,float(arena.get('width',0)),float(arena.get('height',0)),max(150.,float(payload.get('combat',{}).get('move_speed',300))))
 return (True,None if score == -1. else score)

def coin_rows(payload,shorten):
 from collections.abc import Mapping
 cache=_frame.get()
 if cache is None:return None
 lib=kernel()
 if lib is None:return None
 key=('coins',id(payload),shorten)
 if key not in cache:
  positions=[]
  for item in payload.get('pickups',()):
   if isinstance(item,Mapping) and item.get('category',item.get('kind'))=='material':
    p=item.get('position',{});positions.append((float(p.get('x',0)),float(p.get('y',0))))
  values=[v for p in positions for v in p];arr=(D*len(values))(*values);out=(D*(len(positions)*10))()
  p=payload.get('player',{}).get('position',{})
  lib.coin_progress(arr,len(positions),float(p.get('x',0)),float(p.get('y',0)),int(shorten),out)
  cache[key]=(payload,{p:tuple(out[i*10:i*10+10]) for i,p in enumerate(positions)})
 return cache[key][1]
