from copy import deepcopy
from brotato_ai.control.movement_calibration import calibrate_startup_speed
from brotato_ai.control.hazards import UnifiedHazardScorer

def test_bad_startup_speed_does_not_lock_center_movement():
 state={'phase':'combat','player':{'position':{'x':1024,'y':768}},'arena':{'width':2048,'height':1536},'combat':{'move_speed':1500,'move_speed_source':'stat_fallback'},'projectile_paths':{'boundary_action_risk':[1]*9,'action_risk':[.1]*9}}
 before=deepcopy(state);fixed=calibrate_startup_speed(state)
 assert state==before and fixed['combat']['move_speed']==300
 assert fixed['projectile_paths']['action_risk']==[.1]*9
 assert all(r.boundary_total==0 for r in UnifiedHazardScorer().all_risks(fixed).values())
 state['combat']['move_speed_source']='measured_velocity'
 assert calibrate_startup_speed(state) is state

def test_real_wall_still_blocks_outward_motion():
 state={'phase':'combat','player':{'position':{'x':20,'y':500}},'arena':{'width':2048,'height':1536},'combat':{'move_speed':1500,'move_speed_source':'stat_fallback'}}
 risks=UnifiedHazardScorer().all_risks(calibrate_startup_speed(state))
 assert risks[3].boundary_total>risks[4].boundary_total
