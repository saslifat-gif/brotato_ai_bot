from brotato_ai.control.arbiter import FinalActionArbiter
from brotato_ai.control.hazards import UnifiedHazardScorer
from brotato_ai.control.recovery import CrowdRecoveryGuard
from brotato_ai.domain.decisions import SafetyDecision

def test_coin_move_survives_only_when_aligned_with_escape():
 for target, expected in (((800,600),4),((200,600),6)):
  shield=UnifiedHazardScorer();guard=CrowdRecoveryGuard(shield=shield)
  guard.state=guard.ESCAPE;guard.safe_zone.target=target
  guard.apply=lambda state,action,**kwargs:SafetyDecision(action,6,0,0)
  arbiter=FinalActionArbiter(safety_shield=shield,crowd_recovery_guard=guard)
  state={'phase':'combat','arena':{'width':1000,'height':1000},'player':{'position':{'x':500,'y':500},'health':100,'max_health':100},'pickups':[{'category':'material','position':{'x':650,'y':500}}]}
  assert arbiter.apply(state,0).decision.applied_action==expected
