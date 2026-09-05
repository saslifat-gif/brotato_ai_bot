from copy import deepcopy
from dataclasses import replace
from brotato_ai.control.kill_forecast import KillForecast
from brotato_ai.control.hazards import UnifiedHazardScorer


def state():
    return {'phase': 'combat', 'session': 'a', 'wave': {'number': 1}, 'timestamp_ms': 1000,
            'player': {'position': {'x': 500, 'y': 500}, 'health': 100, 'max_health': 100},
            'combat': {'move_speed': 300, 'weapon_range': 400, 'ranged_count': 6, 'melee_count': 0},
            'enemies': [{'runtime_id': 'one', 'health': 100, 'position': {'x': 750, 'y': 500},
                         'velocity': {'x': 0, 'y': 0}, 'radius': 30}]}


def feed(forecast, payload):
    for i, hp in enumerate((100, 70, 40, 10)):
        payload['timestamp_ms'] = 1000+i*50
        payload['enemies'][0]['health'] = hp
        risks = UnifiedHazardScorer().all_risks(payload)
        filtered, adjusted = forecast.update(payload, risks)
    return risks, filtered, adjusted


def test_sustained_damage_relaxes_only_spacing():
    forecast = KillForecast()
    payload = state()
    risks, filtered, adjusted = feed(forecast, payload)
    assert forecast.clearable_count == 1
    assert not filtered['enemies']
    assert len(payload['enemies']) == 1
    assert adjusted[4].ranged_spacing < risks[4].ranged_spacing
    for a in risks:
        assert replace(adjusted[a], ranged_spacing=risks[a].ranged_spacing) == risks[a]


def test_single_hit_missing_identity_and_strong_enemy_do_not_qualify():
    for change in ({'runtime_id': ''}, {'is_boss': True}, {'is_elite': True}, {'is_charging': True}):
        forecast = KillForecast()
        payload = state()
        payload['enemies'][0].update(change)
        risks, _, adjusted = feed(forecast, payload)
        assert risks == adjusted
    forecast = KillForecast()
    payload = state()
    forecast.update(payload, UnifiedHazardScorer().all_risks(payload))
    payload['timestamp_ms'] += 50
    payload['enemies'][0]['health'] = 1
    risks = UnifiedHazardScorer().all_risks(payload)
    assert forecast.update(payload, risks)[1] == risks


def test_imminent_contact_and_out_of_range_remain_blocked():
    for x in (600, 900):
        forecast = KillForecast()
        payload = state()
        payload['enemies'][0]['position']['x'] = x
        risks, _, adjusted = feed(forecast, payload)
        assert risks == adjusted


def test_stale_damage_and_new_session_clear_prediction():
    for change in ({'timestamp_ms': 1400}, {'session': 'b'}, {'wave': {'number': 2}}):
        forecast = KillForecast()
        payload = state()
        feed(forecast, payload)
        payload['timestamp_ms'] += 50
        payload.update(change)
        risks = UnifiedHazardScorer().all_risks(payload)
        assert forecast.update(payload, risks)[1] == risks
        assert forecast.clearable_count == 0
