from v4.ui_build_policy import RangedSustainTeacher


def choice(stat, value=1):
    return {'category': 'item', 'effects': [{'key':stat, 'value':value}]}


def test_preferred_stats_rank_in_requested_order_at_early_and_late_waves():
    teacher=RangedSustainTeacher()
    for wave in (1, 8, 15):
        scores=[teacher.score_choice(choice(s),wave) for s in
                ('stat_ranged_damage','stat_lifesteal','stat_percent_damage','stat_attack_speed')]
        assert scores == sorted(scores,reverse=True)
        assert teacher.score_choice(choice('stat_armor'),wave) < scores[2]


def test_negative_stat_does_not_receive_priority_bonus():
    teacher=RangedSustainTeacher()
    assert teacher.score_choice(choice('stat_ranged_damage',-1),1) < 0


def test_unaffordable_preferred_item_is_not_purchased():
    teacher=RangedSustainTeacher()
    action={'role':'buy','choice':dict(choice('stat_ranged_damage'),price=100,affordable=False)}
    assert teacher.select({'wave':{'number':3},'counters':{'materials':5}},[action]) is None
