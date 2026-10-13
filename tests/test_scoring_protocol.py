import hashlib
import json

from carryrl.envs.atomic_suite import SCORING_VERSION, SUITE_VERSION
from carryrl.training.sequence import (
    sequence_evaluation_protocol,
    sequence_evaluation_summaries,
)
from carryrl.training.sequence_config import SequenceRunConfig


def test_scoring_rules_are_part_of_evaluation_identity():
    protocol = sequence_evaluation_protocol(SequenceRunConfig())
    assert protocol['scoring_version'] == SCORING_VERSION
    assert protocol['suite_version'] == SUITE_VERSION
    legacy = {k: v for k, v in protocol.items() if k not in {'suite_version', 'scoring_version'}}
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    assert digest(protocol) != digest(legacy)


def test_perfect_score_requires_visible_success_and_known_outcome():
    records = [
        {'policy': 'frozen_deterministic', 'task': 'navigation', 'truncated': False,
         'level_id': str(i), 'semantic_level_id': str(i), 'outcome': outcome}
        for i, outcome in enumerate([
            {'known': True, 'success': True, 'failure': False, 'score': 5},
            {'known': True, 'success': True, 'failure': False, 'score': 4},
            {'known': True, 'success': False, 'failure': True, 'score': 5},
            {'known': False, 'success': True, 'failure': False, 'score': 5},
        ])
    ]
    result = sequence_evaluation_summaries(
        records, tasks=['navigation'], policies=['frozen_deterministic'])['frozen_deterministic']
    assert result['outcomes']['success'] == 2
    assert result['perfect_score_successes'] == 1
    assert result['perfect_score_rate'] == .25
    assert result['tasks']['navigation']['perfect_score_successes'] == 1
