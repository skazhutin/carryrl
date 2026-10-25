"""Compare observed outcomes on an identical, ordered episode bank."""

import json
from collections.abc import Mapping, Sequence

DESCRIPTOR_FIELDS = ('task', 'seed', 'instruction', 'policy', 'action_seed', 'episode_index',
                     'difficulty', 'level_index', 'level_id', 'presentation',
                     'initial_frame_sha256')
OPTIONAL_DESCRIPTOR_FIELDS = ('runtime_fingerprint', 'semantic_level_id')
PAIRED_OUTCOMES_VERSION = 'carryrl.paired_outcomes.v1'


def episode_descriptor(record):
    """Preserve legacy protocol fields and optional-field presence exactly.

    The returned mapping deliberately has the same JSON representation as the
    historical overview protocol descriptor. Missing optional fields and explicit
    nulls describe different protocols.
    """
    if not isinstance(record, Mapping) or any(key not in record for key in DESCRIPTOR_FIELDS):
        raise ValueError('неполный протокол эпизода')
    if not isinstance(record['task'], str) or not record['task']:
        raise ValueError('неверная игра в протоколе эпизода')
    descriptor = {key: record[key] for key in DESCRIPTOR_FIELDS}
    descriptor.update({key: record[key] for key in OPTIONAL_DESCRIPTOR_FIELDS if key in record})
    try:
        json.dumps(descriptor, sort_keys=True, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ValueError('повреждённый протокол эпизода') from error
    return descriptor


def _validated(records):
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise ValueError('нужен непустой упорядоченный набор эпизодов')
    identities, result = set(), []
    for record in records:
        descriptor = episode_descriptor(record)
        identity = json.dumps(descriptor, sort_keys=True, allow_nan=False)
        if identity in identities:
            raise ValueError('повторённый эпизод проверки')
        identities.add(identity)
        outcome = record.get('outcome')
        if (not isinstance(outcome, Mapping) or not isinstance(outcome.get('known'), bool)
                or outcome['known'] and not isinstance(outcome.get('success'), bool)):
            raise ValueError('исход эпизода повреждён')
        result.append((identity, descriptor['task'], outcome))
    return result


def _counts():
    return dict.fromkeys(('episodes', 'known_pairs', 'unknown_pairs', 'known_before',
                          'known_after', 'gained', 'lost', 'retained_wins',
                          'retained_nonwins'), 0)


def paired_outcomes(before_records, after_records):
    """Count transitions only when both saved outcomes are known.

    Scores, rewards and unknown outcomes never imply success or failure. Every
    descriptor, its position and its uniqueness must match before any count is
    reported; repeated levels with distinct episode indices remain valid.
    """
    before, after = _validated(before_records), _validated(after_records)
    if len(before) != len(after) or any(a[0] != b[0] for a, b in zip(before, after, strict=False)):
        raise ValueError('протокол/упорядоченный набор эпизодов отличается')
    total, games = _counts(), {}
    for (_, task, left), (_, _, right) in zip(before, after, strict=True):
        game = games.setdefault(task, _counts())
        for counts in (total, game):
            counts['episodes'] += 1
            counts['known_before'] += int(left['known'])
            counts['known_after'] += int(right['known'])
            if not left['known'] or not right['known']:
                counts['unknown_pairs'] += 1
                continue
            counts['known_pairs'] += 1
            category = ('retained_wins' if left['success'] and right['success'] else
                        'lost' if left['success'] else
                        'gained' if right['success'] else 'retained_nonwins')
            counts[category] += 1
    for counts in (total, *games.values()):
        counts['coverage'] = counts['known_pairs'] / counts['episodes']
        counts['net_wins'] = counts['gained'] - counts['lost']
    return {'version': PAIRED_OUTCOMES_VERSION, **total,
            'games': [{'task': task, **counts} for task, counts in games.items()]}
