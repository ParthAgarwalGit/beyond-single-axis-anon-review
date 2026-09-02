"""Required T03 test: all-240 prompt-index balance (METHOD_FREEZE §4)."""

from collections import Counter

from src.config import block_question_ids, prompt_index_for


def test_blocks_partition_0_to_239(cfg):
    blocks = block_question_ids(cfg)
    assert set(blocks) == {"E80", "C80-A", "C80-B"}
    all_ids = [q for ids in blocks.values() for q in ids]
    assert len(all_ids) == 240
    assert set(all_ids) == set(range(240))
    for name, ids in blocks.items():
        assert len(ids) == len(set(ids)) == 80, name


def test_duplicate_text_ids_share_c80a(cfg):
    blocks = block_question_ids(cfg)
    assert 7 in blocks["C80-A"] and 227 in blocks["C80-A"]


def test_each_block_uses_each_prompt_index_16_times(cfg):
    for ids in block_question_ids(cfg).values():
        counts = Counter(prompt_index_for(cfg, ids, pos) for pos in range(len(ids)))
        assert counts == {i: 16 for i in range(5)}


def test_all_240_assignments_use_each_prompt_index_48_times(cfg):
    counts = Counter()
    for ids in block_question_ids(cfg).values():
        for pos in range(len(ids)):
            counts[prompt_index_for(cfg, ids, pos)] += 1
    assert counts == {i: 48 for i in range(5)}
    assert sum(counts.values()) == 240
