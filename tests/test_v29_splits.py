from src.scenarios.generator import SPLITS, split_seed_block, split_seeds


def test_v29_seed_domains_are_disjoint_and_ordered():
    names = ['route_dev', 'route_train', 'validation8', 'release12']
    ranges = [set(range(SPLITS[n][0], SPLITS[n][1] + 1)) for n in names]
    for i, left in enumerate(ranges):
        for right in ranges[i + 1:]:
            assert left.isdisjoint(right)
    assert split_seeds('validation8', 3) == [26000, 26001, 26002]
    assert split_seeds('release12', 2) == [27000, 27001]


def test_split_seed_block_accepts_fresh_training_domain():
    assert split_seed_block('route_train', 10, 3) == [25010, 25011, 25012]
