from infrastructure.cache import cache


def test_cache_data_handles_list_arguments():
    calls = {"n": 0}

    @cache.cache_data()
    def sum_list(values):
        calls["n"] += 1
        return sum(values)

    assert sum_list([1, 2]) == 3
    assert sum_list([1, 2]) == 3
    assert calls["n"] == 1

    sum_list.clear()
