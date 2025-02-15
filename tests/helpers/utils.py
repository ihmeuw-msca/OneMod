def assert_equal_unordered(actual, expected):
    """Recursively compare two data structures, treating lists as unordered collections."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        assert set(actual.keys()) == set(
            expected.keys()
        ), f"Dict keys differ: {actual.keys()} != {expected.keys()}"
        for key in actual:
            assert_equal_unordered(actual[key], expected[key])
    elif isinstance(actual, list) and isinstance(expected, list):
        assert len(actual) == len(
            expected
        ), f"List lengths differ: {len(actual)} != {len(expected)}"
        unmatched_expected_items = expected.copy()
        for actual_item in actual:
            match_found = False
            for expected_item in unmatched_expected_items:
                try:
                    assert_equal_unordered(actual_item, expected_item)
                    unmatched_expected_items.remove(expected_item)
                    match_found = True
                    break
                except AssertionError:
                    continue
            if not match_found:
                raise AssertionError(
                    f"No matching item found for {actual_item} in expected list."
                )
        if unmatched_expected_items:
            raise AssertionError(
                f"Expected items not matched: {unmatched_expected_items}"
            )
    else:
        assert actual == expected, f"Values differ: {actual} != {expected}"
