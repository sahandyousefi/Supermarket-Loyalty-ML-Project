import warnings

import pytest


def test_warnings_from_the_package_fail_the_tests():
    with pytest.raises(UserWarning):
        warnings.warn_explicit(
            "raised in the package", UserWarning, "x.py", 1, module="loyalty_spend.x"
        )


def test_warnings_from_other_libraries_are_left_alone():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        warnings.warn_explicit("elsewhere", UserWarning, "y.py", 1, module="other")
    assert len(caught) == 1
