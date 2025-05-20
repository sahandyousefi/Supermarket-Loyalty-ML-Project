import numpy as np
import pandas as pd

from loyalty_spend.data.profile import profile_frame, profile_markdown
from loyalty_spend.reporting.markdown import format_value, markdown_table


def test_markdown_table_layout():
    df = pd.DataFrame({"a": [1.23456, np.nan], "b": ["x|y", None]})
    lines = markdown_table(df, digits=2).splitlines()
    assert lines[0] == "| a | b |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| 1.23 | x\\|y |"
    assert lines[3] == "|  |  |"


def test_format_value_keeps_integers_and_text():
    assert format_value(3) == "3"
    assert format_value(np.float32(0.5), 1) == "0.5"
    assert format_value("Europe") == "Europe"


def test_profile_markdown_has_both_sections(loyalty_df):
    text = profile_markdown(profile_frame(loyalty_df))
    assert "### Numeric columns" in text
    assert "### Category columns" in text
    assert "| first_month | 125 |" in text
