import pytest

from loyalty_spend.synthetic.conditional import ConditionalCopulaSynthesizer
from loyalty_spend.synthetic.membership import membership_inference


@pytest.fixture(scope="module")
def parts(train_df):
    data = train_df.drop(columns="customer_id").sample(frac=1.0, random_state=1)
    return data.iloc[:500].reset_index(drop=True), data.iloc[500:].reset_index(
        drop=True
    )


def test_released_copies_are_detected(parts):
    members, others = parts
    result = membership_inference(members, others, members, n_boot=100)
    assert result["auc"] > 0.95 and result["advantage"] > 0.9


def test_conditional_synthesiser_resists_the_attack(parts):
    members, others = parts
    synth = ConditionalCopulaSynthesizer("loyalty_years", seed=3).fit(members)
    result = membership_inference(members, others, synth.sample(len(members)))
    assert result["auc"] < 0.6 and result["advantage"] < 0.15
    assert result["auc_lower"] <= result["auc"] <= result["auc_upper"]
