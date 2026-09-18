import numpy as np
import pandas as pd
from src.research_revision import point_loss, weekly_interval, subset_name


def test_bootstrap_zero_difference_is_zero():
    t=pd.Series(pd.date_range('2026-01-01', periods=30, tz='UTC'))
    ci=weekly_interval(np.zeros(30), t, samples=100)
    assert ci['estimate']==ci['ci95_low']==ci['ci95_high']==0


def test_bootstrap_preserves_constant_and_seed():
    t=pd.Series(pd.date_range('2026-01-01', periods=30, tz='UTC'))
    ci=weekly_interval(np.ones(30)*.1, t, samples=100)
    assert np.isclose(ci['ci95_low'], .1) and np.isclose(ci['ci95_high'], .1)
    assert weekly_interval(np.arange(30),t,samples=100)==weekly_interval(np.arange(30),t,samples=100)


def test_loss_and_subset_names():
    assert np.allclose(point_loss([0,1],[.5,.5]), np.log(2))
    assert subset_name(['cohesion','team','individual'])=='CTIS'
    assert subset_name([])=='C'
