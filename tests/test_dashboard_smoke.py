"""
Dashboard smoke test using Streamlit's AppTest harness.

This catches the class of bug a unit test on the engine modules CANNOT
catch: errors that only happen inside the Streamlit script itself
(button callbacks, page navigation, widget state issues -- exactly the
two real bugs found and fixed during development: the empty-data crash
and the weight-rebalancing bug).

Run with: pytest tests/test_dashboard_smoke.py -v
(requires `pip install streamlit` -- already in requirements.txt)
"""

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def app():
    at = AppTest.from_file("app/dashboard.py")
    at.run(timeout=60)
    return at


def test_initial_render_has_no_exceptions(app):
    assert list(app.exception) == []


def test_build_portfolio_button_works(app):
    build_buttons = [b for b in app.button if "Build Portfolio" in b.label]
    assert len(build_buttons) == 1
    build_buttons[0].click().run(timeout=60)
    assert list(app.exception) == []


def test_all_pages_navigate_without_error(app):
    build_buttons = [b for b in app.button if "Build Portfolio" in b.label]
    build_buttons[0].click().run(timeout=60)

    for page_label in ["2. Risk Analytics", "3. Backtesting",
                        "4. Risk Attribution", "5. Stress Testing"]:
        radio = app.sidebar.radio[0]
        radio.set_value(page_label)
        app.run(timeout=60)
        assert list(app.exception) == [], f"Exception on page '{page_label}'"


def test_adding_ticker_keeps_weights_at_100_percent(app):
    """Regression test for the exact bug reported: adding a custom ticker
    must auto-rebalance, not leave the total off 100%."""
    def get_total():
        metrics = [m for m in app.metric if "Total weight" in m.label]
        return metrics[0].value

    assert get_total() == "100.0%"

    text_inputs = [t for t in app.text_input if "custom ticker" in t.label]
    text_inputs[0].set_value("AAPL").run(timeout=60)

    assert get_total() == "100.0%"
    assert list(app.exception) == []
