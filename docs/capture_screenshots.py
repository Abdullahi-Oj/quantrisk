import time
from playwright.sync_api import sync_playwright

URL = "http://localhost:8511"
OUT_DIR = "/home/claude/quantrisk/docs/screenshots"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    page.goto(URL)
    page.wait_for_timeout(3000)

    # Page 1: Portfolio Builder -- click "Build Portfolio"
    page.get_by_role("button", name="Build Portfolio").click()
    page.wait_for_timeout(3000)
    page.screenshot(path=f"{OUT_DIR}/1_portfolio_builder.png", full_page=True)
    print("Captured: 1_portfolio_builder.png")

    nav_targets = [
        ("2. Risk Analytics", "2_risk_analytics.png"),
        ("3. Backtesting", "3_backtesting.png"),
        ("4. Risk Attribution", "4_risk_attribution.png"),
        ("5. Stress Testing", "5_stress_testing.png"),
    ]

    for label, filename in nav_targets:
        page.get_by_text(label, exact=True).click()
        page.wait_for_timeout(3500)
        page.screenshot(path=f"{OUT_DIR}/{filename}", full_page=True)
        print(f"Captured: {filename}")

    browser.close()

print("Done.")
