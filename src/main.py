"""
main.py — glues parser.py, executor.py, and reporter.py together for a
full end-to-end run: plain-English steps -> LLM/rule parsing -> Playwright
execution -> HTML + JSON report with history trend tracking and
failure screenshots.
"""

import sys
import logging

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

sys.path.insert(0, "src")
from parser import parse_steps
from executor import run_plan
from reporter import generate_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

load_dotenv()

with open("tests/sample_test_steps.txt") as f:
    raw_text = f.read()

print("Parsing steps...")
steps = parse_steps(raw_text)
print(f"Parsed {len(steps)} step(s)\n")

print("Running in browser...")
with sync_playwright() as p:
    browser = p.chromium.launch(headless=False, slow_mo=500)
    page = browser.new_page()
    results = run_plan(page, steps, screenshot_dir="reports/screenshots")
    browser.close()

print("\n--- RESULTS ---")
for r in results:
    print(f"Step {r.step_number} [{r.action}] {r.status.upper()}: {r.message}")

html_path, json_path = generate_report(
    results,
    output_dir="reports",
    test_name="Login Flow — the-internet.herokuapp.com",
    screenshot_dir="reports/screenshots",
    metadata={
        "browser": "chromium",
        "base_url": "https://the-internet.herokuapp.com",
    },
)

print(f"\nReport written:\n  HTML: {html_path}\n  JSON: {json_path}")