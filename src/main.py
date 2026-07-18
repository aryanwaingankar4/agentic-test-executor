"""
main.py — glues parser.py and executor.py together for a first end-to-end run.
"""

import sys
import logging

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

sys.path.insert(0, "src")
from parser import parse_steps
from executor import run_plan

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
    results = run_plan(page, steps)
    browser.close()

print("\n--- RESULTS ---")
for r in results:
    print(f"Step {r.step_number} [{r.action}] {r.status.upper()}: {r.message}")