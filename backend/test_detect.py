import unittest
import os
import sys

# Add current directory to path so we can import detect
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_DIR)

from detect import clean_plate_text, MODEL_PATH, DEFAULT_DB_PATH, TEST_IMAGE

class TestBNUVehicleDetection(unittest.TestCase):

    def test_clean_plate_text_normal(self):
        self.assertEqual(clean_plate_text("  LEW-14-1234  "), "LEW-14-1234")
        self.assertEqual(clean_plate_text("LED1234"), "LED1234")

    def test_clean_plate_text_special_chars(self):
        self.assertEqual(clean_plate_text("L.E.D. @1234!"), "LED 1234")
        self.assertEqual(clean_plate_text(""), "NOT DETECTED")
        self.assertEqual(clean_plate_text(None), "NOT DETECTED")

    def test_clean_plate_text_spaces(self):
        self.assertEqual(clean_plate_text("   ABC    123   "), "ABC 123")

    def test_paths_exist(self):
        # Verify directories or template files exist
        self.assertTrue(os.path.exists(TEST_IMAGE), f"Test image not found at: {TEST_IMAGE}")
        self.assertTrue(os.path.exists(os.path.dirname(MODEL_PATH)), f"Model directory not found at: {os.path.dirname(MODEL_PATH)}")

if __name__ == '__main__':
    unittest.main()
