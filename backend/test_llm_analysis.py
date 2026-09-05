import os
import unittest
from unittest.mock import patch

from services.llm_analysis import analyze_security_with_llm


class LlmAnalysisTests(unittest.TestCase):
    def test_missing_key_falls_back(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}, clear=True):
            result = analyze_security_with_llm("claim", {"prediction": 0}, None, None, None)
        self.assertEqual(result["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
