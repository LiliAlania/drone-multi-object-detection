import json
import unittest
from pathlib import Path


class NotebookTests(unittest.TestCase):
    def test_notebook_is_valid_and_unexecuted(self):
        notebook_path = Path(__file__).parents[1] / "notebooks" / "01_eda.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
        observations = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
            and "**Observations:** TODO" in "".join(cell["source"])
        ]
        self.assertEqual(len(observations), 12)


if __name__ == "__main__":
    unittest.main()
