"""Execute a release notebook with rich outputs without launching a TCP kernel.

This supplements, rather than claims to replace, ordinary Jupyter execution.
The shipped notebooks use plain Python cells and can run with either workflow.
"""

import io
import base64
from pathlib import Path
import sys
import nbformat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from IPython.display import display


def png(figure):
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    return buffer.getvalue()


def show(*args, **kwargs):
    for number in plt.get_fignums():
        figure = plt.figure(number)
        display(figure)
        plt.close(figure)


shell = InteractiveShell.instance()
shell.display_formatter.formatters['image/png'].for_type(Figure, png)
plt.show = show
path = Path(sys.argv[1]).resolve()
notebook = nbformat.read(path, as_version=4)
namespace = {"__name__": "__main__"}
count = 0
for index, cell in enumerate(notebook.cells):
    if cell.cell_type != "code":
        continue
    count += 1
    with capture_output(display=True) as captured:
        exec(compile(cell.source, f"{path.name}:cell{index}", "exec"), namespace)
    cell.execution_count = count
    cell.outputs = []
    if captured.stdout:
        cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
    if captured.stderr:
        cell.outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
    for output in captured.outputs:
        data = {mime: base64.b64encode(value).decode('ascii')
                if isinstance(value, bytes) else value
                for mime, value in output.data.items()}
        cell.outputs.append(nbformat.v4.new_output("display_data", data=data,
                                                   metadata=output.metadata))
    print(f"Validated cell {index}", flush=True)
notebook.metadata['validation_execution'] = 'Sequential in-process Python execution; no separate Jupyter kernel'
nbformat.validate(notebook)
nbformat.write(notebook, path)
print(f"Saved {count} executed code cells: {path}")
