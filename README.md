# Two groups heatmap

Generate an heatmap comparing overlaps between 2 set of labels in a pandas dataframe. This is useful in single-cell to compare the overlap between two clustering methods or 2 sets of cell type labels.

## Installation

```bash
pip install twogroups_heatmap
```

## Usage

```python
from twogroups_heatmap import intersection_heatmap


df = pandas.DataFrame(
    {
        "cluster1": ["A", "A", "B", "B", "C", "C"],
        "cluster2": ["A", "B", "B", "C", "C", "D"],
    }
)

intersection_heatmap(df, xcol="cluster1", ycol="cluster2")
```
