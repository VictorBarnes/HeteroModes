# Bug: `_array_to_gifti` incorrectly splits single-hemisphere data

## Description

The `_array_to_gifti` helper function in `neuromaps/parcellate.py` always splits input data into two parts, regardless of whether the data represents a single hemisphere or both hemispheres. This causes incorrect behavior when parcellating single-hemisphere numpy array data.

## Current Behavior

When calling `Parcellater.transform()` with a numpy array and `hemi='L'` or `hemi='R'`, the data is incorrectly split in half at line 150:

```python
if isinstance(data, np.ndarray):
    if space == 'MNI152':
        # ... error handling
    else:
        data = _array_to_gifti(data)  # Always splits into 2 parts
```

The `_array_to_gifti` function (lines 21-23) always performs a split:

```python
def _array_to_gifti(data):
    """Convert numpy `array` to tuple of gifti images."""
    return tuple(construct_shape_gii(arr) for arr in np.split(data, 2))
```

This means that even when `hemi='L'`, the left hemisphere data gets incorrectly split into two separate GiftiImages, treating the first half as "left" and the second half as "right".

## Expected Behavior

When `hemi='L'` or `hemi='R'` is specified, `_array_to_gifti` should wrap the data in a single-element tuple without splitting. The split should only occur when `hemi=None`, indicating that the data contains both hemispheres.

## Environment

- neuromaps version: 0.0.5
- Python version: 3.11
- Operating system: macOS

