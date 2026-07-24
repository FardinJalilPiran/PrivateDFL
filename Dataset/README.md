# Dataset

Binary `.choir_dat` files consumed by `privatedfl.data.read_choir_file`.

## Format

```
int32                       n_features
int32                       n_classes
repeated per sample:
    float32 * n_features    feature vector
    int32                   label (0-indexed)
```

## Shipped here

| File | Samples | Features | Classes |
| --- | --- | --- | --- |
| `UCIHAR_train.choir_dat` | 7,352 | 561 | 6 |
| `UCIHAR_test.choir_dat` | 2,947 | 561 | 6 |

UCI Human Activity Recognition Using Smartphones, preprocessed to the standard
561-feature representation with values already in `[-1, 1]`. The loader applies
L2 row normalisation on top, matching the released implementation.

## Adding MNIST or ISOLET

The paper also evaluates MNIST and ISOLET. Convert them to the same format and
name them `MNIST_train.choir_dat` / `MNIST_test.choir_dat`, then run:

```bash
privatedfl --dataset MNIST
privatedfl --list-datasets    # shows what is present
```
