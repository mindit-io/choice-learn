# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Choice-Learn is a Python package for discrete choice modeling that combines classical econometric models with modern machine learning approaches. It handles large-scale choice datasets efficiently while minimizing RAM usage. The package supports both single-choice models (e.g., transport mode selection) and multiple-choice models (e.g., shopping baskets).

## Development Commands

### Environment Setup
```bash
# Install dependencies in virtual environment (uses conda by default, or venv with USE_CONDA=false)
make install

# Install pre-commit hooks
make install_precommit
```

### Testing
```bash
# Run all tests with coverage (uses pytest-xdist for parallel execution)
pytest -n auto --cov=choice_learn tests/

# Run specific test file
pytest tests/integration_tests/models/test_conditional_logit.py

# Run with verbose output
pytest -v tests/unit_tests/
```

### Code Quality
```bash
# Linting and formatting are automated via pre-commit hooks
# Manual runs:
ruff check --fix choice_learn/
ruff format choice_learn/

# Security checks
bandit -r choice_learn -x tests
```

### Documentation
```bash
# Serve documentation locally on port 8001
make serve_docs_locally

# Deploy documentation to GitHub Pages
make deploy_docs
```

## Architecture

### Core Data Structures

**ChoiceDataset** (`choice_learn/data/choice_dataset.py`)
- Main data container for single-choice problems
- Uses `ChoiceDatasetIndexer` for efficient memory management through indexing
- Features stored via `Storage` classes (`FeaturesStorage`, `OneHotStorage`) to minimize duplication
- Supports three feature types:
  - `shared_features_by_choice`: session-level features (e.g., customer income)
  - `items_features_by_choice`: item-level features varying by choice (e.g., price)
  - `available_items_by_choice`: item availability masks
- Features can be stored as tuples for multiple feature groups
- Creates from various formats via class methods: `from_single_long_df()`, `from_single_wide_df()`

**TripDataset** (`choice_learn/basket_models/data/basket_dataset.py`)
- Extends ChoiceDataset for basket/bundle choice problems
- Handles multiple simultaneous choices (e.g., grocery shopping trips)
- Groups choices by session to avoid data duplication

### Model Architecture

All models inherit from `ChoiceModel` (`choice_learn/models/base_model.py`):

**Key Base Class Features:**
- Standardized optimizer interface: L-BFGS (default), Adam, SGD, Adamax
- L-BFGS optimization via `_fit_with_lbfgs()` wrapping TensorFlow Probability
- Custom loss function `CustomCategoricalCrossEntropy` with label smoothing
- Model persistence: `save_model()` / `load_model()` using TensorFlow SavedModel format
- Weight specification via `add_coefficients()` and `add_shared_coefficient()`
- Evaluation and reporting with `evaluate()` and `report` property

**Single Choice Models** (`choice_learn/models/`):
- `SimpleMNL`: Basic Multinomial Logit
- `ConditionalLogit`: MNL with item-specific and shared features
- `NestedLogit`: Hierarchical choice structure
- `LatentClassConditionalLogit`: Mixture model with customer segmentation
- `HaloMNL` / `LowRankHaloMNL`: Cross-item effects modeling
- Neural network-based: `RUMnet` (GPU/CPU variants), `TasteNet`, `ResLogit`, `LearningMNL`

**Basket/Multiple Choice Models** (`choice_learn/basket_models/`):
- All inherit from `BaseBasketModel`
- `Shopper`: Probabilistic model with substitutes and complements
- `AleaCarta`: Advanced negative sampling for product interactions (ECML-PKDD 2025)
- `AttentionBasedContextEmbedding`: Sequential basket modeling

**Model-Specific Notes:**
- `RUMnet` has separate GPU/CPU implementations; package auto-selects based on hardware
- Neural models typically use batch training; classical models often use L-BFGS
- Models support regularization (L1, L2, L1L2) via `regularization` parameter

### TensorFlow Integration

**tf_ops.py** (`choice_learn/tf_ops.py`)
- Custom TensorFlow operations and loss functions
- `CustomCategoricalCrossEntropy`: Handles label smoothing and sparse/dense labels
- Utility functions for probability computations and transformations

### Datasets Module

**Academic Datasets** (`choice_learn/datasets/`):
- Pre-loaded datasets: SwissMetro, ModeCanada, Train, Heating, Electricity, TaFeng, Expedia, etc.
- Load functions return pandas DataFrames or ChoiceDataset objects
- Basket datasets in `choice_learn/basket_models/datasets/`: Bakery, synthetic badminton data

### Toolbox

**Optimization Tools** (`choice_learn/toolbox/`):
- `assortment_optimizer.py`: Assortment and pricing optimization algorithms
- Backend implementations: `gurobi_opt.py`, `or_tools_opt.py`
- Uses trained choice models to optimize product offerings

## Testing Strategy

Tests are organized in two directories:
- `tests/unit_tests/`: Unit tests for individual components
- `tests/integration_tests/`: End-to-end model training and evaluation tests

Integration tests follow pattern: `test_{model_name}_on_{dataset_type}.py`

## Code Style

- **Docstring format**: NumPy style (enforced by ruff with `convention = "numpy"`)
- **Line length**: 100 characters
- **Type hints**: Enforced via ruff's ANN rules (with some exceptions for self, cls)
- **Import sorting**: First-party packages: `choice_learn`, `config`, `tests`
- Pre-commit hooks enforce: ruff linting/formatting, trailing whitespace, nbstripout (for notebooks), bandit security checks

## Key Conventions

### Adding a New Model

1. Inherit from `ChoiceModel` (single choice) or `BaseBasketModel` (baskets)
2. Implement required abstract methods:
   - `compute_batch_utility()`: Calculate utility values for items
   - `trainable_weights` property: Return list of TensorFlow variables
3. Define weight specification in `__init__` or via utility methods
4. Create companion notebook in `notebooks/models/` with example usage
5. Add integration test in `tests/integration_tests/models/`
6. Update model list in appropriate `__init__.py`
7. Document with NumPy-style docstrings

### Working with ChoiceDataset

- Access features via indexing: `dataset[i]` returns tuple of (choice, features)
- Use named feature access when available: `dataset.get_choice_features(choice_idx, "feature_name")`
- Batch iteration: `dataset.batch(batch_size)` for model training
- Feature names are crucial for ConditionalLogit and similar models

### TensorFlow Version Constraints

- TensorFlow 2.14-2.16 (Windows uses tensorflow-intel for CPU builds)
- tf_keras <3 required for compatibility
- Models save/load using TensorFlow SavedModel format (not Keras H5)

## Important Notes

- **MAC M1/M2 users**: Install TensorFlow via conda (`conda install -c apple tensorflow`) to avoid crashes
- **Optimization backends**: Either Gurobi or OR-Tools needed for assortment optimization (optional dependencies)
- **GPU detection**: RUMnet automatically switches between GPU/CPU implementation
- **Memory efficiency**: The Storage classes and indexing system are designed to handle large datasets; avoid loading full feature matrices when possible
- **Documentation**: All examples should be Jupyter notebooks; MkDocs automatically processes them for the website

## Citation

When referencing this package:
- Primary paper: Auriau et al. (2024), Journal of Open Source Software, DOI: 10.21105/joss.06899
- For AleaCarta model: Désir et al. (2025), ECML-PKDD proceedings
