"""Test basic functionality to ensure we didn't break anything."""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'choice_learn')

from choice_learn.data.choice_dataset import ChoiceDataset

def test_basic_creation():
    """Test basic dataset creation."""
    print("Testing basic dataset creation...")

    # Create simple dataset
    choices = np.array([0, 1, 2, 0, 1])
    shared_features = np.random.rand(5, 3)
    items_features = np.random.rand(5, 3, 4)

    dataset = ChoiceDataset(
        choices=choices,
        shared_features_by_choice=shared_features,
        items_features_by_choice=items_features,
    )

    assert len(dataset) == 5, "Dataset should have 5 choices"
    assert dataset.get_n_items() == 3, "Dataset should have 3 items"
    print("[OK] Basic dataset creation works")


def test_from_long_df_simple():
    """Test simple long dataframe conversion."""
    print("\nTesting simple long dataframe conversion...")

    # Create simple dataframe
    df = pd.DataFrame({
        'choice_id': [1, 1, 1, 2, 2, 2],
        'item_id': [0, 1, 2, 0, 1, 2],
        'price': [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
        'choice': [0, 0, 0, 1, 1, 1]
    })

    dataset = ChoiceDataset.from_single_long_df(
        df,
        choices_column='choice',
        items_id_column='item_id',
        choices_id_column='choice_id',
        items_features_columns=['price'],
        choice_format='items_id'
    )

    assert len(dataset) == 2, "Dataset should have 2 choices"
    assert dataset.get_n_items() == 3, "Dataset should have 3 items"
    print("[OK] Long dataframe conversion works")


def test_batch_access():
    """Test batch access."""
    print("\nTesting batch access...")

    choices = np.array([0, 1, 2, 0, 1])
    shared_features = np.random.rand(5, 3)
    items_features = np.random.rand(5, 3, 4)

    dataset = ChoiceDataset(
        choices=choices,
        shared_features_by_choice=shared_features,
        items_features_by_choice=items_features,
    )

    # Test indexing
    batch = dataset.batch[[0, 1, 2]]
    assert len(batch) == 4, "Batch should have 4 elements"

    # Test iteration
    count = 0
    for batch in dataset.iter_batch(batch_size=2):
        count += 1
    assert count > 0, "Should have at least one batch"

    print("[OK] Batch access works")


def test_slicing():
    """Test dataset slicing."""
    print("\nTesting dataset slicing...")

    choices = np.array([0, 1, 2, 0, 1])
    shared_features = np.random.rand(5, 3)
    items_features = np.random.rand(5, 3, 4)

    dataset = ChoiceDataset(
        choices=choices,
        shared_features_by_choice=shared_features,
        items_features_by_choice=items_features,
    )

    # Test slicing
    sub_dataset = dataset[[0, 2, 4]]
    assert len(sub_dataset) == 3, "Sub-dataset should have 3 choices"
    assert sub_dataset.get_n_items() == 3, "Sub-dataset should have 3 items"

    print("[OK] Dataset slicing works")


def test_from_wide_df():
    """Test wide dataframe conversion."""
    print("\nTesting wide dataframe conversion...")

    df = pd.DataFrame({
        'price_0': [1.0, 1.5],
        'price_1': [2.0, 2.5],
        'price_2': [3.0, 3.5],
        'choice': [0, 1]
    })

    dataset = ChoiceDataset.from_single_wide_df(
        df,
        items_id=[0, 1, 2],
        items_features_prefixes=['price'],
        choices_column='choice',
        choice_format='items_id'
    )

    assert len(dataset) == 2, "Dataset should have 2 choices"
    assert dataset.get_n_items() == 3, "Dataset should have 3 items"

    print("[OK] Wide dataframe conversion works")


def run_all_tests():
    """Run all basic tests."""
    print("=" * 70)
    print(" " * 15 + "TESTING BASIC FUNCTIONALITY")
    print("=" * 70)

    try:
        test_basic_creation()
        test_from_long_df_simple()
        test_batch_access()
        test_slicing()
        test_from_wide_df()

        print("\n" + "=" * 70)
        print(" " * 15 + "ALL BASIC TESTS PASSED!")
        print("=" * 70)
        return True
    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
