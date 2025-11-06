"""Test script to verify items_names fixes and availability detection."""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'choice_learn')

from choice_learn.data.choice_dataset import ChoiceDataset


def test_items_names_from_long_df():
    """Test that items_names is properly stored from from_single_long_df."""
    print("=" * 70)
    print("Test 1: items_names from from_single_long_df")
    print("=" * 70)

    # Create a sample long format dataframe
    df = pd.DataFrame({
        'choice_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'item_id': ['apple', 'banana', 'cherry', 'apple', 'banana', 'cherry',
                    'apple', 'banana', 'cherry'],
        'price': [1.0, 0.5, 1.5, 1.2, 0.6, 1.4, 1.1, 0.55, 1.6],
        'choice': ['apple', 'apple', 'apple', 'banana', 'banana', 'banana',
                   'cherry', 'cherry', 'cherry']
    })

    dataset = ChoiceDataset.from_single_long_df(
        df,
        choices_column='choice',
        items_id_column='item_id',
        choices_id_column='choice_id',
        items_features_columns=['price'],
        choice_format='items_id'
    )

    print(f"[OK] Dataset created successfully")
    print(f"  Number of items: {dataset.get_n_items()}")
    print(f"  Number of choices: {dataset.get_n_choices()}")
    print(f"  Items names: {dataset.get_items_names()}")

    assert dataset.get_items_names() is not None, "items_names should not be None"
    assert len(dataset.get_items_names()) == 3, "Should have 3 items"
    assert list(dataset.get_items_names()) == ['apple', 'banana', 'cherry'], \
        "Items should be sorted alphabetically"

    print("[OK]items_names correctly stored and accessible")
    print()


def test_items_names_from_long_df_optimized():
    """Test that the optimized from_single_long_df works correctly."""
    print("=" * 70)
    print("Test 2: Optimized from_single_long_df performance")
    print("=" * 70)

    # Create a sample long format dataframe
    df = pd.DataFrame({
        'choice_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'item_id': ['apple', 'banana', 'cherry', 'apple', 'banana', 'cherry',
                    'apple', 'banana', 'cherry'],
        'price': [1.0, 0.5, 1.5, 1.2, 0.6, 1.4, 1.1, 0.55, 1.6],
        'choice': ['apple', 'apple', 'apple', 'banana', 'banana', 'banana',
                   'cherry', 'cherry', 'cherry']
    })

    dataset = ChoiceDataset.from_single_long_df(
        df,
        choices_column='choice',
        items_id_column='item_id',
        choices_id_column='choice_id',
        items_features_columns=['price'],
        choice_format='items_id'
    )

    print(f"[OK] Dataset created successfully")
    print(f"  Number of items: {dataset.get_n_items()}")
    print(f"  Number of choices: {dataset.get_n_choices()}")
    print(f"  Items names: {dataset.get_items_names()}")

    assert dataset.get_items_names() is not None, "items_names should not be None"
    assert len(dataset.get_items_names()) == 3, "Should have 3 items"
    assert list(dataset.get_items_names()) == ['apple', 'banana', 'cherry'], \
        "Items should be sorted alphabetically"

    print("[OK]Optimized version works correctly")
    print()


def test_availability_with_zero_features():
    """Test that items with all-zero features are correctly marked as available."""
    print("=" * 70)
    print("Test 3: Availability detection with zero features")
    print("=" * 70)

    # Create dataframe where some items have zero features but are still available
    df = pd.DataFrame({
        'choice_id': [1, 1, 1, 2, 2],
        'item_id': ['apple', 'banana', 'cherry', 'apple', 'banana'],
        'price': [0.0, 0.0, 1.5, 1.2, 0.6],  # apple and banana have zero price in choice 1
        'promotion': [0.0, 0.0, 0.0, 1.0, 0.0],  # all zeros for some items
        'choice': ['apple', 'apple', 'apple', 'banana', 'banana']
    })

    dataset = ChoiceDataset.from_single_long_df(
        df,
        choices_column='choice',
        items_id_column='item_id',
        choices_id_column='choice_id',
        items_features_columns=['price', 'promotion'],
        choice_format='items_id'
    )

    print(f"[OK] Dataset created successfully")
    print(f"  Availability for choice 1: {dataset.available_items_by_choice[0]}")
    print(f"  Availability for choice 2: {dataset.available_items_by_choice[1]}")

    # In choice 1, all 3 items (apple, banana, cherry) are present in dataframe
    # even though apple and banana have all-zero features
    assert dataset.available_items_by_choice[0][0] == 1.0, \
        "Apple should be available in choice 1 (even with zero features)"
    assert dataset.available_items_by_choice[0][1] == 1.0, \
        "Banana should be available in choice 1 (even with zero features)"
    assert dataset.available_items_by_choice[0][2] == 1.0, \
        "Cherry should be available in choice 1"

    # In choice 2, only apple and banana are present
    assert dataset.available_items_by_choice[1][0] == 1.0, \
        "Apple should be available in choice 2"
    assert dataset.available_items_by_choice[1][1] == 1.0, \
        "Banana should be available in choice 2"
    assert dataset.available_items_by_choice[1][2] == 0.0, \
        "Cherry should NOT be available in choice 2"

    print("[OK]Availability correctly detected based on row presence, not feature values")
    print()


def test_items_names_from_wide_df():
    """Test that items_names is properly stored from from_single_wide_df."""
    print("=" * 70)
    print("Test 4: items_names from from_single_wide_df")
    print("=" * 70)

    # Create a sample wide format dataframe
    df = pd.DataFrame({
        'price_apple': [1.0, 1.2, 1.1],
        'price_banana': [0.5, 0.6, 0.55],
        'price_cherry': [1.5, 1.4, 1.6],
        'choice': ['apple', 'banana', 'cherry']
    })

    items_id = ['apple', 'banana', 'cherry']

    dataset = ChoiceDataset.from_single_wide_df(
        df,
        items_id=items_id,
        items_features_prefixes=['price'],
        choices_column='choice',
        choice_format='items_id'
    )

    print(f"[OK] Dataset created successfully")
    print(f"  Number of items: {dataset.get_n_items()}")
    print(f"  Number of choices: {dataset.get_n_choices()}")
    print(f"  Items names: {dataset.get_items_names()}")

    assert dataset.get_items_names() is not None, "items_names should not be None"
    assert len(dataset.get_items_names()) == 3, "Should have 3 items"
    assert list(dataset.get_items_names()) == items_id, \
        "Items should match the provided items_id"

    print("[OK]items_names correctly stored and accessible")
    print()


def test_items_names_preserved_in_slicing():
    """Test that items_names is preserved when creating sub-datasets."""
    print("=" * 70)
    print("Test 5: items_names preserved in dataset slicing")
    print("=" * 70)

    # Create a sample dataset
    df = pd.DataFrame({
        'choice_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'item_id': ['apple', 'banana', 'cherry', 'apple', 'banana', 'cherry',
                    'apple', 'banana', 'cherry'],
        'price': [1.0, 0.5, 1.5, 1.2, 0.6, 1.4, 1.1, 0.55, 1.6],
        'choice': ['apple', 'apple', 'apple', 'banana', 'banana', 'banana',
                   'cherry', 'cherry', 'cherry']
    })

    dataset = ChoiceDataset.from_single_long_df(
        df,
        choices_column='choice',
        items_id_column='item_id',
        choices_id_column='choice_id',
        items_features_columns=['price'],
        choice_format='items_id'
    )

    # Create a sub-dataset
    sub_dataset = dataset[[0, 2]]  # Select choices 1 and 3

    print(f"[OK] Original dataset: {dataset.get_n_choices()} choices")
    print(f"[OK] Sub-dataset: {sub_dataset.get_n_choices()} choices")
    print(f"  Original items_names: {dataset.get_items_names()}")
    print(f"  Sub-dataset items_names: {sub_dataset.get_items_names()}")

    assert sub_dataset.get_items_names() is not None, \
        "items_names should be preserved in sub-dataset"
    assert list(sub_dataset.get_items_names()) == list(dataset.get_items_names()), \
        "items_names should be the same in sub-dataset"

    print("[OK]items_names correctly preserved in sub-dataset")
    print()


def run_all_tests():
    """Run all tests."""
    print("\n")
    print("=" * 70)
    print(" " * 15 + "TESTING ITEMS_NAMES FIXES")
    print("=" * 70)
    print()

    try:
        test_items_names_from_long_df()
        test_items_names_from_long_df_optimized()
        test_availability_with_zero_features()
        test_items_names_from_wide_df()
        test_items_names_preserved_in_slicing()

        print("=" * 70)
        print(" " * 20 + "ALL TESTS PASSED!")
        print("=" * 70)
        print()
        return True
    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        print()
        return False
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
