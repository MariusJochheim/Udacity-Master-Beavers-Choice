import pandas as pd

from beaverschoice.inventory import generate_sample_inventory
from beaverschoice.logistics import get_supplier_delivery_date


def test_generate_sample_inventory_is_deterministic():
    first = generate_sample_inventory(coverage=0.5, seed=42)
    second = generate_sample_inventory(coverage=0.5, seed=42)
    third = generate_sample_inventory(coverage=0.5, seed=43)

    pd.testing.assert_frame_equal(first, second)
    assert not third.equals(first)


def test_get_supplier_delivery_date_thresholds():
    assert get_supplier_delivery_date("2025-01-10T00:00:00", 5) == "2025-01-10"
    assert get_supplier_delivery_date("2025-01-10", 50) == "2025-01-11"
    assert get_supplier_delivery_date("2025-01-10", 500) == "2025-01-14"
    assert get_supplier_delivery_date("2025-01-10", 5000) == "2025-01-17"
