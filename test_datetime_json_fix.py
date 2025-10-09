#!/usr/bin/env python3
"""Test to verify datetime serialization in GenAIPerfMetric works correctly"""

from datetime import datetime
from aiperf.exporters.json_exporter import GenAIPerfMetric
import json

print("="*70)
print("Testing DateTime Fix for JSON Export")
print("="*70)

# Test 1: Create GenAIPerfMetric with datetime values (the fix)
print("\n[Test 1] Creating GenAIPerfMetric with datetime values...")
dt = datetime(2025, 10, 9, 8, 27, 48, 87848)
try:
    metric = GenAIPerfMetric(
        unit="datetime",
        avg=dt,
        min=datetime(2025, 10, 9, 8, 27, 48, 0),
        max=datetime(2025, 10, 9, 8, 27, 50, 0),
    )
    print("✓ GenAIPerfMetric created successfully with datetime values")
except Exception as e:
    print(f"✗ FAILED: {e}")
    exit(1)

# Test 2: Verify model_dump serializes datetime to ISO string
print("\n[Test 2] Verifying datetime serialization...")
try:
    dumped = metric.model_dump()
    print(f"  avg: {dumped['avg']} (type: {type(dumped['avg']).__name__})")
    print(f"  min: {dumped['min']} (type: {type(dumped['min']).__name__})")
    print(f"  max: {dumped['max']} (type: {type(dumped['max']).__name__})")
    
    assert isinstance(dumped['avg'], str), f"avg should be string, got {type(dumped['avg'])}"
    assert isinstance(dumped['min'], str), f"min should be string, got {type(dumped['min'])}"
    assert isinstance(dumped['max'], str), f"max should be string, got {type(dumped['max'])}"
    assert dumped['avg'] == "2025-10-09T08:27:48.087848", f"Wrong ISO format: {dumped['avg']}"
    print("✓ Datetime values correctly serialized to ISO 8601 strings")
except Exception as e:
    print(f"✗ FAILED: {e}")
    exit(1)

# Test 3: Create GenAIPerfMetric with numeric values (existing behavior)
print("\n[Test 3] Creating GenAIPerfMetric with numeric values...")
try:
    metric2 = GenAIPerfMetric(
        unit="ms",
        avg=123.45,
        min=100,
        max=150.0,
        p50=125.0,
    )
    print("✓ GenAIPerfMetric created successfully with numeric values")
except Exception as e:
    print(f"✗ FAILED: {e}")
    exit(1)

# Test 4: Verify numeric values are not converted
print("\n[Test 4] Verifying numeric values remain unchanged...")
try:
    dumped2 = metric2.model_dump()
    print(f"  avg: {dumped2['avg']} (type: {type(dumped2['avg']).__name__})")
    print(f"  min: {dumped2['min']} (type: {type(dumped2['min']).__name__})")
    print(f"  max: {dumped2['max']} (type: {type(dumped2['max']).__name__})")
    
    assert isinstance(dumped2['avg'], float), f"avg should be float, got {type(dumped2['avg'])}"
    assert isinstance(dumped2['min'], int), f"min should be int, got {type(dumped2['min'])}"
    assert isinstance(dumped2['max'], float), f"max should be float, got {type(dumped2['max'])}"
    assert dumped2['avg'] == 123.45, f"avg value changed: {dumped2['avg']}"
    print("✓ Numeric values correctly preserved")
except Exception as e:
    print(f"✗ FAILED: {e}")
    exit(1)

# Test 5: Mixed datetime and None values
print("\n[Test 5] Creating GenAIPerfMetric with mixed values...")
try:
    metric3 = GenAIPerfMetric(
        unit="datetime",
        avg=datetime(2025, 10, 9, 8, 0, 0),
        max=datetime(2025, 10, 9, 9, 0, 0),
    )
    dumped3 = metric3.model_dump(exclude_unset=True)
    print(f"  Dumped keys: {list(dumped3.keys())}")
    assert dumped3['avg'] == "2025-10-09T08:00:00"
    assert dumped3['max'] == "2025-10-09T09:00:00"
    # Check that unset fields are not included
    assert 'min' not in dumped3
    assert 'p50' not in dumped3
    print("✓ Mixed values correctly handled (unset fields excluded)")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 6: Verify JSON serialization
print("\n[Test 6] Verifying JSON serialization...")
try:
    json_str = json.dumps(dumped)
    json_obj = json.loads(json_str)
    print(f"  JSON serialized successfully")
    print(f"  Parsed back: avg={json_obj['avg']}")
    assert "2025-10-09T08:27:48.087848" in json_str
    print("✓ JSON serialization works correctly")
except Exception as e:
    print(f"✗ FAILED: {e}")
    exit(1)

# Test 7: Test all percentile fields with datetime
print("\n[Test 7] Testing all percentile fields with datetime...")
try:
    dt_base = datetime(2025, 10, 9, 10, 0, 0)
    metric4 = GenAIPerfMetric(
        unit="datetime",
        avg=dt_base,
        p1=dt_base,
        p5=dt_base,
        p10=dt_base,
        p25=dt_base,
        p50=dt_base,
        p75=dt_base,
        p90=dt_base,
        p95=dt_base,
        p99=dt_base,
        std=dt_base,
    )
    dumped4 = metric4.model_dump()
    for field in ['avg', 'p1', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99', 'std']:
        assert isinstance(dumped4[field], str), f"{field} should be string"
        assert dumped4[field] == "2025-10-09T10:00:00", f"{field} has wrong value"
    print("✓ All percentile fields correctly handle datetime")
except Exception as e:
    print(f"✗ FAILED: {e}")
    exit(1)

print("\n" + "="*70)
print("ALL TESTS PASSED! ✓✓✓")
print("="*70)
print("\nThe fix correctly handles:")
print("  ✓ Datetime values in all metric fields")
print("  ✓ Serialization of datetime to ISO 8601 strings")
print("  ✓ Numeric values remain unchanged")
print("  ✓ None values are properly handled")
print("  ✓ JSON serialization works correctly")
print("  ✓ All percentile fields support datetime")
print("\nThis resolves the original error:")
print('  "Input should be a valid number [type=float_type,')
print('   input_value=datetime.datetime(...), input_type=datetime]"')
print("="*70)

