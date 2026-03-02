#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add DEBUG logging to simulation.py to trace DG activation
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# Read simulation.py
with open('microgrid/simulation.py', 'r', encoding='utf-8') as f:
    original_code = f.read()

# Check if debugging is already added
if '# DEBUG_DG_TRACE' in original_code:
    print("DEBUG logging already added to simulation.py")
else:
    print("Adding DEBUG logging to simulation.py...")
    
    # Find the DG activation section and add logging
    import re
    
    # Add logging after DG hysteresis check
    pattern = r'(                if avg_soc_frac\[t\] <= ems_policy\.dg_start_soc:\n                    dg_enabled = True)'
    replacement = r'\1\n                    # DEBUG_DG_TRACE\n                    if t >= 936 and t <= 950:\n                        print(f"[t={t}] DG ENABLED: SOC={avg_soc_frac[t]:.3f} <= {ems_policy.dg_start_soc}")'
    
    modified_code = re.sub(pattern, replacement, original_code)
    
    # Add logging at DG activation check
    pattern2 = r'(                # --- DG 補缺（在電池之後） ---\n                if deficit > 0 and fuel_remaining > 0 and dg_enabled:)'
    replacement2 = r'\1\n                    # DEBUG_DG_TRACE\n                    if t >= 936 and t <= 950:\n                        print(f"[t={t}] DG ACTIVATION CHECK: deficit={deficit:.0f}, fuel={fuel_remaining:.0f}, dg_enabled={dg_enabled}, U_DG={U_DG}")'
    
    modified_code = re.sub(pattern2, replacement2, modified_code)
    
    # Add logging inside DG loop
    pattern3 = r'(                    for i in range\(n_DG\):\n                        if U_DG\[i\] == 0:)'
    replacement3 = r'\1\n                            # DEBUG_DG_TRACE\n                            if t >= 936 and t <= 950:\n                                print(f"[t={t}] DG unit {i} UNAVAILABLE (U_DG[{i}]=0)")'
    
    modified_code = re.sub(pattern3, replacement3, modified_code)
    
    # Add logging when DG activates
    pattern4 = r'(                        P_dg_t \+= desired)'
    replacement4 = r'# DEBUG_DG_TRACE\n                        if t >= 936 and t <= 950:\n                            print(f"[t={t}] DG unit {i} ACTIVATED: P_dg={desired:.0f} kW")\n                        \1'
    
    modified_code = re.sub(pattern4, replacement4, modified_code)
    
    if modified_code != original_code:
        # Save modified version
        with open('microgrid/simulation_debug.py', 'w', encoding='utf-8') as f:
            f.write(modified_code)
        print("✓ Created microgrid/simulation_debug.py with DEBUG logging")
        print("\nTo use it, temporarily rename:")
        print("  simulation.py → simulation_orig.py")
        print("  simulation_debug.py → simulation.py")
    else:
        print("✗ Failed to add DEBUG logging (patterns not found)")

print("\n" + "="*100)
print("ALTERNATIVE: Run simulation with manual instrumentation")
print("="*100)

# Create a simpler test that manually checks conditions
print("""
Based on the code analysis, the DG should activate when:

1. NOT grid_available (island mode) ✓
2. avg_soc_frac[t] <= 0.30 → dg_enabled = True ✓
3. deficit > 0 (after battery discharge) ✓
4. fuel_remaining > 0 ✓
5. U_DG[i] == 1 (DG unit available) ✓

The graph shows DG IS available (green) during hours 936-1000.

CRITICAL QUESTION: Is the `dg_enabled` flag being PROPERLY PERSISTED across hours?

Looking at line 266-274:
```python
if not grid_available:
    if dg_enabled:  # ← This checks the CURRENT value
        if avg_soc_frac[t] >= ems_policy.dg_stop_soc:
            dg_enabled = False
    else:  # ← dg_enabled is currently False
        if avg_soc_frac[t] <= ems_policy.dg_start_soc:
            dg_enabled = True
```

Wait! There's NO 'else' clause to MAINTAIN dg_enabled = True if SOC is between 30%-70%!

Let me check if dg_enabled is a LOCAL variable or persists across loop iterations...
""")

# Check if dg_enabled is initialized outside the loop
import re
init_match = re.search(r'(\s*)dg_enabled = False\n.*?for t in range', original_code, re.DOTALL)
if init_match:
    print("\n✓ dg_enabled is initialized OUTSIDE the timestep loop:")
    print("  Line ~139: dg_enabled = False")
    print("  This means it PERSISTS across hours (not reset each hour)")
else:
    print("\n✗ Could not find dg_enabled initialization pattern")

print("""
So the logic should work like a STATE MACHINE:
- Starts: dg_enabled = False
- When SOC drops to ≤30%: dg_enabled = True (and stays True)
- When SOC rises to ≥70%: dg_enabled = False (and stays False)

BUT WAIT! Let me re-read the hysteresis logic more carefully:

```python
if not grid_available:
    if dg_enabled:  # Currently enabled
        if avg_soc_frac[t] >= ems_policy.dg_stop_soc:
            dg_enabled = False
    else:  # Currently disabled
        if avg_soc_frac[t] <= ems_policy.dg_start_soc:
            dg_enabled = True
else:
    dg_enabled = False  # ← Reset when grid available
```

The logic handles 4 states:
1. Grid available → dg_enabled = False (forced)
2. Not grid available + dg_enabled=True + SOC≥70% → dg_enabled = False
3. Not grid available + dg_enabled=False + SOC≤30% → dg_enabled = True  
4. Not grid available + (other conditions) → dg_enabled unchanged

This looks CORRECT!

So the issue must be elsewhere. Let me check if maybe the battery is NOT actually at minimum SOC...
""")

print("\n" + "="*100)
print("HYPOTHESIS: Battery is NOT at minimum SOC during deficit calculation")
print("="*100)
print("""
The diagnostic output showed:
  SOC = 0.100 (10%)
  Battery discharge = 0 kW

This suggests battery is at minimum SOC (10%) and cannot discharge.

But what if the SOC shown in diagnostic is the FINAL SOC after discharge,
not the INITIAL SOC at the start of the hour?

Let me check the diagnostic script...

In my earlier diagnostic (analyze_aug_9_10_power_source.py), I used:
  `avg_soc = sim_result.avg_soc_frac[t]`

This is the avg_soc_frac calculated at the START of hour t (using B[i][t] which equals B[i][t-1]).
So it's the SOC at the BEGINNING of the hour, before any discharge.

But then during the island mode logic, battery tries to discharge and B[i][t] is updated.
So the FINAL SOC might be different from the INITIAL SOC.

The question is: Is B[i][t] updated BEFORE or AFTER the avg_soc_frac[t] calculation?

Looking at the code:
1. Line 209: B[i][t] = B[i][t-1] (initialize to previous hour)
2. Line 214-222: avg_soc_frac[t] = ... (calculate using B[i][t])
3. Line 266-274: Hysteresis logic uses avg_soc_frac[t]
4. Line 349-383: Battery discharge updates B[i][t]

So the hysteresis uses the INITIAL SOC, which is correct!

If initial SOC = 10% ≤ 30%, then dg_enabled = True.
Then battery tries to discharge but cannot (already at min SOC).
Then DG should activate to cover deficit.

Unless... the deficit is actually 0 after battery discharge?
But battery cannot discharge if at min SOC, so deficit should remain > 0.

I'm missing something fundamental here. Let me create a test that prints EVERYTHING...
""")
