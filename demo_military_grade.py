#!/usr/bin/env python3
"""
Military-Grade NanoTensor Demonstration
========================================

This script demonstrates the enhanced agency capabilities of the military-grade NanoTensor.
Run this to see autonomous learning, health monitoring, and intelligent decision-making in action.
"""

import sympy as sp
import time
import sys
import warnings
import os

# Import NanoTensor from symbo.py file
import importlib.util
import os
spec = importlib.util.spec_from_file_location("symbo_module", 
    os.path.join(os.path.dirname(__file__), 'symbo.py'))
symbo_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(symbo_module)
NanoTensor = symbo_module.NanoTensor


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def demo_health_monitoring():
    """Demonstrate health monitoring capabilities."""
    print_section("1. Health Monitoring System")
    
    nt = NanoTensor((2, 2), max_order=2, base_vars=['x', 'y'])
    x, y = sp.symbols('x y')
    
    # Set up some expressions
    nt.data[0, 0] = x**2 + y**2
    nt.data[1, 1] = x * y
    
    print("Created NanoTensor with expressions:")
    print(f"  [0,0] = {nt.data[0, 0]}")
    print(f"  [1,1] = {nt.data[1, 1]}")
    print(f"\nInitial state: {nt}")
    
    # Perform operations
    print("\nPerforming 50 operations...")
    for i in range(50):
        nt.diff(x)
        nt.eval_numeric({'x': i * 0.1, 'y': 1.0})
    
    # Check health
    health = nt.health_check()
    print(f"\n✓ Health Status: {health['status'].upper()}")
    print(f"✓ Success Rate: {health['metrics']['success_rate']:.2%}")
    print(f"✓ Cache Hit Rate: {health['metrics']['cache_hit_rate']:.2%}")
    print(f"✓ Avg Operation Time: {health['metrics']['avg_operation_time']:.6f}s")
    print(f"✓ Total Operations: {health['metrics']['total_operations']}")


def demo_learning_system():
    """Demonstrate learning from experiences."""
    print_section("2. Learning & Memory System")
    
    nt = NanoTensor((1,), max_order=2, base_vars=['x', 'y'])
    x, y = sp.symbols('x y')
    nt.data[0] = x**3 + y**3
    
    print("Training the NanoTensor brain...")
    print("Performing multiple operations to build experience...")
    
    # Perform different types of operations
    for i in range(20):
        nt.diff(x)
        nt.eval_numeric({'x': i * 0.5, 'y': 1.0})
        if i % 5 == 0:
            nt.simplify()
    
    # Check what was learned
    agency = nt.get_agency_status()
    print(f"\n✓ Experiences Recorded: {agency['experiences_recorded']}")
    print(f"✓ Patterns Learned: {agency['patterns_learned']}")
    
    health = nt.health_check()
    print("\nLearned Patterns:")
    for op_type, pattern in health['learned_patterns'].items():
        print(f"  • {op_type}:")
        print(f"    - Executed {pattern['count']} times")
        print(f"    - Average duration: {pattern['avg_duration']:.6f}s")
        print(f"    - Success rate: {pattern['success_rate']:.2%}")


def demo_security_validation():
    """Demonstrate security and validation features."""
    print_section("3. Security & Validation Layer")
    
    nt = NanoTensor((1,), max_order=1, base_vars=['x', 'y'])
    x, y = sp.symbols('x y')
    nt.data[0] = x + 2*y
    
    print("Setting security bounds:")
    print("  x: [-10.0, 10.0]")
    print("  y: [-5.0, 5.0]")
    
    nt.set_validation_bounds('x', -10.0, 10.0)
    nt.set_validation_bounds('y', -5.0, 5.0)
    
    # Test valid inputs
    print("\n✓ Testing valid input: x=5.0, y=2.0")
    result = nt.eval_numeric({'x': 5.0, 'y': 2.0})
    print(f"  Result: {result[0]:.2f}")
    
    # Test invalid inputs
    print("\n✗ Testing invalid input: x=15.0 (out of bounds)")
    try:
        nt.eval_numeric({'x': 15.0, 'y': 2.0})
    except ValueError as e:
        print(f"  Caught: {e}")
    
    print("\n✗ Testing invalid input: NaN value")
    try:
        nt.eval_numeric({'x': float('nan'), 'y': 2.0})
    except ValueError:
        print(f"  Caught: Invalid value detected")


def demo_autonomous_recovery():
    """Demonstrate autonomous error recovery."""
    print_section("4. Autonomous Error Recovery")
    
    nt = NanoTensor((1,), max_order=2, base_vars=['x', 'y'])
    x, y = sp.symbols('x y')
    
    # Create a potentially problematic expression
    nt.data[0] = (x**5 + y**5) / (x - y + 0.001)
    
    print("Created complex expression:")
    print(f"  f(x,y) = {nt.data[0]}")
    
    print("\nAttempting differentiation (may trigger auto-recovery)...")
    start = time.time()
    
    try:
        nt.diff(x)
        elapsed = time.time() - start
        print(f"✓ Differentiation succeeded in {elapsed:.4f}s")
        
        # Check if recovery was needed
        health = nt.health_check()
        if health['metrics']['total_operations'] > 0:
            print(f"✓ Success rate: {health['metrics']['success_rate']:.2%}")
    except Exception as e:
        print(f"✗ Failed: {e}")


def demo_cache_performance():
    """Demonstrate intelligent caching."""
    print_section("5. Intelligent Caching System")
    
    nt = NanoTensor((2,), max_order=2, base_vars=['x', 'y'])
    x, y = sp.symbols('x y')
    nt.data[0] = x**2 + y
    nt.data[1] = x * y**2
    
    print("Testing cache performance...")
    
    # First call - cache miss
    print("\nFirst call (cache miss):")
    start = time.time()
    nt.diff_cached('x', 1)
    time1 = time.time() - start
    print(f"  Time: {time1:.6f}s")
    
    # Second call - cache hit
    print("\nSecond call (cache hit):")
    start = time.time()
    nt.diff_cached('x', 1)
    time2 = time.time() - start
    print(f"  Time: {time2:.6f}s")
    
    if time1 > time2:
        speedup = time1 / time2 if time2 > 0 else float('inf')
        print(f"\n✓ Speedup from caching: {speedup:.1f}x")
    
    # Check cache statistics
    health = nt.health_check()
    print(f"✓ Cache hit rate: {health['metrics']['cache_hit_rate']:.2%}")
    print(f"✓ Cache sizes: {health['cache_sizes']}")


def demo_agent_with_brain():
    """Demonstrate using NanoTensor as an agent brain."""
    print_section("6. Agent with NanoTensor Brain")
    
    class SimpleAgent:
        """Simple agent with a NanoTensor brain."""
        
        def __init__(self):
            self.brain = NanoTensor((1,), max_order=1, base_vars=['state'])
            state = sp.Symbol('state')
            # Simple control law: action = -0.5 * state
            self.brain.data[0] = -0.5 * state
            self.brain.set_validation_bounds('state', -100, 100)
        
        def act(self, state_value):
            """Decide action based on state."""
            return self.brain.eval_numeric({'state': state_value})[0]
        
        def get_brain_health(self):
            """Get brain health status."""
            return self.brain.health_check()
    
    print("Creating agent with NanoTensor brain...")
    agent = SimpleAgent()
    print("✓ Agent created with control law: action = -0.5 * state")
    
    print("\nAgent operating for 20 steps:")
    state = 10.0
    for step in range(20):
        action = agent.act(state)
        state = state + action  # Simple dynamics
        
        if step % 5 == 0:
            health = agent.get_brain_health()
            print(f"  Step {step:2d}: state={state:6.2f}, "
                  f"brain_health={health['status']}, "
                  f"ops={health['metrics']['total_operations']}")
    
    print("\nFinal brain status:")
    health = agent.get_brain_health()
    agency = agent.brain.get_agency_status()
    print(f"✓ Health: {health['status']}")
    print(f"✓ Operations: {health['metrics']['total_operations']}")
    print(f"✓ Experiences: {agency['experiences_recorded']}")
    print(f"✓ Recommendations: {agency['recommendations'][0]}")


def demo_persistence():
    """Demonstrate brain persistence (saving/loading)."""
    print_section("7. Persistence (Save/Load Brain)")

    filename = "agent_brain.pkl"

    # 1. Create and train a brain
    print("Training original brain...")
    nt = NanoTensor((1,), max_order=1, base_vars=['x'])
    x = sp.Symbol('x')
    nt.data[0] = 2 * x

    # Generate some experience
    nt.eval_numeric({'x': 10.0})
    nt.eval_numeric({'x': 20.0})

    print(f"Original brain operations: {nt._operation_count}")

    # 2. Save it
    print(f"Saving brain to {filename}...")
    nt.save_brain(filename)

    # 3. Load into new instance
    print("Loading brain into new instance...")
    nt_loaded = NanoTensor.load_brain(filename)

    # 4. Verify state
    print(f"Loaded brain operations: {nt_loaded._operation_count}")
    print(f"Loaded brain experiences: {len(nt_loaded._experience_buffer)}")

    if nt._operation_count == nt_loaded._operation_count:
        print("✓ State persistence verified!")
    else:
        print("✗ State mismatch!")

    # Clean up
    if os.path.exists(filename):
        os.remove(filename)


def demo_anomaly_detection():
    """Demonstrate statistical anomaly detection."""
    print_section("8. Statistical Anomaly Detection")

    nt = NanoTensor((1,), max_order=1, base_vars=['x'])
    nt.set_validation_bounds('x', -1000, 1000)

    print("Training baseline statistics (mean=10, std=1)...")
    import random
    for _ in range(50):
        val = random.gauss(10, 1)
        nt.eval_numeric({'x': val})

    print("Injecting anomaly (value=50, ~40 sigma)...")
    try:
        # This is within bounds [-1000, 1000] but statistically anomalous
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            nt.eval_numeric({'x': 50.0})
            if w:
                for warning in w:
                    print(f"✓ Caught warning: {warning.message}")
            else:
                print("✗ Failed to detect anomaly")
    except Exception as e:
        print(f"Error: {e}")


def demo_optimization():
    """Demonstrate storage optimization (CSE)."""
    print_section("9. Advanced Optimization (CSE)")

    nt = NanoTensor((10,), max_order=1, base_vars=['x'])
    x = sp.Symbol('x')

    # Create repetitive expression
    expr = (x + 1)**10 + (x + 1)**9 + (x + 1)**8
    for i in range(10):
        nt.data[i] = expr

    print("Original storage unoptimized.")
    print("Running optimization...")
    nt.optimize_storage()

    if nt._optimized_data:
        replacements, reduced = nt._optimized_data
        print(f"✓ Optimization successful!")
        print(f"  Replacements found: {len(replacements)}")
        print(f"  Reduced expressions: {len(reduced)}")
    else:
        print("✗ Optimization failed or no common subexpressions found.")


def demo_recommendations():
    """Demonstrate intelligent recommendations."""
    print_section("10. Intelligent Recommendations")
    
    nt = NanoTensor((1,), max_order=2, base_vars=['x'])
    x = sp.Symbol('x')
    nt.data[0] = x**2
    
    print("Performing operations and getting recommendations...")
    
    # Perform various operations
    for i in range(30):
        nt.diff(x)
        nt.eval_numeric({'x': float(i)})
    
    # Get recommendations
    agency = nt.get_agency_status()
    print(f"\nBrain Status: {agency['health']}")
    print(f"Experiences: {agency['experiences_recorded']}")
    print("\nRecommendations:")
    for i, rec in enumerate(agency['recommendations'], 1):
        print(f"  {i}. {rec}")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("  MILITARY-GRADE NANOTENSOR DEMONSTRATION")
    print("  Enhanced with Agency Capabilities")
    print("="*70)
    
    try:
        demo_health_monitoring()
        demo_learning_system()
        demo_security_validation()
        demo_autonomous_recovery()
        demo_cache_performance()
        demo_agent_with_brain()
        demo_persistence()
        demo_anomaly_detection()
        demo_optimization()
        demo_recommendations()
        
        print("\n" + "="*70)
        print("  ✓ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
