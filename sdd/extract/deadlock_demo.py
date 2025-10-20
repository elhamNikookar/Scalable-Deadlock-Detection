#!/usr/bin/env python3
"""
Deadlock Detection Demo for Dining Philosopher Problem
"""

from deadlock_detection import DeadlockDetector
from dining_philosopher_database import DiningPhilosopherDatabase

def main():
    """Demonstrate deadlock detection capabilities."""
    print("=== DINING PHILOSOPHER DEADLOCK DETECTION DEMO ===\n")
    
    # Create deadlock detector
    detector = DeadlockDetector()
    
    # Update database with deadlock information
    print("1. Updating database with deadlock labels...")
    detector.update_database_with_deadlocks()
    
    # Show deadlock summary
    print("\n2. Deadlock Analysis Results:")
    summary = detector.get_deadlock_summary()
    
    if summary['total_deadlock_states'] > 0:
        print(f"✅ Found {summary['total_deadlock_states']} deadlock states!")
        
        print("\n📊 Deadlock States:")
        for state in summary['deadlock_states']:
            print(f"  🔴 State {state['state_id']}: {state['deadlock_type']}")
            print(f"     Reason: {state['deadlock_reason']}")
            if state['is_start_state']:
                print("     ⚠️  This is a START STATE!")
            if state['is_final_state']:
                print("     ⚠️  This is a FINAL STATE!")
            if state['has_reach_property']:
                print("     ⚠️  This has reach_2 property!")
            print()
        
        print("📈 Deadlock Type Distribution:")
        for deadlock_type, count in summary['deadlock_type_counts'].items():
            print(f"  • {deadlock_type}: {count} states")
    
    else:
        print("✅ No deadlocks detected in this system!")
    
    # Show prevention rules
    print("\n3. Deadlock Prevention Strategies:")
    rules = detector.create_deadlock_prevention_rules()
    for i, rule in enumerate(rules, 1):
        print(f"  {i}. {rule}")
    
    # Show enhanced queries
    print("\n4. Enhanced Database Queries for Deadlock Analysis:")
    print("""
    -- Find all deadlock states
    SELECT state_id, deadlock_type, deadlock_reason 
    FROM states WHERE is_deadlock = 1;
    
    -- Find transitions leading to deadlocks
    SELECT t.from_state, t.to_state, t.action, s.deadlock_type
    FROM transitions t
    JOIN states s ON t.to_state = s.state_id
    WHERE s.is_deadlock = 1;
    
    -- Find states with resource contention
    SELECT s.state_id, 
           COUNT(CASE WHEN sc.node_type = 'Philosopher' AND sc.node_state = 'hungry' THEN 1 END) as hungry_philosophers,
           COUNT(CASE WHEN sc.node_type = 'Fork' AND sc.node_state = 'available' THEN 1 END) as available_forks
    FROM states s
    JOIN state_configurations sc ON s.state_id = sc.state_id
    GROUP BY s.state_id
    HAVING hungry_philosophers > available_forks;
    """)
    
    # Show ML capabilities
    print("\n5. Machine Learning for Deadlock Detection:")
    print("""
    The database now includes deadlock labels for ML training:
    
    ✅ Deadlock Classification Features:
       - philosopher_count, fork_count
       - hungry_philosophers, available_forks, held_forks
       - deadlock_risk, resource_contention
       - thinking_ratio, hungry_ratio, fork_utilization
    
    ✅ ML Models Available:
       - Deadlock Classification Model
       - Deadlock Risk Prediction
       - Resource Contention Analysis
    
    ✅ Use Cases:
       - Predict deadlock probability for new states
       - Identify deadlock-prone configurations
       - Optimize resource allocation strategies
       - Automated deadlock prevention
    """)
    
    detector.close()
    print("\n🎉 Deadlock detection demo completed successfully!")
    print("\nNext steps:")
    print("1. Run: python ml_data_preparation.py")
    print("2. Run: python deep_learning_example.py")
    print("3. Use the trained models for deadlock prediction")

if __name__ == "__main__":
    main() 