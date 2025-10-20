#!/usr/bin/env python3
"""
Regenerate Dining Philosopher Databases with Deadlock Detection
"""

from generate_philosopher_databases import PhilosopherDatabaseGenerator
import os

def main():
    """Regenerate databases with deadlock detection."""
    print("=== REGENERATING DATABASES WITH DEADLOCK DETECTION ===\n")
    
    generator = PhilosopherDatabaseGenerator()
    
    # Check if databases already exist
    if os.path.exists(f"{generator.base_dir}/up_to_40_phil_database.db"):
        print("⚠️  Combined database already exists. Regenerating with deadlock detection...")
    
    # Generate databases for all sizes (2-40)
    print("1. Generating individual databases with deadlock detection...")
    for n in range(2, 41):
        print(f"Creating database for {n} philosophers...")
        db_path = generator.create_database(n)
        print(f"✅ Created: {db_path}")
    
    # Generate combined database
    print("\n2. Creating combined database...")
    combined_db_path = generator.create_combined_database(40)
    print(f"✅ Combined database: {combined_db_path}")
    
    # Generate summary report
    print("\n3. Generating summary report...")
    generator.generate_summary_report(40)
    
    print("\n🎉 Database regeneration completed successfully!")
    print(f"\nFiles created in: {generator.base_dir}/")
    print(f"Combined database: up_to_40_phil_database.db")
    print(f"Summary report: database_summary_report.txt")
    
    print("\n📊 New Features:")
    print("- Deadlock detection for all problem sizes")
    print("- is_deadlock, deadlock_type, deadlock_reason columns")
    print("- Automatic deadlock labeling for s4 states")
    print("- Reach property deadlock detection")

if __name__ == "__main__":
    main() 