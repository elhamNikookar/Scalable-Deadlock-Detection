#!/usr/bin/env python3
"""
Demo: Generate Dining Philosopher Databases for Different Problem Sizes
"""

from generate_philosopher_databases import PhilosopherDatabaseGenerator

def main():
    """Demonstrate database generation for different problem sizes."""
    print("=== DINING PHILOSOPHER DATABASE GENERATION DEMO ===\n")
    
    generator = PhilosopherDatabaseGenerator()
    
    # Generate databases for a few sample sizes
    sample_sizes = [2, 3, 5, 10, 20, 40]
    
    print("1. Generating individual databases for sample sizes...")
    for n in sample_sizes:
        print(f"Creating database for {n} philosophers...")
        db_path = generator.create_database(n)
        print(f"✅ Created: {db_path}")
    
    print("\n2. Creating combined database...")
    combined_db_path = generator.create_combined_database(40)
    print(f"✅ Combined database: {combined_db_path}")
    
    print("\n3. Generating summary report...")
    generator.generate_summary_report(40)
    
    print("\n🎉 Database generation demo completed!")
    print(f"\nFiles created in: {generator.base_dir}/")
    print(f"Combined database: up_to_40_phil_database.db")
    print(f"Summary report: database_summary_report.txt")
    
    print("\n📊 Database Statistics:")
    print("- Individual databases: phil_2_database.db to phil_40_database.db")
    print("- Combined database: up_to_40_phil_database.db")
    print("- Problem sizes: 2 to 40 philosophers")
    print("- Each database contains: states, transitions, configurations")
    print("- Combined database has prefixed state IDs (e.g., phil_5_s0)")

if __name__ == "__main__":
    main() 