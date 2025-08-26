#!/usr/bin/env python3
"""
Configurable CSV Test Data Generator for Batch Email Generator

Usage:
    python scripts/generate_test_data.py [number_of_entries] [output_filename]
    
Examples:
    python scripts/generate_test_data.py 100                          # 100 entries -> uploads/test_data_100.csv
    python scripts/generate_test_data.py 10000                        # 10000 entries -> uploads/test_data_10000.csv
    python scripts/generate_test_data.py 5000 my_custom_data.csv       # 5000 entries -> uploads/my_custom_data.csv
"""

import csv
import random
import sys
import os
from pathlib import Path

# Lists of fake data for generating realistic entries
FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
    "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Christopher", "Karen", "Charles", "Nancy", "Daniel", "Lisa",
    "Matthew", "Betty", "Anthony", "Helen", "Mark", "Sandra", "Donald", "Donna",
    "Steven", "Carol", "Paul", "Ruth", "Andrew", "Sharon", "Joshua", "Michelle",
    "Kenneth", "Laura", "Kevin", "Sarah", "Brian", "Kimberly", "George", "Deborah",
    "Timothy", "Dorothy", "Ronald", "Lisa", "Jason", "Nancy", "Edward", "Karen",
    "Jeffrey", "Betty", "Ryan", "Helen", "Jacob", "Sandra", "Gary", "Donna",
    "Nicholas", "Carol", "Eric", "Ruth", "Jonathan", "Sharon", "Stephen", "Michelle",
    "Larry", "Laura", "Justin", "Sarah", "Scott", "Kimberly", "Brandon", "Deborah",
    "Benjamin", "Dorothy", "Samuel", "Amy", "Gregory", "Angela", "Alexander", "Ashley",
    "Patrick", "Brenda", "Jack", "Emma", "Dennis", "Olivia", "Jerry", "Cynthia",
    "Tyler", "Rachel", "Aaron", "Carolyn", "Jose", "Janet", "Henry", "Virginia",
    "Adam", "Maria", "Douglas", "Heather", "Nathan", "Diane", "Peter", "Julie",
    "Zachary", "Joyce", "Kyle", "Victoria", "Noah", "Kelly", "William", "Christina",
    "Austin", "Joan", "Sean", "Evelyn", "Carl", "Lauren", "Harold", "Judith",
    "Arthur", "Megan", "Lawrence", "Cheryl", "Roger", "Catherine", "Joe", "Frances",
    "Juan", "Samantha", "Jack", "Debra", "Albert", "Rachel", "Wayne", "Carolyn",
    "Ralph", "Janet", "Roy", "Virginia", "Eugene", "Maria", "Louis", "Heather",
    "Philip", "Diane", "Bobby", "Ruth", "Johnny", "Julie", "Mason", "Joyce"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas",
    "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White",
    "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young",
    "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
    "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz", "Parker",
    "Cruz", "Edwards", "Collins", "Reyes", "Stewart", "Morris", "Morales", "Murphy",
    "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan", "Cooper", "Peterson", "Bailey",
    "Reed", "Kelly", "Howard", "Ramos", "Kim", "Cox", "Ward", "Richardson",
    "Watson", "Brooks", "Chavez", "Wood", "James", "Bennett", "Gray", "Mendoza",
    "Hughes", "Price", "Myers", "Long", "Foster", "Sanders", "Ross", "Morales",
    "Powell", "Sullivan", "Russell", "Ortiz", "Jenkins", "Gutierrez", "Perry", "Butler",
    "Barnes", "Fisher", "Henderson", "Coleman", "Simmons", "Patterson", "Jordan", "Reynolds",
    "Hamilton", "Graham", "Kim", "Gonzales", "Alexander", "Ramos", "Wallace", "Griffin",
    "West", "Cole", "Hayes", "Chavez", "Gibson", "Bryant", "Ellis", "Stevens",
    "Murray", "Ford", "Marshall", "Owens", "Mcdonald", "Harrison", "Ruiz", "Kennedy",
    "Wells", "Alvarez", "Woods", "Mendoza", "Castillo", "Olson", "Webb", "Washington"
]

COMPANY_PREFIXES = [
    "Tech", "Digital", "Smart", "Cloud", "Data", "Cyber", "Quantum", "Global",
    "Advanced", "Future", "Innovation", "Dynamic", "Strategic", "Elite", "Premier",
    "Optimized", "Integrated", "Synergy", "Precision", "Velocity", "Apex", "Nexus",
    "Pinnacle", "Summit", "Vertex", "Matrix", "Core", "Prime", "Edge", "Flux",
    "Spark", "Bolt", "Wave", "Flow", "Stream", "Bridge", "Link", "Connect",
    "Unite", "Merge", "Fusion", "Blend", "Mix", "Craft", "Build", "Create",
    "Alpha", "Beta", "Gamma", "Delta", "Omega", "Titan", "Nova", "Stellar",
    "Cosmic", "Infinity", "Meta", "Ultra", "Super", "Mega", "Giga", "Terra"
]

COMPANY_SUFFIXES = [
    "Solutions", "Technologies", "Systems", "Corp", "Inc", "Labs", "Group",
    "Enterprises", "Innovations", "Dynamics", "Analytics", "Consulting", "Services",
    "Partners", "Associates", "Ventures", "Capital", "Holdings", "Industries",
    "Networks", "Platforms", "Studios", "Works", "Hub", "Center", "Institute",
    "Agency", "Firm", "Company", "Organization", "Collective", "Alliance", "Union",
    "Foundation", "Consortium", "Federation", "Syndicate", "League", "Assembly",
    "Council", "Bureau", "Division", "Department", "Authority", "Commission"
]

COMPANY_TYPES = [
    "Software", "Hardware", "Biotech", "Fintech", "Healthtech", "Edtech", "Proptech",
    "Logistics", "Manufacturing", "Retail", "Energy", "Automotive", "Aerospace",
    "Pharmaceutical", "Consulting", "Marketing", "Design", "Media", "Entertainment",
    "Finance", "Insurance", "Real Estate", "Construction", "Agriculture", "Food",
    "Transportation", "Telecommunications", "Security", "Legal", "Research",
    "AI", "Robotics", "Gaming", "Blockchain", "IoT", "VR", "AR", "Machine Learning",
    "Cybersecurity", "Cloud Computing", "DevOps", "Mobile", "Web", "E-commerce"
]

def generate_name():
    """Generate a random full name"""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    return f"{first} {last}"

def generate_company():
    """Generate a random company name"""
    patterns = [
        lambda: f"{random.choice(COMPANY_PREFIXES)} {random.choice(COMPANY_SUFFIXES)}",
        lambda: f"{random.choice(COMPANY_TYPES)} {random.choice(COMPANY_PREFIXES)} {random.choice(COMPANY_SUFFIXES)}",
        lambda: f"{random.choice(LAST_NAMES)} {random.choice(COMPANY_SUFFIXES)}",
        lambda: f"{random.choice(COMPANY_PREFIXES)} {random.choice(COMPANY_TYPES)} {random.choice(COMPANY_SUFFIXES)}",
        lambda: f"{random.choice(COMPANY_PREFIXES)}{random.choice(COMPANY_TYPES)}",
        lambda: f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)} {random.choice(COMPANY_SUFFIXES)}"
    ]
    
    return random.choice(patterns)()

def generate_linkedin_url(name):
    """Generate a realistic LinkedIn URL based on the name"""
    clean_name = name.lower().replace(" ", "")
    
    variations = [
        clean_name,
        clean_name + str(random.randint(1, 999)),
        clean_name.replace("a", "").replace("e", "").replace("i", "").replace("o", "").replace("u", ""),
        clean_name[:8] + str(random.randint(10, 99)),
        name.lower().replace(" ", "-"),
        name.lower().replace(" ", "."),
        clean_name + "-" + str(random.randint(1, 99)),
        clean_name + str(random.randint(1970, 2000)),
        clean_name[:5] + clean_name[-3:],
        clean_name + "-" + random.choice(["dev", "eng", "mgr", "dir", "ceo", "cto", "vp"])
    ]
    
    username = random.choice(variations)
    return f"https://linkedin.com/in/{username}"

def generate_test_csv(filename, num_entries=1000):
    """Generate a CSV file with fake test data"""
    print(f"Generating {num_entries:,} fake entries...")
    
    # Ensure uploads directory exists
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    
    filepath = uploads_dir / filename
    
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['name', 'company', 'linkedin_url']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        
        # Generate entries
        progress_interval = max(1, num_entries // 20)  # Show progress 20 times
        
        for i in range(num_entries):
            name = generate_name()
            company = generate_company()
            linkedin_url = generate_linkedin_url(name)
            
            writer.writerow({
                'name': name,
                'company': company,
                'linkedin_url': linkedin_url
            })
            
            # Progress indicator
            if (i + 1) % progress_interval == 0:
                progress = ((i + 1) / num_entries) * 100
                print(f"Progress: {i + 1:,}/{num_entries:,} entries ({progress:.1f}%)")
    
    file_size = filepath.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    
    print(f"Successfully generated {filepath}")
    print(f"File size: {file_size_mb:.2f} MB ({file_size:,} bytes)")
    print(f"Total entries: {num_entries:,} + header = {num_entries + 1:,} lines")
    
    return str(filepath)

def main():
    """Main function to handle command-line arguments"""
    # Default values
    default_entries = 1000
    
    # Parse command-line arguments
    if len(sys.argv) == 1:
        # No arguments - use defaults
        num_entries = default_entries
        filename = f"test_data_{num_entries}.csv"
    elif len(sys.argv) == 2:
        # Only number of entries provided
        try:
            num_entries = int(sys.argv[1])
            filename = f"test_data_{num_entries}.csv"
        except ValueError:
            print("Error: Number of entries must be a valid integer")
            print_usage()
            sys.exit(1)
    elif len(sys.argv) == 3:
        # Both number and filename provided
        try:
            num_entries = int(sys.argv[1])
            filename = sys.argv[2]
            if not filename.endswith('.csv'):
                filename += '.csv'
        except ValueError:
            print("Error: Number of entries must be a valid integer")
            print_usage()
            sys.exit(1)
    else:
        print("Error: Too many arguments")
        print_usage()
        sys.exit(1)
    
    # Validate inputs
    if num_entries <= 0:
        print("Error: Number of entries must be positive")
        sys.exit(1)
    
    if num_entries > 1000000:
        print("Warning: Generating more than 1 million entries may take a long time and use significant disk space")
        response = input("Do you want to continue? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Operation cancelled")
            sys.exit(0)
    
    # Generate the data
    print(f"Target: {num_entries:,} entries -> uploads/{filename}")
    generated_file = generate_test_csv(filename, num_entries)
    
    # Show recommended batch sizes for testing
    for batch_size in [10, 25, 50, 100, 200]:
        batches = (num_entries + batch_size - 1) // batch_size  # Ceiling division
        print(f"   BATCH_SIZE={batch_size:3d} -> {batches:4d} batches")

def print_usage():
    """Print usage information"""

if __name__ == "__main__":
    main()
