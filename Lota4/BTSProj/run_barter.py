#!/usr/bin/env python3
"""
Script to run the barter system model with a specific Google Sheet
"""
import argparse
from barter_model import load_data, create_barter_model, visualize_exchanges
from gurobipy import GRB

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run barter system optimization on data from Google Sheets')
    parser.add_argument('--sheet-id', type=str, default="1lEnQYqa-WjGByLrgw5wWHOgWTQaP1chga_p7-z00Ac4",
                        help='Google Sheet ID containing the data')
    parser.add_argument('--gid', type=str, default="0",
                        help='GID of the specific sheet to use')
    parser.add_argument('--output', type=str, default="barter_exchanges.png",
                        help='Output file for the visualization')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from sheet {args.sheet_id}, gid {args.gid}...")
    person_item_dict, names, names_all, wants_list = load_data(args.sheet_id, args.gid)
    
    print(f"Loaded {len(names)} people offering items")
    print(f"Found {len(wants_list)} wants")
    
    # Create and solve model
    print("Solving barter optimization model...")
    results = create_barter_model(person_item_dict, wants_list)
    
    # Print results
    status_text = "Optimal" if results['status'] == GRB.OPTIMAL else "Time Limit" if results['status'] == GRB.TIME_LIMIT else "Failed"
    print(f"Status: {status_text}")
    print(f"Objective value: {results['objective_value']:.2f}")
    
    print("\nExchanges:")
    for giver, receiver in results['exchanges']:
        item = person_item_dict[giver]
        print(f"{giver} gives {item} to {receiver}")
    
    print("\nWho Received What:")
    received_items = {}
    for giver, receiver in results['exchanges']:
        item = person_item_dict[giver]
        received_items[receiver] = (giver, item)
    
    for person in sorted(person_item_dict.keys()):
        if person in received_items:
            giver, item = received_items[person]
            print(f"{person} received {item} from {giver}")
        else:
            print(f"{person} did not receive any item")
    
    print("\nCredit Balances (promises for future items):")
    for person, credit in results['credit_balances'].items():
        if abs(credit) > 0.001:  # Only show non-zero credits
            print(f"{person}: {credit:.1f}")
    
    print(f"\nTotal exchanges: {len(results['exchanges'])}")
    
    # Visualize results
    print(f"Generating visualization...")
    plt = visualize_exchanges(results, person_item_dict)
    plt.savefig(args.output)
    print(f"Visualization saved as '{args.output}'")

if __name__ == "__main__":
    main()