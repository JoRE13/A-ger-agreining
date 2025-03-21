"""
Run the barter system model with data from Google Sheets
"""

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pulp as pl
from barter_model import (
    load_data,
    create_barter_model,
    visualize_exchanges,
    analyze_results
)

def main():
    # Google Sheets ID and sheet ID
    sheet_id = "1lEnQYqa-WjGByLrgw5wWHOgWTQaP1chga_p7-z00Ac4"
    gid = "0"  # Main data
    # gid = "2115678362"  # Test data
    
    # Load data
    print("Loading data from Google Sheets...")
    person_item_dict, names, names_all, wants_list = load_data(sheet_id, gid)
    
    print(f"Loaded {len(names)} participants offering items")
    print(f"Found {len(wants_list)} desired exchanges")
    
    # Initialize previous credits (can be modified to load from file)
    previous_credits = {person: 0 for person in names}
    
    # Create and solve the model
    print("\nOptimizing barter exchanges...")
    model, exchanges, credit_balances, used_credits = create_barter_model(
        person_item_dict, wants_list, previous_credits, alpha=0.1
    )
    
    # Print results
    print(f"\nOptimization Status: {pl.LpStatus[model.status]}")
    print("\nOptimized Exchanges:")
    for giver, receiver, item in exchanges:
        print(f"{giver} gives {item} to {receiver}")
    
    print("\nCredit Balances (positive means promised future items):")
    for person, credit in credit_balances.items():
        if abs(credit) > 0.01:  # Only show non-zero credits
            print(f"{person}: {credit:.2f}")
    
    # Analyze results
    print("\nAnalysis:")
    analysis = analyze_results(model, exchanges, credit_balances, person_item_dict, wants_list)
    
    print(f"Total exchanges: {analysis['num_exchanges']}")
    print(f"Participation rate: {analysis['participation_rate']:.1f}%")
    print(f"People with credits: {analysis['credit_holders']}")
    print(f"Percentage of wants fulfilled: {analysis['wants_fulfilled']:.1f}%")
    
    # Visualize the exchanges
    print("\nCreating visualization...")
    G = visualize_exchanges(exchanges, credit_balances, person_item_dict)
    plt.savefig("barter_network.png", dpi=300, bbox_inches='tight')
    print("Visualization saved as 'barter_network.png'")
    
    # Save results to CSV
    result_df = pd.DataFrame(exchanges, columns=['Giver', 'Receiver', 'Item'])
    result_df.to_csv("barter_exchanges.csv", index=False)
    print("Exchange results saved as 'barter_exchanges.csv'")
    
    credits_df = pd.DataFrame([{"Person": person, "Credit": credit} 
                               for person, credit in credit_balances.items() 
                               if abs(credit) > 0.01])
    credits_df.to_csv("credit_balances.csv", index=False)
    print("Credit balances saved as 'credit_balances.csv'")
    
    # Save analysis to CSV
    analysis_df = pd.DataFrame([analysis])
    analysis_df.to_csv("barter_analysis.csv", index=False)
    print("Analysis saved as 'barter_analysis.csv'")
    

if __name__ == "__main__":
    main()