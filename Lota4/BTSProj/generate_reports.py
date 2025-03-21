#!/usr/bin/env python3
"""
Script to generate detailed reports of barter system exchanges
"""
import pandas as pd
import argparse
from barter_model import load_data, create_barter_model, visualize_exchanges
from gurobipy import GRB
import matplotlib.pyplot as plt
import networkx as nx

def generate_exchange_report(results, person_item_dict, filename="exchange_report.csv"):
    """
    Generate a CSV report of all exchanges
    """
    data = []
    
    # For each exchange
    for giver, receiver in results['exchanges']:
        item = person_item_dict[giver]
        data.append({
            'Giver': giver,
            'Receiver': receiver,
            'Item': item
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Exchange report saved to {filename}")
    return df

def generate_receipt_report(results, person_item_dict, filename="receipt_report.csv"):
    """
    Generate a CSV report showing what each person received
    """
    # Create dictionary of who received what
    received_items = {}
    for giver, receiver in results['exchanges']:
        item = person_item_dict[giver]
        received_items[receiver] = (giver, item)
    
    data = []
    # For each person
    for person in person_item_dict.keys():
        if person in received_items:
            giver, item = received_items[person]
            data.append({
                'Person': person,
                'Received_Item': item,
                'From_Person': giver,
                'Status': 'Received item'
            })
        else:
            data.append({
                'Person': person,
                'Received_Item': None,
                'From_Person': None,
                'Status': 'Did not receive item'
            })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Receipt report saved to {filename}")
    return df

def generate_credit_report(results, filename="credit_report.csv"):
    """
    Generate a CSV report of all credit balances
    """
    data = []
    
    # For each person
    for person, credit in results['credit_balances'].items():
        status = "No credit"
        if credit > 0.001:
            status = "Has promise for future item"
        elif credit < -0.001:
            status = "Owes future item"
            
        data.append({
            'Person': person,
            'Credit_Balance': round(credit, 1),
            'Status': status
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Credit report saved to {filename}")
    return df

def generate_detailed_graph(results, person_item_dict, filename="detailed_barter_graph.png"):
    """
    Generate a more detailed graph visualization
    """
    G = nx.DiGraph()
    
    # Prepare data for metrics
    gave_items = set()
    received_items = set()
    for giver, receiver in results['exchanges']:
        gave_items.add(giver)
        received_items.add(receiver)
    
    # Add nodes for people
    for person, item in person_item_dict.items():
        credit = results['credit_balances'].get(person, 0)
        
        # Determine node status
        gave = person in gave_items
        received = person in received_items
        
        if gave and received:
            node_color = 'lightgreen'  # Both gave and received
            status = "Gave & Received"
        elif gave:
            node_color = 'lightblue'   # Only gave
            status = "Only Gave"
        elif received:
            node_color = 'lightyellow' # Only received
            status = "Only Received"
        else:
            node_color = 'lightgray'   # Neither gave nor received
            status = "Did Not Participate"
        
        # Build detailed label
        label = f"{person}\nOffers: {item}\nStatus: {status}"
        if abs(credit) > 0.001:
            if credit > 0:
                label += f"\nCredit: +{credit:.1f}"
            else:
                label += f"\nCredit: {credit:.1f}"
                
        G.add_node(person, label=label, color=node_color, status=status)
    
    # Add edges for exchanges
    for giver, receiver in results['exchanges']:
        G.add_edge(giver, receiver, item=person_item_dict[giver])
    
    # Calculate network metrics
    metrics = {
        'total_exchanges': len(results['exchanges']),
        'participation_rate': (len(gave_items) + len(received_items)) / len(person_item_dict),
        'avg_degree': sum(dict(G.degree()).values()) / len(G),
    }
    
    # Create plot
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, seed=42, k=0.3)
    
    # Get node colors
    node_colors = [G.nodes[n]['color'] for n in G.nodes]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=900, alpha=0.9)
    
    # Draw edges 
    nx.draw_networkx_edges(G, pos, edge_color='darkgray', width=2, arrowsize=20, 
                          connectionstyle='arc3,rad=0.1')
    
    # Draw labels
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight='bold', 
                           font_family='sans-serif')
    
    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'item')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, 
                                font_color='darkblue', label_pos=0.3)
    
    # Add title with metrics
    plt.title(f"Barter System Exchanges\n" +
              f"Total Exchanges: {metrics['total_exchanges']} | " +
              f"Participation: {metrics['participation_rate']*100:.1f}%", 
              fontsize=18)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=15, label='Gave & Received'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=15, label='Only Gave'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightyellow', markersize=15, label='Only Received'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=15, label='Did Not Participate')
    ]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.01, 0.99))
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Detailed graph saved to {filename}")
    
    return metrics

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate detailed reports for barter system')
    parser.add_argument('--sheet-id', type=str, default="1lEnQYqa-WjGByLrgw5wWHOgWTQaP1chga_p7-z00Ac4",
                        help='Google Sheet ID containing the data')
    parser.add_argument('--gid', type=str, default="0",
                        help='GID of the specific sheet to use')
    parser.add_argument('--output-dir', type=str, default="./reports",
                        help='Directory for output files')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from sheet {args.sheet_id}, gid {args.gid}...")
    person_item_dict, names, names_all, wants_list = load_data(args.sheet_id, args.gid)
    
    print(f"Loaded {len(names)} people offering items")
    print(f"Found {len(wants_list)} wants")
    
    # Create and solve model
    print("Solving barter optimization model...")
    results = create_barter_model(person_item_dict, wants_list)
    
    # Check if model is solved
    if results['status'] == GRB.OPTIMAL or results['status'] == GRB.TIME_LIMIT:
        # Generate reports
        exchange_report = generate_exchange_report(
            results, person_item_dict, 
            f"{args.output_dir}/exchange_report.csv")
            
        receipt_report = generate_receipt_report(
            results, person_item_dict,
            f"{args.output_dir}/receipt_report.csv")
            
        credit_report = generate_credit_report(
            results, 
            f"{args.output_dir}/credit_report.csv")
            
        metrics = generate_detailed_graph(
            results, person_item_dict,
            f"{args.output_dir}/detailed_barter_graph.png")
        
        # Generate standard visualization
        plt = visualize_exchanges(results, person_item_dict)
        plt.savefig(f"{args.output_dir}/barter_exchanges.png", dpi=300, bbox_inches='tight')
        
        print("\nSummary:")
        print(f"Total exchanges: {len(results['exchanges'])}")
        print(f"Participation rate: {metrics['participation_rate']*100:.1f}%")
        print(f"Reports saved to {args.output_dir}/")
    else:
        print("Failed to solve the model")

if __name__ == "__main__":
    main()