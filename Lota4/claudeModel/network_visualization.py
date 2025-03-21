"""
Network visualization module for the barter system.
This module provides additional visualization tools beyond the basic network graph.
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def create_credit_distribution_plot(credit_balances):
    """
    Create a visualization of credit distribution among participants.
    
    Parameters:
    -----------
    credit_balances : dict
        Dictionary mapping persons to their credit balances
        
    Returns:
    --------
    fig : matplotlib figure
        The generated figure
    """
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame([
        {"Person": person, "Credit": credit} 
        for person, credit in credit_balances.items()
        if abs(credit) > 0.01  # Only include non-zero credits
    ])
    
    if len(df) == 0:
        # No credits to display
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No credits in the system", 
                ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig
    
    # Sort by credit value
    df = df.sort_values('Credit', ascending=False)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bars with color based on credit value
    bars = ax.bar(df['Person'], df['Credit'])
    
    # Color positive credits blue, negative credits red
    for i, bar in enumerate(bars):
        if df.iloc[i]['Credit'] > 0:
            bar.set_color('royalblue')
        else:
            bar.set_color('firebrick')
    
    # Add labels and title
    ax.set_ylabel('Credit Balance')
    ax.set_xlabel('Participant')
    ax.set_title('Credit Distribution Among Participants')
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -10),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig

def create_demand_heatmap(person_item_dict, wants_list):
    """
    Create a heatmap showing demand for items.
    
    Parameters:
    -----------
    person_item_dict : dict
        Dictionary mapping person names to their offered item
    wants_list : list
        List of tuples (A,B) meaning person A wants the item person B is offering
        
    Returns:
    --------
    fig : matplotlib figure
        The generated heatmap
    """
    # Create item to person mapping
    item_to_person = {item: person for person, item in person_item_dict.items()}
    
    # Create a list of all items
    all_items = list(set(person_item_dict.values()))
    
    # Count how many people want each item
    item_demand = {item: 0 for item in all_items}
    
    for receiver, giver in wants_list:
        if giver in person_item_dict:
            offered_item = person_item_dict[giver]
            item_demand[offered_item] += 1
    
    # Create a df with items and their demand counts
    df = pd.DataFrame([
        {"Item": item, "Demand Count": count}
        for item, count in item_demand.items()
    ])
    
    # Sort by demand count
    df = df.sort_values('Demand Count', ascending=False)
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))
    
    # Plot as horizontal bars instead of a true heatmap
    ax = sns.barplot(x='Demand Count', y='Item', data=df, palette='viridis')
    
    # Add value annotations
    for i, v in enumerate(df['Demand Count']):
        ax.text(v + 0.1, i, str(v), va='center')
    
    plt.title('Item Demand Distribution')
    plt.tight_layout()
    
    return plt.gcf()

def create_exchange_network(exchanges, credit_balances, person_item_dict, wants_list):
    """
    Create a more advanced network visualization showing exchange patterns.
    
    Parameters:
    -----------
    exchanges : list
        List of tuples (giver, receiver, item) representing exchanges
    credit_balances : dict
        Dictionary mapping person to their credit balance
    person_item_dict : dict
        Dictionary mapping person names to their offered item
    wants_list : list
        List of tuples (A,B) meaning person A wants the item person B is offering
        
    Returns:
    --------
    fig : matplotlib figure
        The generated network visualization
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes for all people
    for person, item in person_item_dict.items():
        # Calculate node size based on how many people want their item
        demand = sum(1 for receiver, giver in wants_list if giver == person)
        # Calculate node color based on credit balance
        credit = credit_balances.get(person, 0)
        
        G.add_node(person, 
                   item=item, 
                   demand=demand, 
                   credit=credit)
    
    # Add edges for exchanges
    for giver, receiver, item in exchanges:
        G.add_edge(giver, receiver, item=item)
    
    # Create the figure
    plt.figure(figsize=(14, 10))
    
    # Position nodes using Kamada-Kawai layout for better spacing
    pos = nx.kamada_kawai_layout(G)
    
    # Calculate node sizes based on demand (how many people want their item)
    node_sizes = [300 + G.nodes[node].get('demand', 0) * 100 for node in G.nodes()]
    
    # Calculate node colors based on credit balance
    node_colors = []
    for node in G.nodes():
        credit = G.nodes[node].get('credit', 0)
        if abs(credit) < 0.01:
            node_colors.append('lightgray')  # Neutral
        elif credit > 0:
            # Blue for positive credit (owed items)
            intensity = min(1.0, credit / 30)
            node_colors.append((0, 0, 1, intensity))
        else:
            # Red for negative credit (owes items)
            intensity = min(1.0, abs(credit) / 30)
            node_colors.append((1, 0, 0, intensity))
    
    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, 
                          node_size=node_sizes, 
                          node_color=node_colors,
                          edgecolors='black')
    
    # Draw the edges
    nx.draw_networkx_edges(G, pos, 
                          width=2, 
                          edge_color='gray', 
                          arrowstyle='->', 
                          arrowsize=20)
    
    # Create node labels with person name, item and credit info
    node_labels = {}
    for node in G.nodes():
        item = G.nodes[node].get('item', '')
        credit = G.nodes[node].get('credit', 0)
        credit_text = f"\nCredit: {credit:.1f}" if abs(credit) > 0.01 else ""
        node_labels[node] = f"{node}\n{item}{credit_text}"
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9)
    
    # Create edge labels showing items being exchanged
    edge_labels = {}
    for giver, receiver in G.edges():
        edge_labels[(giver, receiver)] = G.edges[(giver, receiver)]['item']
    
    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    # Add title and remove axis
    plt.title("Barter System Exchange Network")
    plt.axis('off')
    plt.tight_layout()
    
    return plt.gcf()

def visualize_all_metrics(exchanges, credit_balances, person_item_dict, wants_list, file_prefix="barter_"):
    """
    Create and save all visualization metrics.
    
    Parameters:
    -----------
    exchanges : list
        List of tuples (giver, receiver, item) representing exchanges
    credit_balances : dict
        Dictionary mapping person to their credit balance
    person_item_dict : dict
        Dictionary mapping person names to their offered item
    wants_list : list
        List of tuples (A,B) meaning person A wants the item person B is offering
    file_prefix : str, optional
        Prefix for saved image files
        
    Returns:
    --------
    None
    """
    # Create and save all visualizations
    
    # 1. Credit distribution
    fig_credit = create_credit_distribution_plot(credit_balances)
    fig_credit.savefig(f"{file_prefix}credit_distribution.png", dpi=300, bbox_inches='tight')
    
    # 2. Demand heatmap
    fig_demand = create_demand_heatmap(person_item_dict, wants_list)
    fig_demand.savefig(f"{file_prefix}demand_heatmap.png", dpi=300, bbox_inches='tight')
    
    # 3. Exchange network
    fig_network = create_exchange_network(exchanges, credit_balances, person_item_dict, wants_list)
    fig_network.savefig(f"{file_prefix}exchange_network.png", dpi=300, bbox_inches='tight')
    
    #plt.close('all')
    
    print(f"All visualizations saved with prefix '{file_prefix}'")
