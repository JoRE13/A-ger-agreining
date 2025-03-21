"""
Barter System Model Implementation using PuLP
"""

import pandas as pd
import pulp as pl
import networkx as nx
import matplotlib.pyplot as plt

def load_data(sheet_id, gid="0"):
    """
    Load data from Google Sheets.
    
    Parameters:
    -----------
    sheet_id : str
        Google Sheets ID
    gid : str
        Sheet ID within the Google Sheets document
        
    Returns:
    --------
    person_item_dict : dict
        Dictionary mapping person names to their offered item
    names : list
        List of all persons offering items
    names_all : list
        List of all persons in the system
    wants_list : list
        List of tuples (A,B) meaning person A wants the item person B is offering
    """
    URL = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    df = pd.read_csv(URL)
    
    # Remove rows where the person is not offering anything
    df = df[df.iloc[:, 1].notna()]
    
    # Dictionary mapping person to their offered item
    person_item_dict = {row.iloc[0]: row.iloc[1] for _, row in df.iterrows()}
    
    # All persons offering items
    names = list(person_item_dict.keys())
    
    # All persons in the system
    names_all = df.columns[2:].tolist()
    
    # List of tuples (A,B) meaning person A wants the item person B is offering
    wants_list = []
    for _, row in df.iterrows():
        name_offering = row.iloc[0]
        for name_receiving, cell in zip(names_all, row[2:]):
            if str(cell).strip().lower() == 'x':
                wants_list.append((name_receiving, name_offering))
    
    return person_item_dict, names, names_all, wants_list

def create_demand_scores(person_item_dict, wants_list):
    """
    Calculate demand scores for each item based on how many people want it.
    
    Parameters:
    -----------
    person_item_dict : dict
        Dictionary mapping person names to their offered item
    wants_list : list
        List of tuples (A,B) meaning person A wants the item person B is offering
        
    Returns:
    --------
    demand_scores : dict
        Dictionary mapping items to their demand scores
    """
    # Create item to person mapping
    item_to_person = {item: person for person, item in person_item_dict.items()}
    
    # Count how many people want each person's item
    person_demand_count = {}
    for receiver, giver in wants_list:
        if giver in person_demand_count:
            person_demand_count[giver] += 1
        else:
            person_demand_count[giver] = 1
    
    # Calculate demand scores for each item
    demand_scores = {}
    for person, item in person_item_dict.items():
        if person in person_demand_count:
            demand_scores[item] = person_demand_count[person] * 10
        else:
            demand_scores[item] = 1  # Minimum score for items nobody wants
    
    return demand_scores

def create_barter_model(person_item_dict, wants_list, previous_credits=None, alpha=0.1):
    """
    Create and solve the barter system optimization model.
    
    Parameters:
    -----------
    person_item_dict : dict
        Dictionary mapping person names to their offered item
    wants_list : list
        List of tuples (A,B) meaning person A wants the item person B is offering
    previous_credits : dict, optional
        Dictionary mapping person to their previous credit balance
    alpha : float, optional
        Penalty factor for credits
        
    Returns:
    --------
    model : PuLP model
        The solved optimization model
    exchanges : list
        List of tuples (giver, receiver) representing exchanges to make
    credit_balances : dict
        Dictionary mapping person to their new credit balance
    """
    # Create the model
    model = pl.LpProblem("Barter_System", pl.LpMaximize)
    
    # Extract people and items
    people = list(person_item_dict.keys())
    items = list(person_item_dict.values())
    
    # Calculate demand scores
    demand_scores = create_demand_scores(person_item_dict, wants_list)
    
    # Initialize previous credits if not provided
    if previous_credits is None:
        previous_credits = {person: 0 for person in people}
    
    # Create wants dictionary
    # Format: {receiver: [givers]} - meaning receiver wants items from givers
    wants_dict = {}
    for receiver, giver in wants_list:
        if receiver not in wants_dict:
            wants_dict[receiver] = []
        wants_dict[receiver].append(giver)
    
    # Decision variables
    # X[i,j] = 1 if person i gives their item to person j
    X = {}
    for giver in people:
        for receiver in people:
            X[giver, receiver] = pl.LpVariable(f"X_{giver}_{receiver}", cat=pl.LpBinary)
    
    # Credit variables
    C = {}  # Credit balance
    UC = {}  # Used credit from previous balance
    abs_C = {}  # Absolute value of credit (for objective function)
    
    for person in people:
        C[person] = pl.LpVariable(f"C_{person}", lowBound=None)
        UC[person] = pl.LpVariable(f"UC_{person}", lowBound=0, upBound=previous_credits[person])
        abs_C[person] = pl.LpVariable(f"abs_C_{person}", lowBound=0)  # For absolute value
    
    # Constraints
    
    # 1. Each person can give their item to at most one person
    for giver in people:
        model += pl.lpSum(X[giver, receiver] for receiver in people) <= 1, f"Give_once_{giver}"
    
    # 2. Each person can receive at most one item
    for receiver in people:
        model += pl.lpSum(X[giver, receiver] for giver in people) <= 1, f"Receive_once_{receiver}"
    
    # 3. A person can only receive an item they want
    for giver in people:
        for receiver in people:
            if receiver not in wants_dict or giver not in wants_dict[receiver]:
                model += X[giver, receiver] == 0, f"Want_{giver}_{receiver}"
    
    # 4. Credit balance constraints
    for person in people:
        # Credit = (Value given out) - (Value received) + (Used previous credit)
        item_value = demand_scores[person_item_dict[person]]
        
        value_given = pl.lpSum(X[person, receiver] * item_value for receiver in people)
        value_received = pl.lpSum(X[giver, person] * demand_scores[person_item_dict[giver]] 
                                for giver in people)
        
        model += C[person] == value_given - value_received + UC[person], f"Credit_balance_{person}"
    
    # 5. System balance constraint
    model += pl.lpSum(C[person] for person in people) == 0, "System_balance"
    
    # 6. Absolute value constraints for credit
    for person in people:
        model += abs_C[person] >= C[person], f"Abs_credit_pos_{person}"
        model += abs_C[person] >= -C[person], f"Abs_credit_neg_{person}"
    
    # Objective: Maximize exchanges while minimizing credits
    # We weight each exchange by its demand score
    exchange_value = pl.lpSum(X[giver, receiver] * demand_scores[person_item_dict[giver]] 
                           for giver in people for receiver in people)
    
    # Use absolute value variables in the objective
    credit_penalty = alpha * pl.lpSum(abs_C[person] for person in people)
    
    model += exchange_value - credit_penalty
    
    # Solve the model
    model.solve(pl.PULP_CBC_CMD(msg=False))
    
    # Extract results
    exchanges = []
    if pl.LpStatus[model.status] == 'Optimal':
        for giver in people:
            for receiver in people:
                if pl.value(X[giver, receiver]) > 0.5:  # Using 0.5 to handle floating point imprecision
                    exchanges.append((giver, receiver, person_item_dict[giver]))
    
    # Extract credit balances
    credit_balances = {person: pl.value(C[person]) for person in people}
    used_credits = {person: pl.value(UC[person]) for person in people}
    
    return model, exchanges, credit_balances, used_credits

def visualize_exchanges(exchanges, credit_balances, person_item_dict):
    """
    Visualize the barter network.
    
    Parameters:
    -----------
    exchanges : list
        List of tuples (giver, receiver, item) representing exchanges
    credit_balances : dict
        Dictionary mapping person to their credit balance
    person_item_dict : dict
        Dictionary mapping person names to their offered item
        
    Returns:
    --------
    G : NetworkX graph
        The generated network graph
    """
    G = nx.DiGraph()
    
    # Add nodes for all people
    for person, item in person_item_dict.items():
        credit = credit_balances.get(person, 0)
        credit_text = f"\nCredit: {credit:.1f}" if abs(credit) > 0.01 else ""
        G.add_node(person, label=f"{person}\n{item}{credit_text}")
    
    # Add edges for exchanges
    for giver, receiver, item in exchanges:
        G.add_edge(giver, receiver, item=item)
    
    # Draw the graph
    plt.figure(figsize=(12, 10))
    
    # Position nodes using spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=5000, node_color='lightblue', alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=2, edge_color='gray', arrowsize=20)
    
    # Draw labels
    node_labels = {node: data['label'] for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
    
    # Draw edge labels
    edge_labels = {(giver, receiver): item for giver, receiver, item in exchanges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title("Barter System Exchanges")
    plt.axis('off')
    plt.tight_layout()
    
    return G

def analyze_results(model, exchanges, credit_balances, person_item_dict, wants_list):
    """
    Analyze the optimization results.
    
    Parameters:
    -----------
    model : PuLP model
        The solved optimization model
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
    analysis : dict
        Dictionary containing analysis metrics
    """
    analysis = {}
    
    # Number of completed exchanges
    analysis['num_exchanges'] = len(exchanges)
    
    # Percentage of people participating in exchanges
    participants = set()
    for giver, receiver, _ in exchanges:
        participants.add(giver)
        participants.add(receiver)
    
    analysis['participation_rate'] = len(participants) / len(person_item_dict) * 100
    
    # Number of people with credits (promises for future items)
    credit_holders = sum(1 for person, credit in credit_balances.items() if credit > 0.01)
    analysis['credit_holders'] = credit_holders
    
    # Total credits in the system (should be close to zero)
    analysis['total_credits'] = sum(credit_balances.values())
    
    # Percentage of wants fulfilled
    wants_dict = {}
    for receiver, giver in wants_list:
        if receiver not in wants_dict:
            wants_dict[receiver] = []
        wants_dict[receiver].append(giver)
    
    received_count = 0
    for giver, receiver, _ in exchanges:
        if receiver in wants_dict and giver in wants_dict[receiver]:
            received_count += 1
    
    total_wants = sum(len(givers) for givers in wants_dict.values())
    analysis['wants_fulfilled'] = received_count / total_wants * 100 if total_wants > 0 else 0
    
    # Objective value
    analysis['objective_value'] = pl.value(model.objective)
    
    return analysis