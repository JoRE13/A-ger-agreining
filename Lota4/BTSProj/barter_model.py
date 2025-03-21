import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import matplotlib.pyplot as plt

def load_data(sheet_id, gid="0"):
    """
    Load data from Google Sheets
    """
    URL = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    df = pd.read_csv(URL)
    
    # Remove rows where the person is not offering anything
    df = df[df.iloc[:, 1].notna()]
    
    # A dict with the name of a person as key and the name of the item they're offering as value
    person_item_dict = {row.iloc[0]: row.iloc[1] for _, row in df.iterrows()}
    
    # The names of everyone offering an item
    names = list(person_item_dict.keys())
    
    # The names of everyone in the class
    names_all = df.columns[2:].tolist()
    
    # List of tuples (A,B) meaning person A wants the item person B is offering
    wants_list = []
    
    for _, row in df.iterrows():
        name_offering = row.iloc[0]
        for name_receiving, cell in zip(names_all, row[2:]):
            if str(cell).strip().lower() == 'x':
                wants_list.append((name_receiving, name_offering))
    
    return person_item_dict, names, names_all, wants_list

def create_barter_model(person_item_dict, wants_list, previous_credits=None):
    """
    Create and solve a Gurobi model for the barter system
    """
    # Create the model
    model = gp.Model("Barter_System")
    
    # Extract sets
    people = list(person_item_dict.keys())
    
    # Prepare wanted items dictionary
    # For each person, create a list of who they want items from
    wanted_items = {}
    for receiver, giver in wants_list:
        if receiver not in wanted_items:
            wanted_items[receiver] = []
        wanted_items[receiver].append(giver)
    
    # Calculate item demand (how many people want each item)
    item_demand = {}
    for _, giver in wants_list:
        if giver not in item_demand:
            item_demand[giver] = 0
        item_demand[giver] += 1
    
    # If previous credits not provided, initialize to 0
    if previous_credits is None:
        previous_credits = {person: 0 for person in people}
    
    # Decision variables
    # x[i,j] = 1 if person i gives their item to person j
    #x = {}
    #for i in people:
    #    for j in people:
    #        if i != j:
    #            x[i,j] = model.addVar(vtype=GRB.#BINARY, name=f"x_{i}_{j}")
    
    x = {}
    for i,j in wants_list:
        x[i,j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    
    # credit[i] = credits (promises) held by person i
    credit = {}
    for i in people:
        credit[i] = model.addVar(lb=-GRB.INFINITY, name=f"credit_{i}")
    
    # used_credit[i] = previous credits used by person i in this round
    used_credit = {}
    for i in people:
         max_use = max(0, previous_credits.get(i, 0))
         used_credit[i] = model.addVar(ub=max_use, name=f"used_credit_{i}")
    
    # Add constraints
    
    # 1. Each person can give their item at most once
    for i in people:
        model.addConstr(gp.quicksum(x[i,j] for j in people if i != j and (i,j) in wants_list) <= 1)
    
    # 2. Each person can receive at most one item
    for j in people:
        model.addConstr(gp.quicksum(x[i,j] for i in people if i != j and (i,j) in wants_list) <= 1)
    
    # 3. Person can only receive items they want
    #for i in people:
    #    for j in people:
    #        if i != j and (j not in wanted_items.get(i, [])):
    #            model.addConstr(x[i,j] == 0)
    
    # 4. Credit balance constraint
    for i in people:
        # Credits = Items given out - Items received
        model.addConstr(
            credit[i] == 
            gp.quicksum(x[i,j] for j in people if i != j and (i,j) in wants_list) -
            gp.quicksum(x[j,i] for j in people if i != j and (j,i) in wants_list) + used_credit[i]
        )
    
    # 5. System balance constraint
    model.addConstr(gp.quicksum(credit[i] for i in people) == 0)
    
    
    # Objective: Maximize exchanges prioritizing high-demand items
    # We slightly penalize credits to minimize their use when direct exchanges are possible
    credit_penalty = 1
    
    # Add auxiliary variables for absolute value of credits
    abs_credit = {}
    for i in people:
        abs_credit[i] = model.addVar(name=f"abs_credit_{i}")
        # Add constraints to define abs_credit[i] = |credit[i]|
        model.addConstr(abs_credit[i] >= credit[i])
        model.addConstr(abs_credit[i] >= -credit[i])
    
    # Base objective: maximize exchanges
    obj = gp.quicksum(x[i,j] for i in people for j in people if i != j and (i,j) in wants_list)
    
    # Add demand-based priority
    for i in people:
        for j in people:
            if i != j and j in item_demand and (i,j) in wants_list and i in item_demand:
                # Add weight based on demand
                obj += x[i,j] * (item_demand[j] / 10 + item_demand[i] / 10)
    
    # Penalize credits using auxiliary variables
    obj -= credit_penalty * gp.quicksum(abs_credit[i] for i in people)
    
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # Set time limit to avoid long computations
    model.setParam('TimeLimit', 30)
    model.setParam('OutputFlag', 0)  # Suppress output
    
    # Solve the model
    model.optimize()
    
    # Check if model is solved
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        # Extract results
        exchanges = []
        for i in people:
            for j in people:
                if i != j and (i,j) in wants_list and x[i,j].X > 0.5:
                    exchanges.append((i, j))
        
        # Extract credit balances
        credit_balances = {i: credit[i].X for i in people}
        used_credit_values = {i: used_credit[i].X for i in people}
        
        return {
            'status': model.status,
            'exchanges': exchanges,
            'credit_balances': credit_balances,
            'used_credit': used_credit_values,
            'objective_value': model.objVal
        }
    else:
        return {
            'status': model.status,
            'exchanges': [],
            'credit_balances': {},
            'used_credit': {},
            'objective_value': 0
        }

def visualize_exchanges(results, person_item_dict):
    """
    Visualize the exchanges as a network graph
    """
    G = nx.DiGraph()
    
    # Add nodes
    for person, item in person_item_dict.items():
        credit = results['credit_balances'].get(person, 0)
        label = f"{person}\nOffers: {item}"
        if abs(credit) > 0.001:  # Only show non-zero credits
            label += f"\nCredit: {credit:.1f}"
        G.add_node(person, label=label)
    
    # Add edges for exchanges
    for giver, receiver in results['exchanges']:
        G.add_edge(giver, receiver, item=person_item_dict[giver])
    
    # Create plot
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=700)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=2, arrowsize=15)
    
    # Draw labels
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels=labels)
    
    # Draw edge labels (items being exchanged)
    edge_labels = nx.get_edge_attributes(G, 'item')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title("Barter System Exchanges")
    plt.axis('off')
    plt.tight_layout()
    
    return plt

def main():
    # Example usage with your Google Sheet ID
    sheet_id = "1lEnQYqa-WjGByLrgw5wWHOgWTQaP1chga_p7-z00Ac4"
    
    # Load data
    person_item_dict, names, names_all, wants_list = load_data(sheet_id)
    
    
    
    # Create and solve model
    results = create_barter_model(person_item_dict, wants_list)
    
    # Print results
    print("Status:", "Optimal" if results['status'] == GRB.OPTIMAL else "Time Limit" if results['status'] == GRB.TIME_LIMIT else "Failed")
    print("\nExchanges:")
    for giver, receiver in results['exchanges']:
        print(f"{giver} gives {person_item_dict[giver]} to {receiver}")
    
    print("\nCredit Balances:")
    for person, credit in results['credit_balances'].items():
        if abs(credit) > 0.001:  # Only show non-zero credits
            print(f"{person}: {credit:.1f}")
    
    print("\nTotal exchanges:", len(results['exchanges']))
    print("Objective value:", results['objective_value'])
    
    # Visualize results
    plt = visualize_exchanges(results, person_item_dict)
    plt.savefig("barter_exchanges.png")
    print("Visualization saved as 'barter_exchanges.png'")

if __name__ == "__main__":
    main()