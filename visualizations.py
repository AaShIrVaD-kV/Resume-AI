import plotly.graph_objects as go
import networkx as nx
import plotly.express as px

def create_ats_gauge(score):
    """Creates a gauge chart for ATS Score."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "ATS Acceptance Probability"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#00d4ff"},
            'steps': [
                {'range': [0, 50], 'color': "#ff4b4b"},
                {'range': [50, 75], 'color': "#ffa600"},
                {'range': [75, 100], 'color': "#21c354"}
            ],
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    return fig

def create_skill_network(skills):
    """Creates a network graph of skills."""
    G = nx.Graph()
    
    # Central Node
    G.add_node("Candidate", size=20, color='#00d4ff')
    
    # Skill Nodes
    # Limit to top 8 skills for cleaner visualization as requested
    display_skills = list(skills)[:8] 
    for skill in display_skills:
        G.add_node(skill, size=20, color='#005bea')
        G.add_edge("Candidate", skill)
        
    # Create layout
    # k parameter controls distance between nodes (higher = further apart)
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            showscale=False,
            color=['#00d4ff' if n == "Candidate" else '#005bea' for n in G.nodes()],
            size=20,
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0,l=0,r=0,t=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    return fig
