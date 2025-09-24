from graphviz import Digraph

# Create a new directed graph
wbs = Digraph('WBS', comment='Project Work Breakdown Structure')
wbs.attr(rankdir='TB', splines='ortho') # TB = Top to Bottom layout
wbs.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue', fontname='Helvetica')
wbs.attr('edge', color='gray40', arrowhead='normal')

# Define the root node
wbs.node('0', 'Content & Media Document Converter Project', fillcolor='aliceblue', shape='box')

# Level 1: Major Phases
wbs.node('1', '1.0 Project Initiation & Management')
wbs.node('2', '2.0 Research & Literature Review')
wbs.node('3', '3.0 System Requirements & Design')
wbs.node('4', '4.0 Development & Integration', fillcolor='lightyellow')
wbs.node('5', '5.0 Analysis & Report Generation')
wbs.node('6', '6.0 Project Closure')

# Connect root to Level 1
for i in range(1, 7):
    wbs.edge('0', str(i))

# Level 2: Sub-tasks for each phase
# 1.0
wbs.node('1.1', '1.1 Project Planning & Scoping')
wbs.edge('1', '1.1')
# 2.0
wbs.node('2.1', '2.1 Collect & Review Literature')
wbs.node('2.2', '2.2 Consolidate Datasets & Metrics')
wbs.edge('2', '2.1')
wbs.edge('2', '2.2')
# 3.0
wbs.node('3.1', '3.1 Define Requirements')
wbs.node('3.2', '3.2 Design System Architecture')
wbs.edge('3', '3.1')
wbs.edge('3', '3.2')
# 4.0
wbs.node('4.1', '4.1 Setup Dev Environment')
wbs.node('4.2', '4.2 Document Parsing Pipeline')
wbs.node('4.3', '4.3 Media & Content Tools')
wbs.node('4.4', '4.4 Gemini Integration')
wbs.edge('4', '4.1')
wbs.edge('4', '4.2')
wbs.edge('4', '4.3')
wbs.edge('4', '4.4')
# 5.0
wbs.node('5.1', '5.1 Analyze Gaps & Challenges')
wbs.node('5.2', '5.2 Draft Final Report')
wbs.edge('5', '5.1')
wbs.edge('5', '5.2')
# 6.0
wbs.node('6.1', '6.1 System Testing & Docs')
wbs.node('6.2', '6.2 Final Presentation & Demo')
wbs.edge('6', '6.1')
wbs.edge('6', '6.2')

# Render and save the diagram
wbs.render('Project_WBS', format='pdf', view=True, cleanup=True)

print("✅ WBS diagram 'Project_WBS.pdf' generated successfully.")