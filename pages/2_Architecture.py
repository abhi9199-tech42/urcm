import streamlit as st
import graphviz

st.set_page_config(page_title="Architecture", layout="wide")

st.header("How It Works: URCM Architecture")

graph = graphviz.Digraph()
graph.attr(rankdir='LR')
graph.attr('node', shape='box', style='rounded,filled', fillcolor='white', fontname='Helvetica')

# Nodes
graph.node('I', 'Input Text', fillcolor='#e3f2fd')
graph.node('P', 'Phoneme\nMapping', fillcolor='#fff3e0')
graph.node('C', 'Compression\n(Resonance)', fillcolor='#e8f5e9')
graph.node('O', 'Output\n(Stable State)', fillcolor='#f3e5f5')

# Edges
graph.edge('I', 'P', label=' Raw Text')
graph.edge('P', 'C', label=' Frequency\nVectors')
graph.edge('C', 'O', label=' Converged\nμ-State')

st.graphviz_chart(graph)

st.markdown("### Process Flow")
st.info("1. **Input:** Raw text is received.\n2. **Phoneme Mapping:** Text is converted into fundamental sound-frequency components.\n3. **Compression:** Frequencies interfere constructively/destructively to find the stable 'truth' state.\n4. **Output:** The final high-density semantic state is returned.")
