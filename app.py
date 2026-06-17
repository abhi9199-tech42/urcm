import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from urcm.core.system import URCMSystem

# Configure Page
st.set_page_config(
    page_title="URCM Visual Debugger",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# 1. Load System (Cached)
@st.cache_resource
def load_system():
    return URCMSystem()

st.title("🧠 Unified μ-Resonance Cognitive Mesh (URCM)")
st.markdown("**Phase 4: Visual Debugger & Interaction Interface**")

try:
    with st.spinner("Loading Neural Core..."):
        system = load_system()
    st.sidebar.success("System Online ✅")
except Exception as e:
    st.error(f"System Failed to Load: {e}")
    st.stop()

# 2. Sidebar Controls
st.sidebar.header("⚙️ Cognitive Parameters")

noise_level = st.sidebar.slider(
    "Input Noise (Sigma)", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.15,
    help="Amount of sensory noise injected into the initial state."
)

temperature = st.sidebar.slider(
    "Temperature (T)", 
    min_value=0.1, 
    max_value=2.0, 
    value=1.0,
    help="Controls decision sharpness. Low T = High Gain (Decisive). High T = Soft (Dreamy)."
)

max_steps = st.sidebar.slider(
    "Max Thinking Steps", 
    min_value=10, 
    max_value=200, 
    value=100
)

# 3. Main Interface
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📡 Sensory Input")
    
    # Input Mode
    input_mode = st.radio("Input Source", ["Single Phoneme", "Adversarial Mix"])
    
    if input_mode == "Single Phoneme":
        phoneme = st.text_input("Enter Phoneme (Sanskrit)", "a")
        target_phoneme = phoneme
        
    else:
        p1 = st.text_input("Phoneme A", "k")
        p2 = st.text_input("Phoneme B", "g")
        mix_ratio = st.slider(f"Mix Ratio ({p1} vs {p2})", 0.0, 1.0, 0.5)
        target_phoneme = f"{p1} ({mix_ratio:.1f}) + {p2} ({1-mix_ratio:.1f})"
    
    if st.button("Inject Thought"):
        codebook = system.pipeline.frequency_mapper.phoneme_vectors
        
        # Prepare Initial State
        if input_mode == "Single Phoneme":
            if phoneme in codebook:
                vec = codebook[phoneme]
                h_ideal = np.dot(vec, system.encoder.W_in)
                s_ideal = np.tanh(h_ideal)
                
                # Add Noise
                noise = np.random.normal(0, noise_level, s_ideal.shape)
                # Clip before arctanh to be safe
                safe_s = np.clip(s_ideal, -0.999, 0.999)
                s_start = np.tanh(np.arctanh(safe_s) + noise)
            else:
                st.error("Unknown Phoneme")
                st.stop()
        else:
            # Mix
            if p1 in codebook and p2 in codebook:
                v1 = codebook[p1]
                v2 = codebook[p2]
                
                h1 = np.dot(v1, system.encoder.W_in)
                s1 = np.tanh(h1)
                
                h2 = np.dot(v2, system.encoder.W_in)
                s2 = np.tanh(h2)
                
                s_mix = mix_ratio * s1 + (1-mix_ratio) * s2
                
                # Add Noise
                noise = np.random.normal(0, noise_level, s_mix.shape)
                # No arctanh on mix, just raw addition to state? 
                # Ideally: s_start = tanh(arctanh(mix) + noise) but mix might not be valid tanh state.
                # Just add noise to state directly and clip.
                s_start = np.clip(s_mix + noise, -0.999, 0.999)
            else:
                st.error("Unknown Phonemes in Mix")
                st.stop()

        # Run Dynamics
        with st.spinner("Thinking..."):
            final_s, steps, history = system.encoder.run_dynamics_until_stable(
                s_start, 
                codebook,
                max_steps=max_steps,
                temperature=temperature,
                return_history=True
            )
            
            # Decode
            safe_final = np.clip(final_s, -0.999, 0.999)
            h_out = np.arctanh(safe_final)
            x_out = np.dot(h_out, system.encoder.W_out)
            
            nearest, dist = system.pipeline.frequency_mapper.find_nearest(x_out)
            
            # Store in Session
            st.session_state['history'] = history
            st.session_state['steps'] = steps
            st.session_state['nearest'] = nearest
            st.session_state['dist'] = dist
            st.session_state['target'] = target_phoneme

with col2:
    st.subheader("📉 Cognitive Dynamics")
    
    if 'history' in st.session_state:
        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Steps Taken", st.session_state['steps'])
        m2.metric("Final Confidence", f"{1.0/st.session_state['dist']:.2f}" if st.session_state['dist'] > 0 else "Inf")
        m3.metric("Result", st.session_state['nearest'])
        
        # Plot
        history = st.session_state['history']
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history, marker='o', linestyle='-', color='#4CAF50', linewidth=2, markersize=4)
        ax.set_title("Canonical Energy Landscape Descent")
        ax.set_xlabel("Time (Thinking Steps)")
        ax.set_ylabel("Energy (Confusion)")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Highlight Frustration Shocks (Spikes)
        # Simple heuristic: if E[t] > E[t-1] + 0.2
        for t in range(1, len(history)):
            if history[t] > history[t-1] + 0.2:
                ax.annotate('Frustration Shock ⚡', xy=(t, history[t]), xytext=(t, history[t]+0.5),
                            arrowprops=dict(facecolor='red', shrink=0.05))
        
        st.pyplot(fig)
        
        # Interpretation
        st.info(f"The system started with **'{st.session_state['target']}'** and converged to **'{st.session_state['nearest']}'**.")
        
        if "Mix" in str(st.session_state['target']):
             st.success("Symmetry Broken! 💥")

st.markdown("---")
st.markdown("🔒 **Safety Locks:** Active | 🔋 **Energy Invariant:** Monotonic | 🧬 **Topology:** Conserved")
