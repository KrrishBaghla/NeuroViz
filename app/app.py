import sys
import os
import time
import streamlit as st
import pandas as pd

# Add the parent directory to sys.path to allow importing from the engine folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import actual engine modules
try:
    from engine.nn import MLP
    from engine.value import Value
except ImportError:
    st.error("Engine modules not found. Ensure 'engine/nn.py' and 'engine/value.py' exist.")
    st.stop()

def mse(pred, target):
    """Mean Squared Error: (prediction - target)^2"""
    return (pred - target) * (pred - target)

# --- STREAMLIT UI ---
st.set_page_config(
    page_title="NeuroViz | Micrograd Explorer",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for a premium look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border-left: 5px solid #6366f1; }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("🛠️ Hyperparameters")
    epochs = st.number_input("Training Epochs", 10, 150, 75)
    
    # Updated Slider for more freedom
    lr = st.slider("Learning Rate", min_value=0.0001, max_value=0.5, value=0.1, step=0.0001, format="%.4f")
    
    st.divider()
    st.markdown("### Model Architecture")
    hidden_layers = st.text_input("Hidden Layers (e.g., 4, 4)", "4, 4")
    try:
        layers = [int(x.strip()) for x in hidden_layers.split(",")]
    except:
        st.warning("Invalid format. Defaulting to [4, 4]")
        layers = [4, 4]

# --- Main UI Layout ---
st.title("🧠 NeuroViz")
st.subheader("Visualizing Backpropagation")

# Updated notice as requested
st.info("We're using ReLU here. If the graph stays flat, Try hitting 'Start' again!")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### XOR Truth Table")
    st.table(pd.DataFrame({
        "Input A": [0, 0, 1, 1],
        "Input B": [0, 1, 0, 1],
        "Output": [0, 1, 1, 0]
    }))

with col2:
    st.markdown("### Target Logic")
    st.code("""
# XOR Dataset
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [0, 1, 1, 0]
    """, language="python")

st.divider()
train_col, metrics_col = st.columns([2, 1])

with train_col:
    if st.button("🚀 Start Training Session", use_container_width=True):
        # Initialize Data
        X = [[Value(0), Value(0)], [Value(0), Value(1)], [Value(1), Value(0)], [Value(1), Value(1)]]
        Y = [0, 1, 1, 0]
        
        # Initialize Model (nin=2, nouts=layers + [1])
        model = MLP(2, layers + [1])
        
        # Setup UI Placeholders
        progress_bar = st.progress(0)
        status_text = st.empty()
        chart_placeholder = st.empty()
        
        losses = []
        
        for epoch in range(epochs):
            # 1. Forward Pass
            preds = [model(x) for x in X]
            loss = sum((mse(p, Value(y)) for p, y in zip(preds, Y)), Value(0))
            
            # 2. Backward Pass
            model.zero_grad()
            loss.backward()
            
            # 3. Update (SGD)
            for p in model.parameters():
                p.data -= lr * p.grad
            
            losses.append(loss.data)
            
            # Update Visuals
            if epoch % 10 == 0 or epoch == epochs - 1:
                progress_bar.progress((epoch + 1) / epochs)
                status_text.markdown(f"**Epoch {epoch+1}** — Loss: `{loss.data:.4f}`")
                chart_placeholder.line_chart(pd.DataFrame(losses, columns=["Loss"]), height=300)
            
            time.sleep(0.005)

        st.success("✅ Done! Check out the results below.")
        
        # Final Results Table
        st.subheader("🎯 How did we do?")
        results = []
        for x, y in zip(X, Y):
            pred = model(x)
            results.append({
                "Input": f"[{int(x[0].data)}, {int(x[1].data)}]",
                "Target": y,
                "Prediction": f"{pred.data:.4f}",
                "Success": "✅ Correct" if (pred.data > 0.5) == bool(y) else "❌ Missed"
            })
        st.dataframe(pd.DataFrame(results), use_container_width=True)

with metrics_col:
    st.markdown("### Live Stats")
    param_count = len(model.parameters()) if 'model' in locals() else 0
    st.metric("Total Parameters", param_count)
    st.metric("Architecture", f"2 → {' → '.join(map(str, layers))} → 1")
    
    if 'losses' in locals() and len(losses) > 1:
        improvement = losses[0] - losses[-1]
        st.metric("Overall Improvement", f"{improvement:.4f}")