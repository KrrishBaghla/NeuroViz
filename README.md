# NeuroViz - Micrograd Neural Network Visualizer

A beautiful Streamlit web app for visualizing neural network training using the **Micrograd** framework.

## Features

- 🧠 **Interactive Neural Network Training** - Train on the XOR problem in real-time
- 📊 **Live Loss Visualization** - Watch the loss decrease as the network learns
- ⚙️ **Customizable Hyperparameters** - Adjust epochs, learning rate, and hidden layers
- 🎯 **Real-time Predictions** - See model predictions on test inputs
- 🚀 **Automatic Differentiation** - Powered by custom micrograd engine

## Project Structure

```
micrograd/
├── app/
│   └── app.py              # Streamlit web app
├── engine/
│   ├── __init__.py
│   ├── value.py            # Autograd Value class
│   ├── nn.py               # Neural network layers
│   └── __pycache__/
├── test/
│   └── test_file.py
├── main.py
├── requirements.txt
└── README.md
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/micrograd.git
   cd micrograd
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:

```bash
streamlit run app/app.py
```

Then open your browser at `http://localhost:8501`

## How It Works

1. **Forward Pass**: Data flows through the network (neurons compute weighted sums + bias, apply ReLU activation)
2. **Loss Computation**: Mean Squared Error between predictions and targets
3. **Backward Pass**: Gradients flow back through the network via chain rule
4. **Parameter Update**: Weights are updated using gradient descent

## Customization

In the sidebar, you can adjust:
- **Training Epochs** (10-150, default: 75)
- **Learning Rate** (0.0001-0.5)
- **Hidden Layer Architecture** (e.g., "4, 4" for two layers with 4 neurons each)

## Technologies

- **Streamlit** - Interactive web app framework
- **Pandas** - Data manipulation
- **Matplotlib** - Visualization
- **Micrograd** - Custom automatic differentiation engine

## License

MIT License

## Author

Created as an educational project for understanding neural networks and backpropagation.
