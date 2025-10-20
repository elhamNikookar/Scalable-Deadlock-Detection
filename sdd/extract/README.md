# Dining Philosopher Problem Database

This project creates a comprehensive database from dining philosopher problem state graph files (`.gst`, `.dot`, and `.gxl` files) and provides tools to analyze the state space and transitions.

## Files Description

### Input Files
- **`*.gst` files**: Individual state graphs showing the configuration of philosophers and forks in each state
- **`*.dot` file**: State transition graph showing how states connect via actions
- **`*.gxl` file**: Labeled transition system showing the complete state space with properties

### Output Files
- **`dining_philosopher.db`**: SQLite database containing all extracted information
- **`dining_philosopher_data.json`**: JSON export of the database for easy analysis

## Database Schema

The database contains the following tables:

### 1. `states`
- `state_id`: Primary key, state identifier (e.g., 's0', 's1')
- `state_name`: State name
- `is_start_state`: Boolean indicating if this is a start state
- `is_final_state`: Boolean indicating if this is a final state
- `has_reach_property`: Boolean indicating if state has reach_2 property
- `description`: Optional description

### 2. `state_configurations`
- `id`: Auto-increment primary key
- `state_id`: Foreign key to states table
- `node_id`: Node identifier within the state
- `node_type`: Type of node ('Philosopher' or 'Fork')
- `node_state`: Current state of the node ('think', 'hungry', 'available', etc.)
- `left_fork_id`: Reference to left fork (for philosophers)
- `right_fork_id`: Reference to right fork (for philosophers)

### 3. `transitions`
- `id`: Auto-increment primary key
- `from_state`: Source state
- `to_state`: Target state
- `action`: Action causing the transition
- `transition_type`: Type of transition ('state_transition' or 'lts_transition')

### 4. `actions`
- `action_name`: Primary key, name of the action
- `description`: Description of what the action does
- `category`: Category of the action ('hunger', 'fork_acquisition', 'fork_release', etc.)

### 5. `nodes`
- `id`: Auto-increment primary key
- `state_id`: Foreign key to states table
- `node_id`: Node identifier
- `node_type`: Type of node
- `attributes`: JSON string of node attributes

### 6. `edges`
- `id`: Auto-increment primary key
- `state_id`: Foreign key to states table
- `from_node`: Source node
- `to_node`: Target node
- `edge_label`: Label on the edge

## Usage

### 1. Create the Database

```bash
python dining_philosopher_database.py
```

This will:
- Process all `.gst`, `.dot`, and `.gxl` files in the current directory
- Create a SQLite database (`dining_philosopher.db`)
- Export data to JSON (`dining_philosopher_data.json`)
- Print a summary of the extracted information

### 2. Query the Database

```bash
python query_database.py
```

This demonstrates various queries including:
- All states and their properties
- All transitions between states
- Detailed information about specific states
- Philosophers in different states (thinking, hungry)
- Fork acquisition and release actions
- Paths to reach specific states
- Action categories and frequencies

### 3. Custom Queries

You can write custom SQL queries to analyze the data:

```python
from dining_philosopher_database import DiningPhilosopherDatabase

db = DiningPhilosopherDatabase()

# Find all philosophers currently holding forks
db.cursor.execute('''
    SELECT state_id, node_id, left_fork_id, right_fork_id
    FROM state_configurations 
    WHERE node_type = 'Philosopher' 
    AND (left_fork_id IS NOT NULL OR right_fork_id IS NOT NULL)
''')

# Find all deadlock states (where no philosopher can eat)
db.cursor.execute('''
    SELECT DISTINCT s.state_id
    FROM states s
    JOIN state_configurations sc ON s.state_id = sc.state_id
    WHERE sc.node_type = 'Philosopher' 
    AND sc.node_state = 'hungry'
    AND NOT EXISTS (
        SELECT 1 FROM state_configurations sc2 
        WHERE sc2.state_id = s.state_id 
        AND sc2.node_type = 'Philosopher' 
        AND sc2.node_state = 'think'
    )
''')
```

## Understanding the Dining Philosopher Problem

The dining philosopher problem is a classic synchronization problem that illustrates challenges in resource allocation and deadlock prevention. In this implementation:

- **Philosophers** can be in states: `think`, `hungry`
- **Forks** can be in states: `available`, `held`
- **Actions** include:
  - `go-hungry`: Philosopher becomes hungry
  - `get-left`/`get-right`: Philosopher picks up a fork
  - `release-left`/`release-right`: Philosopher releases a fork
  - `think`: Philosopher is thinking
  - `hungry`: Philosopher is hungry

### Key Properties

- **Start State (s0)**: Initial configuration where all philosophers are thinking
- **Final State (s8)**: State with `reach_2` property - represents a specific reachability goal
- **Transitions**: Show how the system evolves through different actions

## Analysis Capabilities

The database enables analysis of:

1. **State Space Exploration**: All possible configurations of philosophers and forks
2. **Transition Analysis**: How the system evolves through different actions
3. **Deadlock Detection**: Comprehensive deadlock analysis with classification
4. **Reachability Analysis**: Paths to reach specific states or properties
5. **Resource Allocation**: How forks are distributed among philosophers
6. **Property Verification**: Checking if specific properties hold in the system

### 🔴 **Deadlock Detection Features:**

- **Automatic Deadlock Detection**: Analyzes all states for deadlock conditions
- **Deadlock Classification**: 
  - Classic Dining Philosopher Deadlock
  - Resource Starvation
  - Circular Wait
  - Livelock Detection
- **Deadlock Labeling**: All states are labeled with deadlock information
- **Prevention Strategies**: 8 different deadlock prevention rules
- **ML-Ready Data**: Deadlock labels for machine learning training

## Machine Learning & Deep Learning Capabilities

The database is highly suitable for data mining and deep learning applications:

### 🎯 **ML Tasks Supported:**

1. **Classification Tasks:**
   - Predict if a state has the `reach_2` property
   - Classify states as deadlock-prone or safe
   - Predict state types (start, final, intermediate)

2. **Regression Tasks:**
   - Predict fork utilization rates
   - Estimate resource allocation efficiency
   - Predict system performance metrics

3. **Sequence Prediction:**
   - Predict next states in transition sequences
   - Forecast action sequences
   - Time-series analysis of state evolution

4. **Graph Neural Networks:**
   - State transition graph analysis
   - Node classification (state properties)
   - Edge prediction (transition likelihood)

### 📊 **Data Preparation for ML:**

```bash
# Install ML dependencies
pip install -r requirements_ml.txt

# Update database with deadlock detection
python deadlock_demo.py

# Prepare data for machine learning
python ml_data_preparation.py

# Run deep learning examples
python deep_learning_example.py
```

### 🔍 **ML Analysis Features:**

- **Feature Engineering**: 13 engineered features per state (including deadlock features)
- **Graph Analysis**: NetworkX integration for graph-based ML
- **Sequence Data**: LSTM-ready sequence data
- **Classification Data**: Binary and multi-class classification
- **Regression Data**: Continuous value prediction
- **Deadlock Detection**: Specialized deadlock classification model

### 📈 **Deep Learning Models:**

1. **Classification Model**: Neural network to predict reachability properties
2. **Deadlock Classification Model**: Neural network to predict deadlock states
3. **Regression Model**: Neural network to predict fork utilization
4. **Sequence Model**: LSTM to predict next states in sequences
5. **Feature Importance Analysis**: Permutation importance for interpretability

### 🎯 **Use Cases for Data Mining:**

- **Deadlock Detection**: ML models can learn patterns that lead to deadlocks
- **Resource Optimization**: Predict optimal fork allocation strategies
- **System Verification**: Automated property verification using ML
- **Anomaly Detection**: Identify unusual state configurations
- **Performance Prediction**: Forecast system behavior under different conditions

## Requirements

### Basic Requirements (Database Only)
- Python 3.6+
- Standard libraries: `sqlite3`, `xml.etree.ElementTree`, `os`, `re`, `json`, `typing`

### Machine Learning Requirements
- Install with: `pip install -r requirements_ml.txt`
- Includes: `numpy`, `pandas`, `scikit-learn`, `tensorflow`, `matplotlib`, `networkx` 