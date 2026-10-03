# 19867-Project

# Epidemic-Aware Travel Optimization

This project studies how travel restrictions can reduce epidemic spread in a network of geographic regions. Each region is modeled using an epidemiological model such as SIR, while travel between regions introduces additional cross-region infections.

The project supports fitting epidemic parameters from historical case data, simulating disease spread over a travel network, and optimizing different types of travel interventions with Gurobi.

## Acknowledge

1. The data of population flow come from https://github.com/GeoDS/COVID19USFlows and https://github.com/GeoDS/COVID19USFlows-DailyFlows

2. The data of infected and death cases come from
   https://github.com/CSSEGISandData/COVID-19

## Optimization Strategies

### Binary Edge Reduction

`BinaryReduceEdgeOptimizer`

Each directed travel edge is either:

- kept, or
- completely removed.

The optimizer chooses which edges to remove while minimizing epidemic spread subject to a travel-reduction budget.

Conceptually,

\[
\sum*{(i,j)} F*{ij}(1-y\_{ij}) \le M,
\]

where \(y\_{ij}\in\{0,1\}\) indicates whether edge \((i,j)\) is kept.

---

### Continuous Edge Reduction

`ContinuousReduceEdgeOptimizer`

Instead of completely removing an edge, the optimizer chooses a continuous travel retention ratio:

\[
0 \le y\_{ij} \le 1.
\]

The optimized flow becomes

\[
F'_{ij}=y_{ij}F\_{ij}.
\]

This provides a more flexible benchmark than binary edge removal.

---

### Binary Node Reduction

`BinaryReduceNodeOptimizer`

The optimizer selects geographic nodes to isolate.

If node \(i\) is disabled, all incoming and outgoing travel involving that node is removed:

\[
F*{ij}=F*{ji}=0.
\]

The remaining nodes continue to interact normally.

---

### Travel Bubble Clustering

`BinaryClusterOptimizer`

Each node is assigned to one travel bubble. Travel is preserved within each bubble and removed between different bubbles.

If nodes \(i\) and \(j\) belong to the same cluster,

\[
F'_{ij}=F_{ij}.
\]

Otherwise,

\[
F'\_{ij}=0.
\]

The optimizer supports up to `K` clusters and can impose minimum population requirements on each active cluster.

The cluster optimizer also provides visualizations for:

- cluster membership,
- travel-network structure,
- cluster population.

## Baselines

Two baseline scenarios are evaluated before optimization.

### No Intervention

The original travel matrices are used without modification.

### Uniform Travel Reduction

Every travel edge is reduced by the same fraction:

F'\_{ij} = (1-r)F\_{ij},

where `r` is specified by `--travel_reduce_ratio`.

This provides a useful comparison against optimized interventions using approximately the same travel-reduction scale.

## Project Structure

A typical project layout is:

```text
project/
├── data/
│   ├── case_data/
│   └── flow_data/
│       ├── state/
│       └── county/
│
├── src/
│   ├── main.py
│   ├── node.py
│   ├── model.py
│   ├── optimizer.py
│   ├── data_loader.py
│   └── constants.py
│
└── results/
```

## Requirements

The project requires Python and the following major packages:

```text
numpy
pandas
scipy
matplotlib
networkx
gurobipy
```

Install the Python dependencies with, for example:

```bash
pip install -r requirements.txt
```

Note: A valid Gurobi license is required to run the optimization models.

## Running the Experiment

The main experiment is launched through `main.py`.

With all default parameters:

```bash
python main.py
```

The default configuration is approximately equivalent to:

```bash
python main.py \
    --level state \
    --problem_size small \
    --optimizer cluster \
    --output_dir ../results \
    --data_dir ../data \
    --travel_reduce_ratio 0.2 \
    --clusters 3 \
    --start_date 2020-05-01 \
    --fit_days 90 \
    --opt_days 30 \
    --method euler \
    --model SIR
```

## Command-Line Arguments

### Geographic Level

```bash
--level {state,county}
```

Default:

```text
state
```

Selects whether the experiment operates on state- or county-level nodes.

---

### Problem Size

```bash
--problem_size {small,full}
```

Default:

```text
small
```

Controls whether a reduced network or the full network is used.

---

### Optimizer

```bash
--optimizer {edge,edge_continuous,node,cluster}
```

Default:

```text
cluster
```

Available choices:

| Argument          | Optimizer                 |
| ----------------- | ------------------------- |
| `edge`            | Binary edge removal       |
| `edge_continuous` | Continuous edge reduction |
| `node`            | Binary node isolation     |
| `cluster`         | Travel bubble clustering  |

Example:

```bash
python main.py --optimizer edge
```

---

### Output Directory

```bash
--output_dir PATH
```

Default:

```text
../results
```

Example:

```bash
python main.py --output_dir ../results/experiment1
```

---

### Data Directory

```bash
--data_dir PATH
```

Default:

```text
../data
```

The program expects epidemic case data under:

```text
<data_dir>/case_data
```

and travel flow data under the corresponding flow-data directory.

---

### Travel Reduction Ratio

```bash
--travel_reduce_ratio FLOAT
```

Default:

```text
0.2
```

For the edge and node optimizers, this represents the maximum fraction of total travel that may be removed.

For the uniform-reduction baseline, every edge is multiplied by

\[
1-\text{travel_reduce_ratio}.
\]

For example,

```bash
--travel_reduce_ratio 0.2
```

uniformly reduces all travel by 20%.

---

### Number of Clusters

```bash
--clusters K
```

Default:

```text
3
```

Specifies the maximum number of travel bubbles available to the cluster optimizer.

Example:

```bash
python main.py \
    --optimizer cluster \
    --clusters 4
```

---

### Start Date

```bash
--start_date YYYY-MM-DD
```

Default:

```text
2020-05-01
```

This determines the beginning of the network fitting period.

---

### Fitting Horizon

```bash
--fit_days N
```

Default:

```text
90
```

Number of days used to fit the epidemic and network parameters.

---

### Optimization Horizon

```bash
--opt_days N
```

Default:

```text
30
```

Number of days following the fitting period over which interventions are optimized and evaluated.

The overall timeline is therefore:

```text
start_date
    |
    |------ fit_days ------|
                           |
                           |------ opt_days ------|
```

---

### Numerical Integration Method

```bash
--method {euler,rk4}
```

Default:

```text
euler
```

Available methods:

- Euler
- fourth-order Runge-Kutta (RK4)

Example:

```bash
python main.py --method rk4
```

---

### Epidemic Model

```bash
--model {SIR,SIRD}
```

Default:

```text
SIR
```

Example:

```bash
python main.py --model SIRD
```

## Example Experiments

### Binary Edge Optimization

```bash
python main.py \
    --optimizer edge \
    --travel_reduce_ratio 0.2
```

### Continuous Edge Optimization

```bash
python main.py \
    --optimizer edge_continuous \
    --travel_reduce_ratio 0.2
```

### Node Isolation

```bash
python main.py \
    --optimizer node \
    --travel_reduce_ratio 0.2
```

### Travel Bubble Optimization

```bash
python main.py \
    --optimizer cluster \
    --clusters 3
```

### RK4 Simulation

```bash
python main.py \
    --optimizer cluster \
    --method rk4
```

### Longer Optimization Horizon

```bash
python main.py \
    --fit_days 90 \
    --opt_days 60
```

## Evaluation Metric

The primary intervention metric is the total number of new infections during the optimization horizon.

## Output

The program writes results under the selected `--output_dir`.

Typical output includes:

```text
results/
├── single/
│   └── ...
│
└── network/
    ├── baseline/
    │   └── ...
    │
    ├── optimized/
    │   └── ...
    │
    ├── clusters.png
    ├── cluster_population.png
    └── infected_summary.txt
```

### `infected_summary.txt`

The summary contains values such as:

```text
Baseline infected: ...
Evenly reduced infected: ...
cluster_binary optimized infected: ...
```

These correspond to:

- original travel,
- uniform travel reduction,
- optimized intervention.

## Cluster Visualization

When the cluster optimizer is selected, the program generates:

```text
clusters.png
```

showing the optimized travel bubbles and network structure.

It also generates:

```text
cluster_population.png
```

showing the total population assigned to each travel bubble.

Example:

```bash
python main.py \
    --optimizer cluster \
    --clusters 3
```

## Experimental Comparison

A useful experimental setup is to compare the following strategies under the same network and intervention horizon:

```text
No intervention
Uniform travel reduction
Binary edge reduction
Continuous edge reduction
Node isolation
Travel bubble clustering
```

For each strategy, useful metrics include:

- total new infections,
- infection reduction relative to baseline,
- fraction of travel removed,
- optimization runtime,
- number of edges removed,
- number of nodes isolated,
- number and population of travel bubbles.

This allows the project to study the tradeoff between epidemic control and disruption to mobility.
