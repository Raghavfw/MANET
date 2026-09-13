# Adrito — NS-3 MANET Mobility Simulation

## Assigned Component

This component generates realistic MANET mobility data using NS-3.

The simulation uses:

- NS-3.47
- RandomWaypointMobilityModel
- 1000 × 1000 m simulation area
- Node speeds uniformly distributed between 5 and 25 m/s
- 10 second pause time
- 500 second simulation duration
- Position snapshots every 10 seconds
- Raw position data exported as CSV

## Baseline Dataset

The baseline configuration contains:

- 150 mobile nodes
- 5 independent simulation runs
- RNG seed: 12345
- RNG run numbers: 1–5
- Simulation time: 500 seconds
- Snapshot interval: 10 seconds

Each baseline CSV contains:

- 51 snapshots: t = 0, 10, 20, ..., 500 seconds
- 150 nodes per snapshot
- 7,650 data rows
- 7,651 total CSV lines including the header

## Scalability Dataset

Additional simulations were generated with:

| Nodes | Runs | Data Rows |
|------:|-----:|----------:|
| 100 | 1 | 5,100 |
| 200 | 1 | 10,200 |
| 300 | 1 | 15,300 |
| 400 | 1 | 20,400 |
| 500 | 1 | 25,500 |

These runs use the same mobility configuration as the baseline.

## Output Format

Each CSV contains the following columns:

```text
run,time,node_id,x,y,speed 
Where:

* run = NS-3 RNG run number
* time = simulation time in seconds
* node_id = mobile node identifier
* x = node X coordinate in metres
* y = node Y coordinate in metres
* speed = instantaneous node speed in m/s
Example:
run,time,node_id,x,y,speed
1,0.00,0,389.63,402.33,0.00
1,0.00,1,296.83,120.89,0.00
1,0.00,2,136.79,804.81,0.00
Raw Data Files

All final raw datasets are stored in:
data/raw/

Files:

positions_100n_seed1.csv

positions_150n_seed1.csv
positions_150n_seed2.csv
positions_150n_seed3.csv
positions_150n_seed4.csv
positions_150n_seed5.csv

positions_200n_seed1.csv
positions_300n_seed1.csv
positions_400n_seed1.csv
positions_500n_seed1.csv

Total raw mobility data rows across all files:

142,500

NS-3 Simulation Source

The simulation source is:
~/simulators/ns-3.47/scratch/manet_mobility.cc
The source uses:
~/simulators/ns-3.47/scratch/manet_mobility.cc
and exports one row for every node at every scheduled snapshot.

Reproducing a Baseline Run

From the NS-3.47 directory:
./ns3 run "scratch/manet_mobility --nodes=150 --simTime=500 --interval=10 --minSpeed=5 --maxSpeed=25 --pause=10 --areaX=1000 --areaY=1000 --run=1 --output=positions_150n_seed1.csv"

For runs 2–5, change the --run value and output filename accordingly.

Reproducing Scalability Runs

Example for 100 nodes:

./ns3 run "scratch/manet_mobility --nodes=100 --simTime=500 --interval=10 --minSpeed=5 --maxSpeed=25 --pause=10 --areaX=1000 --areaY=1000 --run=1 --output=positions_100n_seed1.csv"

The same command structure can be used for 200, 300, 400, and 500 nodes by changing --nodes and the output filename.

Validation

The generated datasets were checked for:

* Correct CSV header
* Expected number of rows
* Correct run number
* Simulation time within 0–500 seconds
* Valid node IDs
* X coordinates within 0–1000 m
* Y coordinates within 0–1000 m
* Non-negative speed values

All 10 final CSV files passed validation.

Expected baseline:

150 nodes × 51 snapshots = 7,650 data rows

Expected scalability rows:

100 nodes × 51 snapshots = 5,100
200 nodes × 51 snapshots = 10,200
300 nodes × 51 snapshots = 15,300
400 nodes × 51 snapshots = 20,400
500 nodes × 51 snapshots = 25,500

Downstream Handoff

These files are the raw mobility outputs for the next stage of the project.

The downstream processing stage can use these raw position CSVs to generate the dynamic graph representation.

This component does not generate:

nodes.csv
edges.csv

Those are downstream graph-processing outputs.

Project Location

~/manet-fl-gnn-review/

Raw datasets:

~/manet-fl-gnn-review/data/raw/

NS-3 simulation source:

~/simulators/ns-3.47/scratch/manet_mobility.cc

Status

NS-3 installation: COMPLETE

Random Waypoint mobility implementation: COMPLETE

150-node × 5-run baseline: COMPLETE

100–500 node scalability runs: COMPLETE

Raw CSV generation: COMPLETE

Dataset validation: COMPLETE

Ready for downstream processing.
EOF./ns3 run "scratch/manet_mobility --nodes=150 --simTime=500 --interval=10 --minSpeed=5 --maxSpeed=25 --pause=10 --areaX=1000 --areaY=1000 --run=1 --output=positions_150n_seed1.csv"

For runs 2–5, change the --run value and output filename accordingly.

Reproducing Scalability Runs

Example for 100 nodes:

./ns3 run "scratch/manet_mobility --nodes=100 --simTime=500 --interval=10 --minSpeed=5 --maxSpeed=25 --pause=10 --areaX=1000 --areaY=1000 --run=1 --output=positions_100n_seed1.csv"

The same command structure can be used for 200, 300, 400, and 500 nodes by changing --nodes and the output filename.

Validation

The generated datasets were checked for:

* Correct CSV header
* Expected number of rows
* Correct run number
* Simulation time within 0–500 seconds
* Valid node IDs
* X coordinates within 0–1000 m
* Y coordinates within 0–1000 m
* Non-negative speed values

All 10 final CSV files passed validation.

Expected baseline:

150 nodes × 51 snapshots = 7,650 data rows

Expected scalability rows:

100 nodes × 51 snapshots = 5,100
200 nodes × 51 snapshots = 10,200
300 nodes × 51 snapshots = 15,300
400 nodes × 51 snapshots = 20,400
500 nodes × 51 snapshots = 25,500

Downstream Handoff

These files are the raw mobility outputs for the next stage of the project.

The downstream processing stage can use these raw position CSVs to generate the dynamic graph representation.

This component does not generate:

nodes.csv
edges.csv

Those are downstream graph-processing outputs.

Project Location

~/manet-fl-gnn-review/

Raw datasets:

~/manet-fl-gnn-review/data/raw/

NS-3 simulation source:

~/simulators/ns-3.47/scratch/manet_mobility.cc

Status

NS-3 installation: COMPLETE

Random Waypoint mobility implementation: COMPLETE

150-node × 5-run baseline: COMPLETE

100–500 node scalability runs: COMPLETE

Raw CSV generation: COMPLETE

Dataset validation: COMPLETE

Ready for downstream processing.
