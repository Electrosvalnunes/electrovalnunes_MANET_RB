# Explainable Zero-Shot Portability of a Bayesian IDS in FANETs

This repository contains the source code, OMNeT++/INET simulation model, Bayesian
Network, and dataset instructions associated with the study:

**Explainable Assessment of the Zero-Shot Portability of a Bayesian IDS in FANETs**

The framework trains and evaluates a multiclass Bayesian intrusion detection
system in an OMNeT++/AODV source domain and assesses its zero-shot portability
on an external NS-3 dataset.

## Repository structure

```text
.
├── analysis/
│   └── fanet_bn_xai_pipeline.py
├── bayesian_network/
│   └── cyber_physical_bayesian_network.xdsl
├── data/
│   └── README.md
├── docs/
│   └── methodology_summary.md
├── omnetpp/
│   ├── package.ned
│   ├── src/
│   │   ├── BlackholePacketDropper.cc
│   │   └── BlackholePacketDropper.h
│   └── simulations/
│       ├── BlackholePacketDropper.ned
│       ├── FanetRouter.ned
│       ├── electrosvalpadoca_fanet_topologies.ned
│       ├── omnetpp.ini
│       ├── config.xml
│       └── package.ned
├── results/
├── CITATION.cff
├── requirements.txt
└── .gitignore
```

## Experimental design

The OMNeT++/INET source domain uses:

- AODV routing;
- linear UAV mobility at 10 m/s;
- a 1000 m × 1000 m operational area;
- topologies with 30, 50, and 100 UAVs;
- periodic legitimate UDP traffic with 512-byte packets every 1 second;
- Normal, Flooding, and Blackhole scenarios;
- Flooding traffic with 1024-byte packets every 1 ms;
- 50 repetitions per configuration.

The Blackhole implementation preserves the normal AODV control process and
drops packets handled by the malicious node at the IPv4 forwarding hook.

## Software environment

The supplied project was developed around:

- OMNeT++ 6.1;
- INET 4.5.x;
- Python 3.9 or newer;
- GeNIe Modeler for the `.xdsl` Bayesian Network.

Minor configuration adjustments may be required for other OMNeT++ or INET
versions.

## Python installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the Bayesian IDS pipeline

Place the datasets in the `data/` directory and execute:

```bash
python analysis/fanet_bn_xai_pipeline.py \
  --omnet data/Dataset_electrosvalnunes_manet_v2.csv \
  --ns3 data/ns3_like_packet_loss_causes_v1_50k.csv \
  --protocol OLSR \
  --support 1500 \
  --out results/OLSR
```

Protocol-specific examples:

```bash
python analysis/fanet_bn_xai_pipeline.py --omnet data/Dataset_electrosvalnunes_manet_v2.csv --ns3 data/ns3_like_packet_loss_causes_v1_50k.csv --protocol AODV --support 1500 --out results/AODV
python analysis/fanet_bn_xai_pipeline.py --omnet data/Dataset_electrosvalnunes_manet_v2.csv --ns3 data/ns3_like_packet_loss_causes_v1_50k.csv --protocol OLSR --support 1500 --out results/OLSR
python analysis/fanet_bn_xai_pipeline.py --omnet data/Dataset_electrosvalnunes_manet_v2.csv --ns3 data/ns3_like_packet_loss_causes_v1_50k.csv --protocol RPL --support 1500 --out results/RPL
python analysis/fanet_bn_xai_pipeline.py --omnet data/Dataset_electrosvalnunes_manet_v2.csv --ns3 data/ns3_like_packet_loss_causes_v1_50k.csv --protocol DSR --support 1500 --out results/DSR
```

## OMNeT++ model

See [`omnetpp/README.md`](omnetpp/README.md) for installation and execution
instructions. The main configurations are:

- `Normal30`, `Flooding30`, `Blackhole30`;
- `Normal50`, `Flooding50`, `Blackhole50`;
- `Normal100`, `Flooding100`, `Blackhole100`.

## Datasets

Dataset files are intentionally not embedded in this prepared archive because
the local `file:///media/...` paths are accessible only on the author's
computer. See [`data/README.md`](data/README.md) for filenames, provenance, and
publication options.

## Reproducibility note

The Bayesian Network and Python pipeline currently use three quantile states:
`Low`, `Medium`, and `High` (`q = 3`), matching the reported methodology.

## License

No repository-wide license has been assigned in this prepared version. Before
making the repository public, select licenses that are appropriate for the
code, simulation components, and datasets. Third-party datasets remain subject
to their original terms.

## Contact

Osvaldo Lo Nunes Sebastião  
Polytechnic School of the University of São Paulo  
Instituto Superior Técnico Militar, Angola
