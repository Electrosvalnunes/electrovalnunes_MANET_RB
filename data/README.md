# Datasets

The analysis pipeline expects the following files in this directory:

```text
Dataset_electrosvalnunes_manet_v2.csv
ns3_like_packet_loss_causes_v1_50k.csv
```

## Source domain

`Dataset_electrosvalnunes_manet_v2.csv` contains the OMNeT++/AODV observations
used to train and internally evaluate the Bayesian IDS.

Required fields include:

```text
Scenario
Node_Speed_ms
Neighbor_Count
PDR_Percentage
E2E_Delay_ms
Energy_Consumed_J
Throughput_Kbps
Queue_Drops
Routing_Drops
Control_Packets_Sent
```

## External domain

`ns3_like_packet_loss_causes_v1_50k.csv` is the external NS-3 packet-loss
dataset. Its original causal labels are semantically matched as follows:

```text
benign, mobility, interference -> Normal
congestion                     -> Flooding-like
malicious                      -> Blackhole-like
```

The external dataset was reported as PL-CausesNS3-50k, IEEE DataPort,
DOI: 10.21227/86dj-xd45.

Before redistributing the external CSV, verify that its original license permits
republication. Otherwise, keep only the citation and download instructions in
this repository.

## Adding the local files

On the author's Ubuntu computer, the intended copy commands are:

```bash
cp "/media/osvaldo/UBUNTU 22_0/CAMPIN_SEND_1807/Dataset_electrosvalnunes_manet_v2.csv" data/
cp "/media/osvaldo/UBUNTU 22_0/CAMPIN_SEND_1807/ns3_like_packet_loss_causes_v1_50k.csv" data/
```

Check the file sizes before committing:

```bash
du -h data/*.csv
```

Large datasets may be deposited in Zenodo or another research repository, while
GitHub stores the code, metadata, a small sample, and the persistent data link.
