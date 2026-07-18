# OMNeT++/INET simulation

## Files

- `src/BlackholePacketDropper.cc`
- `src/BlackholePacketDropper.h`
- `simulations/BlackholePacketDropper.ned`
- `simulations/FanetRouter.ned`
- `simulations/electrosvalpadoca_fanet_topologies.ned`
- `simulations/omnetpp.ini`
- `simulations/config.xml`

## Importing into OMNeT++

1. Create an OMNeT++ project or copy the contents of this directory into an
   existing project.
2. Add the INET project as a project reference.
3. Mark the project directory as a NED source folder.
4. Configure the `src` directory with Makemake.
5. Enable include paths and libraries exported by referenced projects.
6. Clean and rebuild the project.

The generated Makefiles and compiled binaries are not versioned because they
contain environment-specific paths and build products.

## Configurations

Run one of the following configurations from `simulations/omnetpp.ini`:

```text
Normal30      Flooding30      Blackhole30
Normal50      Flooding50      Blackhole50
Normal100     Flooding100     Blackhole100
```

## Blackhole behavior

The malicious node keeps the normal AODV routing application active. The
`BlackholePacketDropper` registers an IPv4 forwarding hook and drops packets
that the node would otherwise forward.

At the end of a run, the scalar result

```text
blackholeDroppedPackets
```

records the number of packets discarded by the malicious node.

## Important

The supplied `config.xml` is included as an auxiliary configurator file. The
current `omnetpp.ini` does not explicitly load it. Add an XML configuration
reference only when required by the intended experiment.
