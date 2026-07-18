package wperm_26.simulations;

@license(LGPL);

import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.physicallayer.wireless.ieee80211.packetlevel.Ieee80211ScalarRadioMedium;
import inet.visualizer.common.IntegratedMultiVisualizer;

module FanetTopologyBase
{
    parameters:
        int numAdditionalNodes;
        @display("bgb=1000,1000");

    submodules:
        source: FanetRouter {
            @display("p=100,500;i=misc/drone");
        }

        destination: FanetRouter {
            @display("p=900,500;i=misc/drone");
        }

        node[numAdditionalNodes]: FanetRouter {
            @display("i=misc/drone");
        }

        radioMedium: Ieee80211ScalarRadioMedium {
            @display("p=930,80");
        }

        configurator: Ipv4NetworkConfigurator {
            @display("p=930,180");
        }

        visualizer: IntegratedMultiVisualizer {
            @display("p=930,280");
        }
}

network electrosvalpadoca_30 extends FanetTopologyBase
{
    parameters:
        numAdditionalNodes = 28;
}

network electrosvalpadoca_50 extends FanetTopologyBase
{
    parameters:
        numAdditionalNodes = 48;
}

network electrosvalpadoca_100 extends FanetTopologyBase
{
    parameters:
        numAdditionalNodes = 98;
}
