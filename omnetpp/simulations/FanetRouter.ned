package wperm_26.simulations;

import inet.node.inet.ManetRouter;

module FanetRouter extends ManetRouter
{
    parameters:
        bool blackholeEnabled = default(false);

    submodules:
        blackhole: BlackholePacketDropper {
            parameters:
                enabled = parent.blackholeEnabled;
                @display("p=913,375");
        }
}
