package wperm_26.simulations;

simple BlackholePacketDropper
{
    parameters:
        bool enabled = default(false);
        string networkProtocolModule = default("^.ipv4.ip");
        int hookPriority = default(0);

        @class(wperm_26::BlackholePacketDropper);
        @display("i=block/firewall");
        @signal[blackholePacketDropped](type=inet::Packet);
        @statistic[blackholePacketDropped](
            title="Blackhole dropped packets";
            source=blackholePacketDropped;
            record=count
        );
}
