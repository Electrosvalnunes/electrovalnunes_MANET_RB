#include "BlackholePacketDropper.h"
#include "inet/common/InitStages.h"

namespace wperm_26 {

Define_Module(BlackholePacketDropper);

int BlackholePacketDropper::numInitStages() const
{
    return NUM_INIT_STAGES;
}

void BlackholePacketDropper::initialize(int stage)
{
    cSimpleModule::initialize(stage);

    if (stage == INITSTAGE_LOCAL) {
        enabled = par("enabled").boolValue();
        packetDroppedSignal = registerSignal("blackholePacketDropped");
        WATCH(enabled);
        WATCH(droppedPackets);
    }
    else if (stage == INITSTAGE_NETWORK_LAYER) {
        const char *modulePath = par("networkProtocolModule").stringValue();
        cModule *module = getModuleByPath(modulePath);
        networkProtocol = dynamic_cast<INetfilter *>(module);

        if (networkProtocol == nullptr)
            throw cRuntimeError("Module '%s' does not exist or does not implement INetfilter", modulePath);

        networkProtocol->registerHook(par("hookPriority").intValue(), this);
    }
}

void BlackholePacketDropper::handleMessage(cMessage *message)
{
    throw cRuntimeError("BlackholePacketDropper cannot receive messages");
}

INetfilter::IHook::Result BlackholePacketDropper::datagramPreRoutingHook(Packet *packet)
{
    Enter_Method_Silent();
    return ACCEPT;
}

INetfilter::IHook::Result BlackholePacketDropper::datagramForwardHook(Packet *packet)
{
    Enter_Method_Silent();

    if (!enabled)
        return ACCEPT;

    droppedPackets++;
    emit(packetDroppedSignal, packet);
    return DROP;
}

INetfilter::IHook::Result BlackholePacketDropper::datagramPostRoutingHook(Packet *packet)
{
    Enter_Method_Silent();
    return ACCEPT;
}

INetfilter::IHook::Result BlackholePacketDropper::datagramLocalInHook(Packet *packet)
{
    Enter_Method_Silent();
    return ACCEPT;
}

INetfilter::IHook::Result BlackholePacketDropper::datagramLocalOutHook(Packet *packet)
{
    Enter_Method_Silent();
    return ACCEPT;
}

void BlackholePacketDropper::finish()
{
    if (networkProtocol != nullptr && isRegisteredHook(networkProtocol))
        networkProtocol->unregisterHook(this);

    recordScalar("blackholeDroppedPackets", droppedPackets);
    cSimpleModule::finish();
}

} // namespace wperm_26
