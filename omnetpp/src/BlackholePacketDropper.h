#ifndef __WPERM26_BLACKHOLEPACKETDROPPER_H
#define __WPERM26_BLACKHOLEPACKETDROPPER_H

#include <omnetpp.h>
#include "inet/networklayer/contract/INetfilter.h"

using namespace omnetpp;
using namespace inet;

namespace wperm_26 {

class BlackholePacketDropper : public cSimpleModule, public NetfilterBase::HookBase
{
  protected:
    INetfilter *networkProtocol = nullptr;
    bool enabled = false;
    long droppedPackets = 0;
    simsignal_t packetDroppedSignal;

    virtual int numInitStages() const override;
    virtual void initialize(int stage) override;
    virtual void handleMessage(cMessage *message) override;
    virtual void finish() override;

    virtual Result datagramPreRoutingHook(Packet *packet) override;
    virtual Result datagramForwardHook(Packet *packet) override;
    virtual Result datagramPostRoutingHook(Packet *packet) override;
    virtual Result datagramLocalInHook(Packet *packet) override;
    virtual Result datagramLocalOutHook(Packet *packet) override;
};

} // namespace wperm_26

#endif
