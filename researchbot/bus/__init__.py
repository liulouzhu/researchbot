"""Message bus module for decoupled channel-agent communication."""

from researchbot.bus.events import InboundMessage, OutboundMessage
from researchbot.bus.queue import MessageBus

__all__ = ["MessageBus", "InboundMessage", "OutboundMessage"]
