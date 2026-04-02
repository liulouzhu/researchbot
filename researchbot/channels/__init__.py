"""Chat channels module with plugin architecture."""

from researchbot.channels.base import BaseChannel
from researchbot.channels.manager import ChannelManager

__all__ = ["BaseChannel", "ChannelManager"]
