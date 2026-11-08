# Importing these modules registers the motion-tracking command and its
# observation / reward / termination terms in the MDP registries.
from .hdmi import command, observations, rewards, terminations  # noqa: F401
from .hdmi.command import RobotTracking  # noqa: F401
