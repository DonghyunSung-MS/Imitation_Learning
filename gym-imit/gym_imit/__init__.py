import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='DyrosRed-v0',
    entry_point='gym_imit.envs:DYROSRedEnv'
)

register(
    id='DyrosRed-v1',
    entry_point='gym_imit.envs:RedEnv'
)
register(
    id='HumanoidSource-v0',
    entry_point='gym_imit.envs:SourceCharacter'
)
