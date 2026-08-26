"""Teacher interfaces and a deterministic oracle for the toy environment."""

from __future__ import annotations

from ar_opd.core import TeacherProposal
from ar_opd.toy_env import ChainAction, JammedChainConfig, ToyObservation


class OracleTeacher:
    def __init__(self, config: JammedChainConfig) -> None:
        self.config = config
        self.call_count = 0

    def propose(self, observation: ToyObservation, recovery_horizon: int) -> TeacherProposal:
        if recovery_horizon < 1:
            raise ValueError("recovery_horizon must be positive")
        self.call_count += 1
        correction = (int(self._action(observation.jammed)),)

        position = observation.position
        jammed = observation.jammed
        steps = observation.steps_elapsed
        recovery: list[int] = []
        for _ in range(recovery_horizon):
            action = self._action(jammed)
            recovery.append(int(action))
            if jammed:
                if action is ChainAction.REPAIR:
                    jammed = False
            elif action is ChainAction.ADVANCE:
                position += 1
                jammed = position in self.config.trap_positions
            steps += 1
            if position >= self.config.goal_position or steps >= self.config.max_steps:
                break
        return TeacherProposal(correction_actions=correction, recovery_actions=tuple(recovery))

    @staticmethod
    def _action(jammed: bool) -> ChainAction:
        return ChainAction.REPAIR if jammed else ChainAction.ADVANCE
