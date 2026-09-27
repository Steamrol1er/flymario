"""ViZDoom's stock `defend_the_center` scenario (bundled with the `vizdoom`
pip package -- no custom WAD needed), wrapped for the Fly Marksman
reproduction (experiments/run_marksman.py). Native interface: the game's
own 3 buttons (TURN_LEFT, TURN_RIGHT, ATTACK), confirmed by inspection
(`g.get_available_buttons()`) to match exactly what
data/marksman_reference/fruitfly_lab_fly_doom.py assumes.

The one non-native piece, inherited faithfully from that reference rather
than invented here: enemy bearing comes from ViZDoom's object-label buffer
(which objects are on screen and where), not from raw pixels. Their own
code calls this out as engineered ("ViZDoom object labels give enemy
bearing, substituting for compound eye optics") -- reading real photoreceptor-
level pixels into LC10a properly is the harder "Stage B" problem noted
throughout this project's docs/PLAN.md, not attempted here either.
"""
from __future__ import annotations

import os

import numpy as np
import vizdoom as vzd

HALF_FOV_DOOM = 45.0   # Doom's horizontal FOV is 90 degrees
FLY_SCALE = 2.0        # map the screen's +/-45 deg to the fly's +/-90 deg visual field


class MarksmanEnv:
    def __init__(self, seed: int = 0, visible: bool = False):
        self.game = vzd.DoomGame()
        self.game.load_config(os.path.join(vzd.scenarios_path, "defend_the_center.cfg"))
        self.game.set_window_visible(visible)
        self.game.set_mode(vzd.Mode.ASYNC_PLAYER if visible else vzd.Mode.PLAYER)
        self.game.set_labels_buffer_enabled(True)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.add_available_game_variable(vzd.GameVariable.KILLCOUNT)
        self.game.set_seed(seed)
        self.game.init()
        assert self.game.get_available_buttons() == [
            vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.ATTACK
        ]

    def new_episode(self):
        self.game.new_episode()

    def is_finished(self) -> bool:
        return self.game.is_episode_finished()

    def enemies(self) -> list[tuple[float, float]]:
        """[(azimuth_deg_in_fly_frame, strength_0_to_1), ...] for every
        visible living enemy (corpses are wider than tall; skip those)."""
        state = self.game.get_state()
        if state is None:
            return []
        out = []
        for lb in state.labels:
            if lb.object_name == "DoomPlayer" or lb.height < 4 or lb.width > lb.height * 1.3:
                continue
            cx = lb.x + lb.width / 2
            az = (cx - 160) / 160 * HALF_FOV_DOOM * FLY_SCALE
            strength = min(1.0, 0.35 + lb.height / 60)
            out.append((float(np.clip(az, -90, 90)), float(strength)))
        return out

    def step(self, left: int, right: int, fire: int):
        self.game.make_action([int(left), int(right), int(fire)])

    def kills(self) -> int:
        return int(self.game.get_game_variable(vzd.GameVariable.KILLCOUNT))

    def ammo(self) -> int:
        return int(self.game.get_game_variable(vzd.GameVariable.AMMO2))

    def tics(self) -> int:
        return int(self.game.get_episode_time())

    def close(self):
        self.game.close()
