"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from genmanip.core.skill.utils import SkillFactory

if TYPE_CHECKING:
    from genmanip.core.scene.scene import Scene


class SkillOptions(BaseModel):
    """Base class for skill options. All fields are optional."""


class SkillConfig(BaseModel):
    name: str = Field(..., description="Name of the skill")
    options: SkillOptions = Field(
        default_factory=SkillOptions, description="Options for the skill"
    )


@SkillFactory.register("base")
class BaseSkill:
    def __init__(self, config_dict: dict = {}, demogen_config: dict = {}):
        self.config = SkillConfig(**config_dict)
        self.demogen_config = demogen_config

    def execute(self, scene: "Scene", *args, **kwargs):
        raise NotImplementedError("execute must be implemented in subclass")
