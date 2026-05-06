'''
Implements story generation from a trait list (pipeline step: profile → story). 
Calls the story prompt builder in prompts.py and uses HFClient to generate a UserStory.
Can add caching or swap story logic without touching the simulator loop.
'''

from __future__ import annotations
from dataclasses import dataclass
from .types import UserProfile, UserStory, GenerationConfig
from .hf_client import HFClient
from .prompts import make_story_messages

@dataclass
class StoryGenerator:
    llm: HFClient
    gen: GenerationConfig

    def generate(self, profile: UserProfile) -> UserStory:
        text = self.llm.chat(make_story_messages(profile), self.gen)
        return UserStory(text=text)