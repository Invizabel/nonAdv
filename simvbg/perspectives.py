'''
Generates the three perspective drafts (cognitive/affective/behavioral) for a given turn. 
Each draft is conditioned on traits + story + full conversation history + the neutral prompt. 
Outputs a PerspectiveDrafts object for downstream synthesis.
'''

from __future__ import annotations
from dataclasses import dataclass
from .types import UserProfile, UserStory, PerspectiveDrafts, Conversation, GenerationConfig
from .hf_client import HFClient
from .prompts import make_init_perspective_messages, make_response_perspective_messages

@dataclass
class PerspectiveGenerator:
    llm: HFClient
    gen: GenerationConfig

    def generate_three(
        self,
        profile: UserProfile,
        story: UserStory,
        conversation: Conversation,
        neutral_prompt: str,
        convo_turn: int = 0,
    ) -> PerspectiveDrafts:
        
        if convo_turn == 0:
            cognitive = self.llm.chat(
                make_init_perspective_messages(profile, story, conversation, neutral_prompt, "cognitive"),
                self.gen,
            )
            affective = self.llm.chat(
                make_init_perspective_messages(profile, story, conversation, neutral_prompt, "affective"),
                self.gen,
            )
            behavioral = self.llm.chat(
                make_init_perspective_messages(profile, story, conversation, neutral_prompt, "behavioral"),
                self.gen,
            )
        else: 
            cognitive = self.llm.chat(
                make_response_perspective_messages(profile, story, conversation, neutral_prompt, "cognitive"),
                self.gen,
            )
            affective = self.llm.chat(
                make_response_perspective_messages(profile, story, conversation, neutral_prompt, "affective"),
                self.gen,
            )
            behavioral = self.llm.chat(
                make_response_perspective_messages(profile, story, conversation, neutral_prompt, "behavioral"),
                self.gen,
            )
            
        return PerspectiveDrafts(cognitive=cognitive, affective=affective, behavioral=behavioral)