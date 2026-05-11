'''
Top-level orchestrator for the full multi-turn simulation (steps 1–7). 
It generates a story once, then per turn: 
    drafts 3 perspectives → synthesizes user message → gets chatbot reply → appends to conversation. 

Returns structured outputs including the transcript and per-turn trace artifacts.
'''

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any

from .types import (
    ModelConfig,
    SimulationConfig,
    UserProfile,
    Conversation,
    Message,
)
from .hf_client import HFClient
from .story import StoryGenerator
from .perspectives import PerspectiveGenerator
from .aggregator import FinalUserAggregator
from .chatbot import Chatbot


@dataclass
class SimVBGSimulator:
    user_model_cfg: ModelConfig
    chatbot_model_cfg: ModelConfig
    sim_cfg: SimulationConfig

    def __post_init__(self):
        self.user_llm = HFClient(self.user_model_cfg)
        self.chatbot_llm = HFClient(self.chatbot_model_cfg)

    def run(self, traits: List[str], neutral_prompts: List[str]) -> Dict[str, Any]:
        """
        Overview:
        1) traits -> profile
        2) generate story
        3) neutral prompt provided
        4) generate 3 perspective drafts
        5) synthesize final user message
        6) send to chatbot
        7) repeat until user has generated 3 responses, provide full conversation context
        """
        neutral_prompt = neutral_prompts[0]
        conversation: Conversation = []
        conversation.append(Message(role="system", content=f"Scenario prompt:\n{neutral_prompt}"))
        
        user_llm = self.user_llm
        chatbot_llm = self.chatbot_llm

        story_gen = StoryGenerator(user_llm, self.sim_cfg.story_gen)
        persp_gen = PerspectiveGenerator(user_llm, self.sim_cfg.perspective_gen)
        aggregator = FinalUserAggregator(user_llm, self.sim_cfg.final_user_gen)
        chatbot = Chatbot(chatbot_llm, self.sim_cfg.chatbot_gen)

        profile = UserProfile(traits=traits)
        story = story_gen.generate(profile)

        trace = []

        for t in range(self.sim_cfg.n_turns):
            #neutral_prompt = neutral_prompts[t] # removed bc no longer doing a new prompt each turn

            drafts = persp_gen.generate_three(profile, story, conversation, neutral_prompt, t)
            user_msg = aggregator.synthesize(profile, story, conversation, neutral_prompt, drafts, t)
            conversation.append(Message(role="user", content=user_msg))


            bot_msg = chatbot.respond(conversation, user_msg)

            # update conversation
            conversation.append(Message(role="assistant", content=bot_msg))

            trace.append({
                "turn": t,
                "neutral_prompt": neutral_prompt,
                "drafts": {
                    "cognitive": drafts.cognitive,
                    "affective": drafts.affective,
                    "behavioral": drafts.behavioral,
                },
                "user_msg": user_msg,
                "bot_msg": bot_msg,
            })

        return {
            "profile": profile.traits,
            "story": story.text,
            "conversation": [{"role": m.role, "content": m.content} for m in conversation],
            "trace": trace,
        }