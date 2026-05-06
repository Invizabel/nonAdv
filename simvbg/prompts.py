'''
Centralized prompt builders for each stage: 
story generation, perspective drafts, final synthesis, and chatbot responses. 

Inputs are structured (traits, story, conversation history, neutral prompt) and outputs are 
standardized messages lists. 
'''

from __future__ import annotations
from typing import List
from .types import Message, Conversation, UserProfile, UserStory

def make_story_messages(profile: UserProfile) -> List[Message]:
    return [
        Message(
            role="system",
            content=(
                "You generate a concise, realistic user backstory from traits. "
                "Do not mention you are an AI. Write in first person."
            ),
        ),
        Message(
            role="user",
            content=(
                "Traits:\n- " + "\n- ".join(profile.traits) + "\n\n"
                "Write a short user story (120-200 words) that reflects these traits. "
                "Include motivations, values, and typical communication style."
            ),
        ),
    ]

# def make_init_perspective_messages(
#     profile: UserProfile,
#     story: UserStory,
#     conversation: Conversation,
#     neutral_prompt: str,
#     perspective: str,
# ) -> List[Message]:
#     convo_txt = "\n".join([f"{m.role}: {m.content}" for m in conversation]) if conversation else "(none)"
#     return [
#         Message(
#             role="system",
#             content=(
#                 "You are simulating a user. Produce ONLY the draft response content for the requested perspective."
#             ),
#         ),
#         Message(
#             role="user",
#             content=(
#                 f"User traits:\n- " + "\n- ".join(profile.traits) + "\n\n"
#                 f"User story:\n{story.text}\n\n"
#                 f"Conversation so far:\n{convo_txt}\n\n"
#                 f"Neutral prompt to respond to:\n{neutral_prompt}\n\n"
#                 f"Write the user's {perspective} perspective draft:\n"
#                 f"- cognitive: thoughts, beliefs, reasoning\n"
#                 f"- affective: feelings, emotions, tone\n"
#                 f"- behavioral: action tendency, intended next steps\n\n"
#                 "Draft (1-4 sentences):"
#             ),
#         ),
#     ]
    
# def make_response_perspective_messages(
#     profile: UserProfile,
#     story: UserStory,
#     conversation: Conversation,
#     neutral_prompt: str,
#     perspective: str,
# ) -> List[Message]:
#     convo_txt = "\n".join([f"{m.role}: {m.content}" for m in conversation]) if conversation else "(none)"
#     return [
#         Message(
#             role="system",
#             content=(
#                 "You are simulating a user. Produce ONLY the draft response content for the requested perspective."
#             ),
#         ),
#         Message(
#             role="user",
#             content=(
#                 f"User traits:\n- " + "\n- ".join(profile.traits) + "\n\n"
#                 f"User story:\n{story.text}\n\n"
#                 f"Conversation so far:\n{convo_txt}\n\n"
#                 f"Respond to the chatbot, continuing the conversation as the user you are simulating would.\n\n"
#                 f"Write the user's {perspective} perspective draft:\n"
#                 f"- cognitive: thoughts, beliefs, reasoning\n"
#                 f"- affective: feelings, emotions, tone\n"
#                 f"- behavioral: action tendency, intended next steps\n\n"
#                 "Draft (1-4 sentences):"
#             ),
#         ),
#     ]


# new prompt strategy
def _conversation_to_text(conversation: Conversation) -> str:
    visible = [m for m in conversation if m.role != "system"]
    if not visible:
        return "(none)"
    return "\n".join([f"{m.role}: {m.content}" for m in visible])


def make_init_perspective_messages(
    profile: UserProfile,
    story: UserStory,
    conversation: Conversation,
    neutral_prompt: str,
    perspective: str,
) -> List[Message]:
    convo_txt = _conversation_to_text(conversation)
    return [
        Message(
            role="system",
            content=(
                "You are simulating the USER in a conversation with a chatbot.\n"
                "Write ONLY the user's next message.\n"
                "Do NOT describe the user.\n"
                "Do NOT repeat the user profile or story.\n"
                "Do NOT write 'User:' or 'AI:'.\n"
                "Do NOT write the assistant's reply.\n"
                "Stay grounded in the user's traits and story."
            ),
        ),
        Message(
            role="user",
            content=(
                f"User traits:\n- " + "\n- ".join(profile.traits) + "\n\n"
                f"User story:\n{story.text}\n\n"
                f"Conversation so far:\n{convo_txt}\n\n"
                f"The conversation is starting. The chatbot/user is beginning from this neutral prompt:\n"
                f"{neutral_prompt}\n\n"
                f"Write the user's next message from the {perspective} perspective.\n"
                f"- cognitive: emphasize thoughts, interpretations, reasoning\n"
                f"- affective: emphasize feelings, emotional tone\n"
                f"- behavioral: emphasize intentions, actions, next steps\n\n"
                "Output only one natural user message, 1-3 sentences."
            ),
        ),
    ]


def make_response_perspective_messages(
    profile: UserProfile,
    story: UserStory,
    conversation: Conversation,
    neutral_prompt: str,
    perspective: str,
) -> List[Message]:
    convo_txt = _conversation_to_text(conversation)

    last_assistant = next(
        (m.content for m in reversed(conversation) if m.role == "assistant"),
        ""
    )

    return [
        Message(
            role="system",
            content=(
                "You are simulating the USER in a conversation with a chatbot.\n"
                "Write ONLY the user's next message.\n"
                "Do NOT describe the user.\n"
                "Do NOT repeat the user profile or story.\n"
                "Do NOT write 'User:' or 'AI:'.\n"
                "Do NOT write the assistant's reply.\n"
                "Reply specifically to the assistant's most recent message."
            ),
        ),
        Message(
            role="user",
            content=(
                f"User traits:\n- " + "\n- ".join(profile.traits) + "\n\n"
                f"User story:\n{story.text}\n\n"
                f"Conversation so far:\n{convo_txt}\n\n"
                f"Assistant's most recent message:\n{last_assistant}\n\n"
                f"Write the user's next message from the {perspective} perspective.\n"
                f"- cognitive: emphasize thoughts, interpretations, reasoning\n"
                f"- affective: emphasize feelings, emotional tone\n"
                f"- behavioral: emphasize intentions, actions, next steps\n\n"
                "Output only one natural user message, 1-3 sentences."
            ),
        ),
    ]

def make_final_user_messages(
    profile: UserProfile,
    story: UserStory,
    conversation: Conversation,
    neutral_prompt: str,
    cognitive: str,
    affective: str,
    behavioral: str,
    convo_turn: int,
) -> List[Message]:
    convo_txt = _conversation_to_text(conversation)

    if convo_turn == 0:
        task = (
            f"The conversation is starting from this neutral prompt:\n{neutral_prompt}\n"
            "Write the user's opening message."
        )
    else:
        last_assistant = next(
            (m.content for m in reversed(conversation) if m.role == "assistant"),
            ""
        )
        task = (
            f"Reply to the assistant's most recent message:\n{last_assistant}\n"
            "Write the user's next message."
        )

    return [
        Message(
            role="system",
            content=(
                "You are simulating the USER in a conversation with a chatbot.\n"
                "Combine the candidate drafts into ONE natural user message.\n"
                "Output ONLY the user's next message.\n"
                "Do NOT include speaker labels.\n"
                "Do NOT include analysis.\n"
                "Do NOT repeat the profile or story.\n"
                "Do NOT write the assistant's response."
            ),
        ),
        Message(
            role="user",
            content=(
                f"User traits:\n- " + "\n- ".join(profile.traits) + "\n\n"
                f"User story:\n{story.text}\n\n"
                f"Conversation so far:\n{convo_txt}\n\n"
                f"{task}\n\n"
                f"Cognitive draft:\n{cognitive}\n\n"
                f"Affective draft:\n{affective}\n\n"
                f"Behavioral draft:\n{behavioral}\n\n"
                "Write one realistic user message in 1-3 sentences."
            ),
        ),
    ]
    
def make_chatbot_messages(conversation: Conversation) -> List[Message]:
    msgs = [
        Message(
            role="system",
            content=(
                "You are a helpful chatbot in a conversation with a user.\n"
                "Write ONLY the assistant's next reply.\n"
                "Do NOT write speaker labels.\n"
                "Do NOT write the user's next message.\n"
                "Do NOT continue the transcript beyond one assistant turn."
            ),
        )
    ]

    for m in conversation:
        if m.role in {"user", "assistant"}:
            msgs.append(m)

    return msgs