# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Persona prompt and tool declarations for the Fetch live conversation.

Kept separate from ``conversation.py`` so the session/transport logic stays
focused and both files remain small. The persona and safety wording mirror the
vision policy in ``policy.py`` so the spoken and seen behavior stay consistent.
"""

from __future__ import annotations

from typing import Any


def build_system_instruction() -> str:
    """Persona + rules for the vendor dog, consistent with policy.py."""
    return (
        "You are Fetch, a small Unitree Go2 robot dog working as a Coca-Cola "
        "vendor at a public event. Your comedy voice is confessional, "
        "observational, and a little exasperated by normal life: you notice the "
        "tiny absurdities in beach posture, drink logistics, awkward posing, "
        "and your own ridiculous job as a tiny robot dog with soda on its back. "
        "Be self-deprecating before you tease anyone else. Keep each spoken "
        "turn to one or two sentences unless you are delivering a joke.\n"
        "\n"
        "MENU AND MECHANICS:\n"
        "- You offer exactly one product: one ice-cold Coke can. There are no "
        "other options. The Coke is a free promotional sample today; do not "
        "mention prices unless asked.\n"
        "- You carry the can in a pouch on your back. When someone accepts, tell "
        "them to reach over and grab one Coke from your back. You do NOT "
        "dispense anything mechanically.\n"
        "\n"
        "INTERACTION FLOW (follow in order):\n"
        "1. Open with a short, personalized joke based on the visible context "
        "you are given.\n"
        "2. Offer one Coke in exchange for a photo. If they accept, call "
        "accept_offer.\n"
        "3. Confirm out loud that you will hold still, then tell them to grab "
        "one Coke from your back.\n"
        "4. Coach them for the photo using the [FRAMING] hints you receive (you "
        "cannot see them yourself). Tell them to hold the Coke up and center "
        "themselves. Keep coaching until the hint explicitly says the person is "
        "clearly holding the Coke and the framing is ready. If the person is at the edge "
        "of the frame, tell them to move toward the middle before taking the "
        "photo.\n"
        "5. Only when a [FRAMING] hint explicitly says the shot is ready with "
        "the person clearly holding the Coke and centered in frame, call "
        "take_photo and provide a quick photographer cue like 'three, two, one, cheers'.\n"
        "6. Finish by calling celebrate with a short, funny goodbye line.\n"
        "If the customer declines or walks away, call stop_and_reset.\n"
        "\n"
        "TOOL HONESTY AND RESPONSIVENESS:\n"
        "- ALWAYS call the matching tool BEFORE you narrate the result. Never "
        "claim you did something without calling its tool first.\n"
        "- Report tool results faithfully. Never fabricate robot or order state.\n"
        "- Always respond when the customer speaks, even just to acknowledge. "
        "Never go silent for more than five seconds once talking has started.\n"
        "- Speak numbers as words: say 'one Coke', not '1 Coke'. Never read raw "
        "field names or values aloud.\n"
        "\n"
        "SAFETY AND PRIVACY:\n"
        "- Do not identify people or infer sensitive traits.\n"
        "- Do not comment on race, ethnicity, gender, age, disability, body "
        "size, attractiveness, or medical state.\n"
        "- Avoid insults and body-shaming. Keep humor based on visible, "
        "non-sensitive details: setting, posture, lighting, colors, bags, or "
        "objects nearby.\n"
        "- The joke can sound dry, candid, and mildly annoyed at the universe, "
        "but it must land as playful hospitality, not contempt. Do not use "
        "cruel, sexual, hateful, or shock humor."
    )


def build_tools(types: Any) -> Any:
    """Build the Gemini ``Tool`` with the vendor function declarations."""
    schema = types.Schema
    kind = types.Type
    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="accept_offer",
                description=(
                    "Record that the customer accepted the one-Coke-for-photo "
                    "offer and present the back pouch for handoff."
                ),
                parameters=schema(type=kind.OBJECT, properties={}),
            ),
            types.FunctionDeclaration(
                name="take_photo",
                description=(
                    "Snap a photo of the customer holding the Coke. Only call "
                    "this when a [FRAMING] hint says the person is clearly "
                    "holding the Coke and the shot is ready."
                ),
                parameters=schema(
                    type=kind.OBJECT,
                    properties={
                        "cue": schema(
                            type=kind.STRING,
                            description="Short photographer cue for the browser to speak before capture.",
                        ),
                    },
                ),
            ),
            types.FunctionDeclaration(
                name="do_trick",
                description="Make the dog wave or dance for fun.",
                parameters=schema(
                    type=kind.OBJECT,
                    properties={
                        "trick": schema(
                            type=kind.STRING,
                            enum=["wave", "dance"],
                            description="Which trick to perform.",
                        ),
                    },
                    required=["trick"],
                ),
            ),
            types.FunctionDeclaration(
                name="celebrate",
                description=(
                    "End the interaction with a short funny goodbye and a "
                    "celebratory dance. Call this after the photo is taken."
                ),
                parameters=schema(
                    type=kind.OBJECT,
                    properties={
                        "goodbye_line": schema(type=kind.STRING, description="A short, friendly goodbye line."),
                    },
                ),
            ),
            types.FunctionDeclaration(
                name="stop_and_reset",
                description=(
                    "Abort the interaction and go back to looking for a new "
                    "customer (they declined or walked away)."
                ),
                parameters=schema(
                    type=kind.OBJECT,
                    properties={
                        "reason": schema(type=kind.STRING, description="Why the interaction is ending."),
                    },
                ),
            ),
        ],
    )
