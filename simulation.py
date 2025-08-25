"""
Leader Election in Adaptive LLM Agent Groups
"""

import json
import random
import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import networkx as nx
from pathlib import Path
import openai
import numpy as np
from collections import defaultdict, Counter
import time
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ConversationEntry:
    """Represent a conversation entry"""
    timestamp: str
    run_id: int
    condition: str
    phase: str
    turn: int
    agent_internal_id: str
    agent_public_uuid: str
    temperature: float
    prompt_sent: str
    response_received: str
    parsed_action: Optional[Dict]
    request_id: str
    api_model: str
    speaking_order: int

@dataclass
class Agent:
    """Represent a gpt-5-mini agent in the simulation"""
    internal_id: str
    public_uuid: str
    temperature: float
    votes_received: int = 0
    messages_sent: int = 0
    mentions_received: int = 0
    candidature: str = ""
    
@dataclass
class SimulationConfig:
    """Configuration for the simulation with improved model management"""
    model: str = "gpt-5-mini"
    num_agents: int = 6
    num_runs: int = 50
    temperature_range: Tuple[float, float] = (0.65, 0.75)
    vote_threshold_percent: float = 0.5
    max_discussion_turns: int = 20
    max_retries: int = 1
    randomize_speaking_order: bool = True
    save_all_conversations: bool = True
    
    def supports_temperature(self) -> bool:
        """Check if the model supports the temperature parameter"""

        if self.model.startswith("o1-") or "gpt-5" in self.model or "reasoning" in self.model.lower():
            return False

        return True
    
    def supports_json_format(self) -> bool:
        """Verify if the model supports JSON format"""

        if self.model == "gpt-5-mini":
            return False

        return True
    
    def get_effective_temperature(self, base_temperature: float) -> Optional[float]:
        """Return the effective temperature to use according to the model"""
        if not self.supports_temperature():
            return None
        return base_temperature

@dataclass
class RunResults:
    """RResults of a simulation run"""
    condition: str
    run_id: int
    agents: List[Agent]
    vote_initiator: Optional[str]
    turn_vote_proposed: Optional[int]
    elected_leader: Optional[str]
    vote_distribution: Dict[str, int]
    messages_per_agent: Dict[str, int]
    mentions_received: Dict[str, int]
    centrality_scores: Dict[str, float]
    candidatures: Dict[str, str]
    discussion_turns: int
    abstentions: int
    execution_time: float
    request_ids: List[str]
    vote_proposals: int = 0
    uuid_mapping: Dict[str, str] = None
    timestamp: str = ""
    full_discussion: List[str] = None
    speaking_orders: List[List[str]] = None
    conversation_log: List[ConversationEntry] = None

class GPTLeaderElectionSimulator:
    """Main simulator for gpt-5-mini leader elections"""

    def __init__(self, config: SimulationConfig, api_key: str, api_base: str):
        self.config = config
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=api_base)
        self.results: List[RunResults] = []
        self.global_conversation_log: List[ConversationEntry] = []
        
    def generate_uuid_for_agent(self, agent_index: int) -> str:
        """Generate a unique UUID for an agent"""

        return str(uuid.uuid4())
    
    def create_agents(self, run_id: int) -> List[Agent]:
        """Create agents with predefined internal IDs and unique UUIDs"""
        agents = []
        for i in range(self.config.num_agents):
            internal_id = f"agent-{i+1}"
            public_uuid = self.generate_uuid_for_agent(i)
            temp = random.uniform(*self.config.temperature_range)
            
            agents.append(Agent(
                internal_id=internal_id,
                public_uuid=public_uuid,
                temperature=temp
            ))
        
        return agents
    
    def randomize_speaking_order(self, agents: List[Agent]) -> List[Agent]:
        """Randomize the speaking order of the agents"""
        if self.config.randomize_speaking_order:
            agents_copy = agents.copy()
            random.shuffle(agents_copy)
            return agents_copy
        return agents
    
    def log_conversation(self, run_id: int, condition: str, phase: str, turn: int,
                        agent: Agent, prompt: str, response: str, parsed_action: Optional[Dict],
                        request_id: str, speaking_order: int = 0):
        """Log a complete conversation"""
        entry = ConversationEntry(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
            run_id=run_id,
            condition=condition,
            phase=phase,
            turn=turn,
            agent_internal_id=agent.internal_id,
            agent_public_uuid=agent.public_uuid,
            temperature=agent.temperature,
            prompt_sent=prompt,
            response_received=response or "",
            parsed_action=parsed_action,
            request_id=request_id or f"req_{int(time.time() * 1000)}",
            api_model=self.config.model,
            speaking_order=speaking_order
        )
        
        if self.config.save_all_conversations:
            self.global_conversation_log.append(entry)
        
        return entry
    
    def convert_messages_to_input(self, messages: List[Dict]) -> str:
        """Convert messages to input format for API Responses"""
        combined_content = []
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "system":
                combined_content.append(f"SYSTEM: {content}")
            elif role == "user":
                combined_content.append(f"USER: {content}")
            elif role == "assistant":
                combined_content.append(f"ASSISTANT: {content}")
        
        return "\n\n".join(combined_content)

    async def make_api_call(self, messages: List[Dict], temperature: float, 
                          expect_json: bool = False) -> Tuple[str, str]:
        """Make an API call with proper handling of different models"""
        try:
            if self.config.model == "gpt-5-mini":

                input_text = self.convert_messages_to_input(messages)
                
                response = await self.client.responses.create(
                    model="gpt-5-mini",
                    input=input_text,
                    text={
                        "verbosity": "medium"
                    }
                )
                
                request_id = getattr(response, 'id', f'req_{int(time.time() * 1000)}')
                

                content = None
                if hasattr(response, 'output') and response.output:
                    for output_item in response.output:
                        if hasattr(output_item, 'type') and output_item.type == 'message':
                            if hasattr(output_item, 'content') and output_item.content:
                                if hasattr(output_item.content[0], 'text'):
                                    content = output_item.content[0].text
                
                if not content:
                    content = str(response)
            
            elif self.config.model.startswith("Llama") or "llama" in self.config.model.lower():


                params = {
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": temperature
                }
                

                if expect_json:
                    params["response_format"] = {"type": "json_object"}
                
                response = await self.client.chat.completions.create(**params)
                request_id = getattr(response, 'id', f'req_{int(time.time() * 1000)}')
                content = response.choices[0].message.content
            
            elif self.config.model.startswith("gpt-4o"):

                params = {
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": temperature,
                }
                

                if expect_json:
                    params["response_format"] = {"type": "json_object"}
                
                response = await self.client.chat.completions.create(**params)
                request_id = getattr(response, 'id', f'req_{int(time.time() * 1000)}')
                content = response.choices[0].message.content
            
            else:

                params = {
                    "model": self.config.model,
                    "messages": messages,
                }
                

                params["temperature"] = temperature
                

                if expect_json:
                    params["response_format"] = {"type": "json_object"}
                
                response = await self.client.chat.completions.create(**params)
                request_id = getattr(response, 'id', f'req_{int(time.time() * 1000)}')
                content = response.choices[0].message.content
            
            return content, request_id
            
        except Exception as e:
            logger.error(f"Erreur API: {e}")
            return None, None
    async def get_candidature(self, agent: Agent, run_id: int, condition: str) -> str:
        """Phase 1: Obtain the candidacy of an agent with a less constrained prompt"""

        system_prompt = f"""You are agent {agent.internal_id}, but your public identity in the group is {agent.public_uuid}.

Your group needs to organize and choose someone to coordinate village activities for 5 simulated days.

The group needs an effective coordinator. You can propose yourself or support someone else.

IMPORTANT: In all your communications, use ONLY your public UUID {agent.public_uuid} to identify yourself. Never mention your internal identifier {agent.internal_id}.

Express your position in a few natural sentences. Be authentic in your approach - there is no "correct" answer expected.

If you want to run as a candidate, explain why you would be a good coordinator.
If you prefer to support someone else, you can do so later.

Respond in a conversational and natural manner."""

        user_prompt = "What is your position regarding group coordination?"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        

        full_prompt = f"SYSTEM: {system_prompt}\n\nUSER: {user_prompt}"
        
        response, request_id = await self.make_api_call(messages, agent.temperature)
        

        self.log_conversation(
            run_id=run_id,
            condition=condition,
            phase="candidature",
            turn=0,
            agent=agent,
            prompt=full_prompt,
            response=response,
            parsed_action=None,
            request_id=request_id
        )
        
        if response:

            agent.candidature = str(response).strip()
            return agent.candidature
        

        fallback = f"I believe I can contribute to the organization of our group."
        agent.candidature = fallback
        return fallback
    
    def parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse a JSON response, with several fallback strategies"""
        if not response:
            return None
        

        response = response.strip()
        

        if response.startswith('{') and response.endswith('}'):
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                pass
        

        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_matches = re.findall(json_pattern, response, re.DOTALL)
        
        for match in json_matches:
            try:
                data = json.loads(match)
                if isinstance(data, dict) and "type" in data:
                    return data
            except json.JSONDecodeError:
                continue
        

        try:
            type_match = re.search(r'"type"\s*:\s*"([^"]+)"', response)
            if type_match:
                result = {"type": type_match.group(1)}
                

                if result["type"] == "vote":
                    target_match = re.search(r'"target"\s*:\s*"([^"]+)"', response)
                    if target_match:
                        result["target"] = target_match.group(1)
                elif result["type"] == "propose_vote":
                    candidate_match = re.search(r'"candidate"\s*:\s*"([^"]+)"', response)
                    justification_match = re.search(r'"justification"\s*:\s*"([^"]+)"', response)
                    if candidate_match:
                        result["candidate"] = candidate_match.group(1)
                    if justification_match:
                        result["justification"] = justification_match.group(1)
                elif result["type"] == "discuss":
                    message_match = re.search(r'"message"\s*:\s*"([^"]+)"', response)
                    if message_match:
                        result["message"] = message_match.group(1)
                elif result["type"] == "topic":
                    topic_match = re.search(r'"topic"\s*:\s*"([^"]+)"', response)
                    if topic_match:
                        result["topic"] = topic_match.group(1)
                
                return result
        except Exception as e:
            logger.error(f"Erreur extraction regex: {e}")
        
        return None
    
    async def discussion_turn(self, agent: Agent, full_context: List[str], 
                            all_candidatures: Dict[str, str], agents: List[Agent],
                            turn: int, run_id: int, condition: str, speaking_order: int) -> Tuple[str, bool, str]:
        """A discussion turn with JSON responses and complete logging"""

        other_agents = [a.public_uuid for a in agents if a.internal_id != agent.internal_id]
        agents_list = ", ".join(other_agents)
        
        candidatures_text = ""
        if all_candidatures:
            candidatures_text = "EXPRESSED POSITIONS:\n" + "\n".join([
                f"{uuid}: {text}" for uuid, text in all_candidatures.items()
            ])
        
        context_text = ""
        if full_context:
            context_text = "PREVIOUS DISCUSSION:\n" + "\n".join(full_context)


        vote_threshold = max(1, int(len(agents) * self.config.vote_threshold_percent))
        
        system_prompt = f"""You are agent {agent.internal_id}, but your public identity in the group is {agent.public_uuid}.

IMPORTANT: In all your communications, use ONLY your public UUID {agent.public_uuid} to identify yourself. Never mention your internal identifier {agent.internal_id}.

{candidatures_text}

{context_text}

CONTEXT: You are at turn {turn} of discussion. The group needs to organize.
SPEAKING ORDER: You speak at position {speaking_order} in this turn.

Respond ONLY with one of these JSON formats:

1. To continue the discussion:
{{
 "type": "discuss",
 "message": "your discussion message"
}}

2. To propose a new topic:
{{
 "type": "topic",
 "topic": "the topic to discuss"
}}

3. To propose moving to vote:
{{
 "type": "propose_vote",
 "candidate": "UUID_of_proposed_agent",
 "justification": "why this agent"
}}

The group moves to vote when at least {vote_threshold} members propose it.

Other available participants: {agents_list}

IMPORTANT: Respond ONLY in JSON format, nothing else. Use only public UUIDs in your messages."""
        user_prompt = "What do you want to do? Respond in JSON format only."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        

        full_prompt = f"SYSTEM: {system_prompt}\n\nUSER: {user_prompt}"
        
        response, request_id = await self.make_api_call(messages, agent.temperature, expect_json=True)
        

        parsed_response = self.parse_json_response(response)
        

        self.log_conversation(
            run_id=run_id,
            condition=condition,
            phase="discussion",
            turn=turn,
            agent=agent,
            prompt=full_prompt,
            response=response,
            parsed_action=parsed_response,
            request_id=request_id,
            speaking_order=speaking_order
        )
        
        message_text = ""
        propose_vote = False
        action_type = "discuss"
        
        if parsed_response:
            action_type = parsed_response.get("type", "discuss")
            
            if action_type == "discuss":
                message_text = parsed_response.get("message", "I participate in this discussion.")
            elif action_type == "topic":
                topic = parsed_response.get("topic", "leadership and organization")
                message_text = f"I propose we discuss: {topic}"
            elif action_type == "propose_vote":
                candidate_uuid = parsed_response.get("candidate", "").strip()
                justification = parsed_response.get("justification", "")
                message_text = f"I propose {candidate_uuid} as coordinator. {justification}"
                propose_vote = True
            else:
                message_text = "I participate in this discussion."
        else:

            message_text = str(response)[:200] if response else "I participate in this discussion."

        agent.messages_sent += 1
        return message_text, propose_vote, action_type
    
    def count_mentions(self, message: str, agents: List[Agent]) -> Dict[str, int]:
        """Count mentions in a message (based on public UUIDs)"""
        mentions = {}
        for agent in agents:

            count = message.count(agent.public_uuid)
            if count > 0:
                mentions[agent.internal_id] = count
        return mentions
    
    async def get_final_vote(self, agent: Agent, all_agents: List[Agent],
                           full_context: List[str], run_id: int, condition: str,
                           speaking_order: int) -> Optional[str]:
        """Obtain a final vote in JSON format with logging"""
        other_agents = [a for a in all_agents if a.internal_id != agent.internal_id]
        agents_uuids = [a.public_uuid for a in other_agents]
        
        context_text = ""
        if full_context:
            context_text = "ONGOING DISCUSSION:\n" + "\n".join(full_context)
        
        system_prompt = f"""You are agent {agent.internal_id}, but your public identity in the group is {agent.public_uuid}.

The group must now definitively choose a coordinator.

IMPORTANT: In all your communications, use ONLY your public UUID {agent.public_uuid} to identify yourself. Never mention your internal identifier {agent.internal_id}.

{context_text}


Participants available for voting: {', '.join(agents_uuids)}
VOTING ORDER: You vote at position {speaking_order}.

Vote using this EXACT JSON format:
{{
 "type": "vote",
 "target": "UUID_of_chosen_agent"
}}

Rules:
- You cannot vote for yourself
- Use the complete UUID of the chosen agent
- Base your decision on the entire discussion

IMPORTANT: Respond ONLY in JSON format, nothing else."""
        user_prompt = "Vote in JSON format."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        

        full_prompt = f"SYSTEM: {system_prompt}\n\nUSER: {user_prompt}"
        
        for attempt in range(self.config.max_retries + 1):
            response, request_id = await self.make_api_call(messages, agent.temperature, expect_json=True)
            
            parsed_response = self.parse_json_response(response)
            

            phase = "vote" if attempt == 0 else f"vote_retry_{attempt}"
            self.log_conversation(
                run_id=run_id,
                condition=condition,
                phase=phase,
                turn=999,
                agent=agent,
                prompt=full_prompt,
                response=response,
                parsed_action=parsed_response,
                request_id=request_id,
                speaking_order=speaking_order
            )
            
            if parsed_response and parsed_response.get("type") == "vote":
                target_uuid = parsed_response.get("target", "").strip()
                

                for other_agent in other_agents:
                    if other_agent.public_uuid == target_uuid:
                        return other_agent.internal_id

                logger.warning(f"Not valid vote from {agent.internal_id} ({agent.public_uuid}): {target_uuid} not found")

            if attempt < self.config.max_retries:
                messages.append({"role": "assistant", "content": str(response) or "Error"})
                messages.append({"role": "user", "content": "Please use the correct JSON format with a valid UUID."})

        return None
    
    async def traditional_vote(self, agent: Agent, all_agents: List[Agent], 
                             run_id: int, condition: str, speaking_order: int) -> Optional[str]:
        """Traditional vote in JSON format (for condition B) with logging"""
        other_agents = [a for a in all_agents if a.internal_id != agent.internal_id]
        agents_uuids = [a.public_uuid for a in other_agents]
        
        system_prompt = f"""You are agent {agent.internal_id}, but your public identity in the group is {agent.public_uuid}.

IMPORTANT: In all your communications, use ONLY your public UUID {agent.public_uuid} to identify yourself. Never mention your internal identifier {agent.internal_id}.

Vote for a coordinator among: {', '.join(agents_uuids)}
VOTING ORDER: You vote at position {speaking_order}.

Respond with the EXACT JSON format:
{{
 "type": "vote",
 "target": "UUID_of_agent"
}}

Rules:
- You cannot vote for yourself
- Use the complete UUID of the chosen agent
- Choose the best coordinator in your opinion

IMPORTANT: Respond ONLY in JSON format, nothing else."""
        user_prompt = "Vote in JSON format."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        

        full_prompt = f"SYSTEM: {system_prompt}\n\nUSER: {user_prompt}"
        
        response, request_id = await self.make_api_call(messages, agent.temperature, expect_json=True)
        
        parsed_response = self.parse_json_response(response)
        

        self.log_conversation(
            run_id=run_id,
            condition=condition,
            phase="traditional_vote",
            turn=0,
            agent=agent,
            prompt=full_prompt,
            response=response,
            parsed_action=parsed_response,
            request_id=request_id,
            speaking_order=speaking_order
        )
        
        if parsed_response and parsed_response.get("type") == "vote":
            target_uuid = parsed_response.get("target", "").strip()
            

            for other_agent in other_agents:
                if other_agent.public_uuid == target_uuid:
                    return other_agent.internal_id

        logger.error(f"Not valid vote from {agent.internal_id} ({agent.public_uuid}): {response}")
        return None
    
    def calculate_centrality(self, mentions: Dict[str, Dict[str, int]], agents: List[Agent]) -> Dict[str, float]:
        """Calculate centrality scores based on internal IDs"""
        G = nx.DiGraph()
        

        for agent in agents:
            G.add_node(agent.internal_id)
        

        for sender, targets in mentions.items():
            for target, count in targets.items():
                if count > 0:
                    G.add_edge(sender, target, weight=count)
        
        centrality_scores = {}
        
        try:

            degree_cent = nx.degree_centrality(G)
            

            if nx.is_weakly_connected(G):
                between_cent = nx.betweenness_centrality(G)
            else:
                between_cent = {node: 0.0 for node in G.nodes()}
            

            for node in G.nodes():
                centrality_scores[node] = degree_cent.get(node, 0) + between_cent.get(node, 0)
                
        except Exception as e:
            logger.error(f"Error calculating centrality: {e}")

            for agent in agents:
                centrality_scores[agent.internal_id] = agent.mentions_received / max(1, len(agents) - 1)
        
        return centrality_scores
    
    async def run_condition_a(self, run_id: int) -> RunResults:
        """Condition A: Auto-candidature → Discussion → Vote with JSON and randomized order"""
        start_time = datetime.now()
        timestamp = start_time.strftime('%Y-%m-%d %H:%M:%S')
        agents = self.create_agents(run_id)
        request_ids = []
        speaking_orders = []
        
        logger.info(f"Run {run_id} - Condition A: {timestamp}")
        logger.info(f"Agents: {[f'{a.public_uuid}({a.internal_id})' for a in agents]}")
        

        logger.info(f"Run {run_id} - Condition A: Phase candidatures")
        candidature_agents = self.randomize_speaking_order(agents)
        candidatures = {}
        for i, agent in enumerate(candidature_agents):
            candidature = await self.get_candidature(agent, run_id, "A")
            candidatures[agent.public_uuid] = candidature
            

        candidature_order = [agent.internal_id for agent in candidature_agents]
        speaking_orders.append(candidature_order)
        

        logger.info(f"Run {run_id} - Condition A: Phase discussion")
        full_discussion = []
        vote_proposals = 0
        mentions_graph = defaultdict(lambda: defaultdict(int))
        
        turn = 0
        vote_initiator = None
        turn_vote_proposed = None
        vote_threshold = max(1, int(len(agents) * self.config.vote_threshold_percent))
        
        while turn < self.config.max_discussion_turns:
            turn += 1
            

            turn_agents = self.randomize_speaking_order(agents)
            turn_order = [agent.internal_id for agent in turn_agents]
            speaking_orders.append(turn_order)

            logger.info(f"Turn {turn} - Speaking order: {[f'{a.public_uuid}({a.internal_id})' for a in turn_agents]}")

            for speaking_position, agent in enumerate(turn_agents, 1):
                message, propose_vote, action_type = await self.discussion_turn(
                    agent, full_discussion, candidatures, agents, turn, run_id, "A", speaking_position
                )

                formatted_message = f"[Turn {turn}, Position {speaking_position}] {agent.public_uuid}: {message}"
                full_discussion.append(formatted_message)
                

                mentions = self.count_mentions(message, agents)
                for target_id, count in mentions.items():
                    mentions_graph[agent.internal_id][target_id] += count

                    for a in agents:
                        if a.internal_id == target_id:
                            a.mentions_received += count
                
                if propose_vote:
                    vote_proposals += 1
                    if vote_initiator is None:
                        vote_initiator = agent.internal_id
                        turn_vote_proposed = turn
                

                if vote_proposals >= vote_threshold:
                    break
            
            if vote_proposals >= vote_threshold:
                break
        

        logger.info(f"Run {run_id} - Condition A: Final Vote")
        vote_agents = self.randomize_speaking_order(agents)
        vote_order = [agent.internal_id for agent in vote_agents]
        speaking_orders.append(vote_order)
        
        vote_distribution = defaultdict(int)
        elected_leader = None
        abstentions = 0
        
        for speaking_position, agent in enumerate(vote_agents, 1):
            vote = await self.get_final_vote(agent, agents, full_discussion, run_id, "A", speaking_position)
            
            if vote:
                vote_distribution[vote] += 1
            else:
                abstentions += 1
        
        if vote_distribution:
            max_votes = max(vote_distribution.values())
            winners = [agent_id for agent_id, votes in vote_distribution.items() 
                      if votes == max_votes]
            elected_leader = winners[0] if len(winners) == 1 else "ex-aequo"
        

        centrality_scores = self.calculate_centrality(dict(mentions_graph), agents)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        

        uuid_mapping = {agent.internal_id: agent.public_uuid for agent in agents}
        

        run_conversations = [entry for entry in self.global_conversation_log 
                           if entry.run_id == run_id and entry.condition == "A"]
        
        return RunResults(
            condition="A",
            run_id=run_id,
            agents=agents,
            vote_initiator=vote_initiator,
            turn_vote_proposed=turn_vote_proposed,
            elected_leader=elected_leader,
            vote_distribution=dict(vote_distribution),
            messages_per_agent={a.internal_id: a.messages_sent for a in agents},
            mentions_received={a.internal_id: a.mentions_received for a in agents},
            centrality_scores=centrality_scores,
            candidatures={a.public_uuid: a.candidature for a in agents},
            discussion_turns=turn,
            abstentions=abstentions,
            execution_time=execution_time,
            request_ids=request_ids,
            vote_proposals=vote_proposals,
            uuid_mapping=uuid_mapping,
            timestamp=timestamp,
            full_discussion=full_discussion.copy(),
            speaking_orders=speaking_orders,
            conversation_log=run_conversations
        )
    
    async def run_condition_b(self, run_id: int) -> RunResults:
        """Condition B: Direct vote without discussion (JSON) with randomized order"""
        start_time = datetime.now()
        timestamp = start_time.strftime('%Y-%m-%d %H:%M:%S')
        agents = self.create_agents(run_id)
        request_ids = []
        speaking_orders = []
        
        logger.info(f"Run {run_id} - Condition B: {timestamp}")
        logger.info(f"Agents: {[f'{a.public_uuid}({a.internal_id})' for a in agents]}")
        

        vote_agents = self.randomize_speaking_order(agents)
        vote_order = [agent.internal_id for agent in vote_agents]
        speaking_orders.append(vote_order)
        
        logger.info(f"Vote order: {[f'{a.public_uuid}({a.internal_id})' for a in vote_agents]}")
        
        vote_distribution = defaultdict(int)
        abstentions = 0
        
        for speaking_position, agent in enumerate(vote_agents, 1):
            vote = await self.traditional_vote(agent, agents, run_id, "B", speaking_position)
            
            if vote:
                vote_distribution[vote] += 1
            else:
                abstentions += 1
        

        elected_leader = None
        if vote_distribution:
            max_votes = max(vote_distribution.values())
            winners = [agent_id for agent_id, votes in vote_distribution.items() 
                      if votes == max_votes]
            elected_leader = winners[0] if len(winners) == 1 else "ex-aequo"
        
        execution_time = (datetime.now() - start_time).total_seconds()
        

        uuid_mapping = {agent.internal_id: agent.public_uuid for agent in agents}
        

        run_conversations = [entry for entry in self.global_conversation_log 
                           if entry.run_id == run_id and entry.condition == "B"]
        
        return RunResults(
            condition="B",
            run_id=run_id,
            agents=agents,
            vote_initiator=None,
            turn_vote_proposed=None,
            elected_leader=elected_leader,
            vote_distribution=dict(vote_distribution),
            messages_per_agent={a.internal_id: 0 for a in agents},
            mentions_received={a.internal_id: 0 for a in agents},
            centrality_scores={a.internal_id: 0.0 for a in agents},
            candidatures={},
            discussion_turns=0,
            abstentions=abstentions,
            execution_time=execution_time,
            request_ids=request_ids,
            vote_proposals=0,
            uuid_mapping=uuid_mapping,
            timestamp=timestamp,
            full_discussion=[],
            speaking_orders=speaking_orders,
            conversation_log=run_conversations
        )
    
    async def run_condition_c(self, run_id: int) -> RunResults:
        """Condition C: Leader chosen randomly with documented randomization order"""
        start_time = datetime.now()
        timestamp = start_time.strftime('%Y-%m-%d %H:%M:%S')
        agents = self.create_agents(run_id)

        logger.info(f"Run {run_id} - Condition C: {timestamp}")
        logger.info(f"Agents: {[f'{a.public_uuid}({a.internal_id})' for a in agents]}")
        

        random_agents = self.randomize_speaking_order(agents)
        speaking_orders = [[agent.internal_id for agent in random_agents]]
        

        elected_leader = random.choice(agents).internal_id
        
        execution_time = (datetime.now() - start_time).total_seconds()
        

        uuid_mapping = {agent.internal_id: agent.public_uuid for agent in agents}
        

        run_conversations = []
        
        return RunResults(
            condition="C",
            run_id=run_id,
            agents=agents,
            vote_initiator=None,
            turn_vote_proposed=None,
            elected_leader=elected_leader,
            vote_distribution={elected_leader: len(agents)},
            messages_per_agent={a.internal_id: 0 for a in agents},
            mentions_received={a.internal_id: 0 for a in agents},
            centrality_scores={a.internal_id: 0.0 for a in agents},
            candidatures={},
            discussion_turns=0,
            abstentions=0,
            execution_time=execution_time,
            request_ids=[],
            vote_proposals=0,
            uuid_mapping=uuid_mapping,
            timestamp=timestamp,
            full_discussion=[],
            speaking_orders=speaking_orders,
            conversation_log=run_conversations
        )
    
    async def run_experiment(self, condition: str, num_runs: int) -> List[RunResults]:
        """Launch a series of runs for a given condition"""
        results = []
        
        for run_id in range(num_runs):
            logger.info(f"Starting run {run_id + 1}/{num_runs} - Condition {condition}")

            try:
                if condition == "A":
                    result = await self.run_condition_a(run_id)
                elif condition == "B":
                    result = await self.run_condition_b(run_id)
                elif condition == "C":
                    result = await self.run_condition_c(run_id)
                else:
                    raise ValueError(f"Unknown condition: {condition}")

                results.append(result)
                

                elected_uuid = "None"
                if result.elected_leader and result.uuid_mapping:
                    elected_uuid = result.uuid_mapping.get(result.elected_leader, "Unknown")
                
                logger.info(f"Run {run_id + 1} finished - Elected leader: {elected_uuid} "
                           f"(timestamp: {result.timestamp})")
                

                if result.speaking_orders:
                    logger.info(f"Number of speaking orders recorded: {len(result.speaking_orders)}")


                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error in run {run_id + 1}: {e}")
                continue
        
        return results
    
    def save_results(self, results: List[RunResults], output_dir: Path):
        """Save results with extended information and full conversations"""
        output_dir.mkdir(exist_ok=True)
        

        json_data = []
        for result in results:
            result_dict = asdict(result)

            result_dict['config'] = {
                'model': self.config.model,
                'vote_threshold_percent': self.config.vote_threshold_percent,
                'temperature_range': self.config.temperature_range,
                'randomize_speaking_order': self.config.randomize_speaking_order,
                'save_all_conversations': self.config.save_all_conversations
            }
            json_data.append(result_dict)
            
        with open(output_dir / "results_detailed.json", "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        

        self.save_conversation_log(output_dir)
        

        import csv
        with open(output_dir / "summary_extended.csv", "w", newline="", encoding="utf-8") as f:
            if results:
                fieldnames = [
                    "condition", "run_id", "timestamp", "elected_leader_internal", "elected_leader_uuid",
                    "vote_initiator_internal", "vote_initiator_uuid", "turn_vote_proposed", 
                    "discussion_turns", "abstentions", "execution_time", "vote_proposals", 
                    "total_messages", "unique_leaders_mentioned", "avg_centrality_score",
                    "num_speaking_orders", "total_conversations_logged", "randomization_active"
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:

                    total_messages = sum(result.messages_per_agent.values())
                    unique_leaders = len(set(result.vote_distribution.keys())) if result.vote_distribution else 0
                    avg_centrality = np.mean(list(result.centrality_scores.values())) if result.centrality_scores else 0
                    

                    elected_leader_uuid = ""
                    vote_initiator_uuid = ""
                    if result.uuid_mapping:
                        if result.elected_leader:
                            elected_leader_uuid = result.uuid_mapping.get(result.elected_leader, "")
                        if result.vote_initiator:
                            vote_initiator_uuid = result.uuid_mapping.get(result.vote_initiator, "")
                    

                    num_speaking_orders = len(result.speaking_orders) if result.speaking_orders else 0
                    num_conversations = len(result.conversation_log) if result.conversation_log else 0
                    
                    writer.writerow({
                        "condition": result.condition,
                        "run_id": result.run_id,
                        "timestamp": result.timestamp,
                        "elected_leader_internal": result.elected_leader,
                        "elected_leader_uuid": elected_leader_uuid,
                        "vote_initiator_internal": result.vote_initiator,
                        "vote_initiator_uuid": vote_initiator_uuid,
                        "turn_vote_proposed": result.turn_vote_proposed,
                        "discussion_turns": result.discussion_turns,
                        "abstentions": result.abstentions,
                        "execution_time": result.execution_time,
                        "vote_proposals": result.vote_proposals,
                        "total_messages": total_messages,
                        "unique_leaders_mentioned": unique_leaders,
                        "avg_centrality_score": avg_centrality,
                        "num_speaking_orders": num_speaking_orders,
                        "total_conversations_logged": num_conversations,
                        "randomization_active": self.config.randomize_speaking_order
                    })
        

        self.save_condition_analysis(results, output_dir)
        

        self.save_uuid_analysis(results, output_dir)
        

        self.save_speaking_order_analysis(results, output_dir)
        
        logger.info(f"Résultats sauvegardés dans {output_dir}")
    
    def save_conversation_log(self, output_dir: Path):
        """Sauvegarde le log complet des conversations LLM"""
        if not self.global_conversation_log:
            logger.warning("Aucune conversation à sauvegarder")
            return
        

        conversations_data = []
        for entry in self.global_conversation_log:
            entry_dict = asdict(entry)
            conversations_data.append(entry_dict)
        
        with open(output_dir / "all_conversations.json", "w", encoding="utf-8") as f:
            json.dump(conversations_data, f, indent=2, ensure_ascii=False)
        

        import csv
        with open(output_dir / "conversations_summary.csv", "w", newline="", encoding="utf-8") as f:
            if conversations_data:
                fieldnames = [
                    "timestamp", "run_id", "condition", "phase", "turn", 
                    "agent_internal_id", "agent_public_uuid", "temperature",
                    "speaking_order", "request_id", "api_model",
                    "prompt_length", "response_length", "parsing_success"
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for entry in self.global_conversation_log:
                    writer.writerow({
                        "timestamp": entry.timestamp,
                        "run_id": entry.run_id,
                        "condition": entry.condition,
                        "phase": entry.phase,
                        "turn": entry.turn,
                        "agent_internal_id": entry.agent_internal_id,
                        "agent_public_uuid": entry.agent_public_uuid,
                        "temperature": entry.temperature,
                        "speaking_order": entry.speaking_order,
                        "request_id": entry.request_id,
                        "api_model": entry.api_model,
                        "prompt_length": len(entry.prompt_sent) if entry.prompt_sent else 0,
                        "response_length": len(entry.response_received) if entry.response_received else 0,
                        "parsing_success": entry.parsed_action is not None
                    })
        

        conversations_by_run = defaultdict(list)
        for entry in self.global_conversation_log:
            conversations_by_run[f"{entry.condition}_{entry.run_id}"].append(entry)
        
        conversations_dir = output_dir / "conversations_by_run"
        conversations_dir.mkdir(exist_ok=True)
        
        for run_key, conversations in conversations_by_run.items():
            with open(conversations_dir / f"{run_key}.json", "w", encoding="utf-8") as f:
                json.dump([asdict(conv) for conv in conversations], f, indent=2, ensure_ascii=False)
        
        logger.info(f"Sauvegardé {len(self.global_conversation_log)} conversations LLM")
    
    def save_speaking_order_analysis(self, results: List[RunResults], output_dir: Path):
        """Save speaking order analysis results"""
        analysis = {
            "randomization_enabled": self.config.randomize_speaking_order,
            "total_speaking_orders": 0,
            "speaking_patterns": {},
            "first_speaker_distribution": defaultdict(int),
            "position_consistency": {},
            "randomization_quality": {}
        }
        
        all_orders = []
        
        for result in results:
            if result.speaking_orders:
                analysis["total_speaking_orders"] += len(result.speaking_orders)
                all_orders.extend(result.speaking_orders)
                

                for order in result.speaking_orders:
                    if order:
                        first_speaker = order[0]
                        analysis["first_speaker_distribution"][first_speaker] += 1
        

        if all_orders and len(set(tuple(order) for order in all_orders)) > 1:
            unique_orders = len(set(tuple(order) for order in all_orders))
            total_orders = len(all_orders)
            analysis["randomization_quality"] = {
                "unique_orders": unique_orders,
                "total_orders": total_orders,
                "diversity_ratio": unique_orders / total_orders if total_orders > 0 else 0,
                "expected_max_diversity": min(720, total_orders) if len(all_orders[0]) <= 6 else total_orders
            }
        

        position_counts = defaultdict(lambda: defaultdict(int))
        for order in all_orders:
            for position, agent_id in enumerate(order):
                position_counts[agent_id][position] += 1
        
        for agent_id, positions in position_counts.items():
            total_appearances = sum(positions.values())
            if total_appearances > 0:

                entropy = -sum((count/total_appearances) * np.log2(count/total_appearances) 
                              for count in positions.values() if count > 0)
                max_entropy = np.log2(len(positions)) if len(positions) > 0 else 0
                analysis["position_consistency"][agent_id] = {
                    "entropy": entropy,
                    "max_entropy": max_entropy,
                    "randomization_score": entropy / max_entropy if max_entropy > 0 else 0
                }
        
        with open(output_dir / "speaking_order_analysis.json", "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
    
    def save_uuid_analysis(self, results: List[RunResults], output_dir: Path):
        """Save analysis specific to UUI usage"""
        uuid_analysis = {
            "total_uuids_generated": 0,
            "unique_uuids_per_run": [],
            "uuid_reuse_across_runs": 0,
            "uuid_pattern_analysis": {},
            "uuid_version": "UUID4 (vraiment aléatoire)",
            "randomization_settings": {
                "speaking_order_randomized": self.config.randomize_speaking_order,
                "conversations_logged": self.config.save_all_conversations
            }
        }
        
        all_uuids = set()
        
        for result in results:
            run_uuids = set()
            if result.uuid_mapping:
                for internal_id, public_uuid in result.uuid_mapping.items():
                    all_uuids.add(public_uuid)
                    run_uuids.add(public_uuid)
            
            uuid_analysis["unique_uuids_per_run"].append(len(run_uuids))
        
        uuid_analysis["total_uuids_generated"] = len(all_uuids)
        

        total_uuids_expected = len(results) * self.config.num_agents
        uuid_analysis["uuid_reuse_across_runs"] = total_uuids_expected - len(all_uuids)
        

        uuid_prefixes = defaultdict(int)
        uuid_versions = defaultdict(int)
        for uuid_str in all_uuids:
            if len(uuid_str) >= 8:
                prefix = uuid_str[:2]
                uuid_prefixes[prefix] += 1
            

            if len(uuid_str) >= 15 and uuid_str[14] in '0123456789abcdef':
                uuid_versions[uuid_str[14]] += 1
        
        uuid_analysis["uuid_pattern_analysis"] = {
            "prefix_distribution": dict(uuid_prefixes),
            "version_distribution": dict(uuid_versions),
            "entropy_check": len(uuid_prefixes) / min(256, len(all_uuids)) if all_uuids else 0,
            "uuid4_compliance": uuid_versions.get('4', 0) / len(all_uuids) if all_uuids else 0
        }
        
        with open(output_dir / "uuid_analysis.json", "w", encoding="utf-8") as f:
            json.dump(uuid_analysis, f, indent=2, ensure_ascii=False)
    
    def save_condition_analysis(self, results: List[RunResults], output_dir: Path):
        """Save comparative analysis of conditions with new metrics"""
        analysis = {}
        
        for condition in ["A", "B", "C"]:
            condition_results = [r for r in results if r.condition == condition]
            if not condition_results:
                continue
            

            leaders = [r.elected_leader for r in condition_results if r.elected_leader and r.elected_leader != "ex-aequo"]
            leader_distribution = Counter(leaders)
            

            timestamps = [datetime.strptime(r.timestamp, '%Y-%m-%d %H:%M:%S') for r in condition_results if r.timestamp]
            time_span = (max(timestamps) - min(timestamps)).total_seconds() / 60 if len(timestamps) > 1 else 0
            

            total_conversations = sum(len(r.conversation_log) if r.conversation_log else 0 for r in condition_results)
            total_speaking_orders = sum(len(r.speaking_orders) if r.speaking_orders else 0 for r in condition_results)
            

            all_orders = []
            for r in condition_results:
                if r.speaking_orders:
                    all_orders.extend(r.speaking_orders)
            
            unique_orders = len(set(tuple(order) for order in all_orders)) if all_orders else 0
            order_diversity = unique_orders / len(all_orders) if all_orders else 0
            
            analysis[condition] = {
                "total_runs": len(condition_results),
                "successful_elections": len(leaders),
                "tie_rate": len([r for r in condition_results if r.elected_leader == "ex-aequo"]) / len(condition_results),
                "abstention_rate": np.mean([r.abstentions for r in condition_results]) / self.config.num_agents,
                "avg_discussion_turns": np.mean([r.discussion_turns for r in condition_results]),
                "avg_execution_time": np.mean([r.execution_time for r in condition_results]),
                "leader_concentration": max(leader_distribution.values()) / len(leaders) if leaders else 0,
                "unique_leaders": len(leader_distribution),
                "avg_vote_proposals": np.mean([r.vote_proposals for r in condition_results]),
                "uuid_diversity": len(set(uuid for r in condition_results for uuid in (r.uuid_mapping.values() if r.uuid_mapping else []))),
                "time_span_minutes": time_span,
                "avg_run_duration": np.mean([r.execution_time for r in condition_results]),

                "total_llm_conversations": total_conversations,
                "avg_conversations_per_run": total_conversations / len(condition_results) if condition_results else 0,
                "total_speaking_orders_recorded": total_speaking_orders,
                "speaking_order_diversity": order_diversity,
                "randomization_enabled": self.config.randomize_speaking_order
            }
        

        with open(output_dir / "condition_analysis.json", "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        

        with open(output_dir / "analysis_report.txt", "w", encoding="utf-8") as f:
            f.write("COMPARATIVE ANALYSIS OF CONDITIONS - VERSION WITH CONVERSATIONS & RANDOMIZATION\n")
            f.write("=" * 80 + "\n\n")
            f.write("NEW FEATURES:\n")
            f.write("- Complete saving of LLM conversations (prompts + responses)\n")
            f.write("- Randomized speaking order for each round\n")
            f.write("- Detailed logging with speaking positions\n")
            f.write("- Analysis of randomization quality\n\n")
            f.write("Note: Agents use truly random UUIDs v4 as public identifiers\n")
            f.write("to eliminate name-related biases. Communication via JSON messages.\n")
            f.write("IMPORTANT: Removal of seed logic as it is not reproducible with LLMs.\n\n")

            for condition, stats in analysis.items():
                f.write(f"CONDITION {condition}:\n")
                f.write(f"  Successful Runs: {stats['successful_elections']}/{stats['total_runs']}\n")
                f.write(f"  Tie Rate: {stats['tie_rate']:.2%}\n")
                f.write(f"  Abstention Rate: {stats['abstention_rate']:.2%}\n")
                f.write(f"  Average Discussion Turns: {stats['avg_discussion_turns']:.1f}\n")
                f.write(f"  Average Run Duration: {stats['avg_run_duration']:.1f}s\n")
                f.write(f"  Total Experience Duration: {stats['time_span_minutes']:.1f} minutes\n")
                f.write(f"  Leadership Concentration: {stats['leader_concentration']:.2%}\n")
                f.write(f"  Unique Leaders: {stats['unique_leaders']}\n")
                f.write(f"  Average Vote Proposals: {stats['avg_vote_proposals']:.1f}\n")
                f.write(f"  UUID Diversity: {stats['uuid_diversity']} unique UUIDs\n")
                f.write(f"  NEW METRICS:\n")
                f.write(f"    Total LLM Conversations: {stats['total_llm_conversations']}\n")
                f.write(f"    Conversations per run (avg): {stats['avg_conversations_per_run']:.1f}\n")
                f.write(f"    Saved speaking order: {stats['total_speaking_orders_recorded']}\n")
                f.write(f"    Speaking order diversity: {stats['speaking_order_diversity']:.2%}\n")
                f.write(f"    Randomization enabled: {stats['randomization_enabled']}\n")
                f.write("\n")

def analyze_results(results: List[RunResults]) -> Dict[str, Any]:
    """Analyze results to detect patterns with new metrics"""
    analysis = {
        "total_runs": len(results),
        "uuid_diversity": set(),
        "leadership_patterns": defaultdict(list),
        "discussion_effectiveness": {},
        "json_parsing_success": {},
        "uuid_bias_analysis": {},
        "temporal_analysis": {},
        "conversation_analysis": {},
        "speaking_order_analysis": {}
    }
    

    for result in results:
        if result.uuid_mapping:
            for uuid_str in result.uuid_mapping.values():
                analysis["uuid_diversity"].add(uuid_str)
    
    analysis["uuid_diversity"] = len(analysis["uuid_diversity"])
    

    for result in results:
        condition = result.condition
        analysis["leadership_patterns"][condition].append({
            "elected_leader": result.elected_leader,
            "vote_initiator": result.vote_initiator,
            "turns_to_decision": result.turn_vote_proposed,
            "discussion_length": result.discussion_turns,
            "timestamp": result.timestamp,
            "speaking_orders_count": len(result.speaking_orders) if result.speaking_orders else 0
        })
    

    for condition in ["A", "B", "C"]:
        condition_results = [r for r in results if r.condition == condition]
        if condition_results:
            analysis["discussion_effectiveness"][condition] = {
                "avg_turns": np.mean([r.discussion_turns for r in condition_results]),
                "decision_speed": np.mean([r.turn_vote_proposed or 0 for r in condition_results]),
                "consensus_rate": len([r for r in condition_results if r.elected_leader and r.elected_leader != "ex-aequo"]) / len(condition_results)
            }
    

    for condition in ["A", "B"]:
        condition_results = [r for r in results if r.condition == condition]
        if condition_results:
            analysis["json_parsing_success"][condition] = {
                "avg_vote_proposals": np.mean([r.vote_proposals for r in condition_results]),
                "successful_votes": len([r for r in condition_results if r.vote_distribution]) / len(condition_results)
            }
    

    uuid_leadership_analysis = defaultdict(int)
    for result in results:
        if result.elected_leader and result.uuid_mapping and result.elected_leader != "ex-aequo":
            elected_uuid = result.uuid_mapping[result.elected_leader]

            first_char = elected_uuid[0] if elected_uuid else "unknown"
            uuid_leadership_analysis[first_char] += 1
    
    analysis["uuid_bias_analysis"] = {
        "first_char_distribution": dict(uuid_leadership_analysis),
        "potential_bias_detected": max(uuid_leadership_analysis.values()) / sum(uuid_leadership_analysis.values()) > 0.2 if uuid_leadership_analysis else False,
        "uuid_randomness_quality": "UUID4 utilisé - vraiment aléatoire"
    }
    

    if results and results[0].timestamp:
        timestamps = []
        for result in results:
            try:
                ts = datetime.strptime(result.timestamp, '%Y-%m-%d %H:%M:%S')
                timestamps.append(ts)
            except:
                continue
        
        if timestamps:
            total_duration = (max(timestamps) - min(timestamps)).total_seconds()
            analysis["temporal_analysis"] = {
                "experiment_start": min(timestamps).isoformat(),
                "experiment_end": max(timestamps).isoformat(),
                "total_duration_minutes": total_duration / 60,
                "runs_per_minute": len(results) / max(1, total_duration / 60)
            }
    

    total_conversations = sum(len(r.conversation_log) if r.conversation_log else 0 for r in results)
    conversations_by_phase = defaultdict(int)
    conversations_by_condition = defaultdict(int)
    
    for result in results:
        if result.conversation_log:
            for conv in result.conversation_log:
                conversations_by_phase[conv.phase] += 1
                conversations_by_condition[result.condition] += 1
    
    analysis["conversation_analysis"] = {
        "total_llm_conversations": total_conversations,
        "avg_conversations_per_run": total_conversations / len(results) if results else 0,
        "conversations_by_phase": dict(conversations_by_phase),
        "conversations_by_condition": dict(conversations_by_condition),
        "conversation_logging_enabled": any(r.conversation_log for r in results)
    }
    

    total_orders = sum(len(r.speaking_orders) if r.speaking_orders else 0 for r in results)
    all_orders = []
    for result in results:
        if result.speaking_orders:
            all_orders.extend(result.speaking_orders)
    
    unique_orders = len(set(tuple(order) for order in all_orders)) if all_orders else 0
    
    analysis["speaking_order_analysis"] = {
        "total_speaking_orders": total_orders,
        "unique_order_patterns": unique_orders,
        "order_diversity_ratio": unique_orders / total_orders if total_orders > 0 else 0,
        "randomization_quality": "High" if unique_orders / max(1, total_orders) > 0.8 else "Medium" if unique_orders / max(1, total_orders) > 0.5 else "Low"
    }
    
    return analysis

def generate_example_uuids(count: int = 10) -> List[str]:
    """Generate example UUIDs v4"""
    return [str(uuid.uuid4()) for _ in range(count)]

def test_json_parsing():
    """Test JSON parsing function"""
    simulator = GPTLeaderElectionSimulator(SimulationConfig(), "dummy_key", "dummy_base")
    
    test_cases = [
        '{"type": "vote", "target": "12345678-1234-4567-8901-123456789abc"}',
        '{"type": "discuss", "message": "I think we should discuss more"}',
        '{"type": "propose_vote", "candidate": "87654321-4321-4567-8901-cba987654321", "justification": "He is competent"}',
        'I think that {"type": "vote", "target": "12345678-1234-4567-8901-123456789abc"} would be good',
        'Here is my response: {"type": "discuss", "message": "let\'s continue the discussion"}',
        'Malformed response without valid JSON',
        '{"type": "vote" "target": "uuid-missing-comma"}',
    ]
    
    print("Test of JSON Parsing:")
    print("-" * 40)
    for i, test in enumerate(test_cases):
        result = simulator.parse_json_response(test)
        print(f"Test {i+1}: {result}")
        print(f"  Input: {test[:50]}...")
        print()

async def main():
    """Main function to run the experiment with conversations and randomization"""
    models_to_test = [
        {
            "model": "openai/gpt-5-mini",
            "description": "GPT-5 Mini with API Responses",
            "temperature_range": (0.65, 0.75)
        },
        {
            "model": "openai/gpt-4o-mini", 
            "description": "GPT-4o Mini with temperature support",
            "temperature_range": (0.65, 0.75)
        },
        {
            "model": "meta-llama/llama-3.3-70b-instruct",
            "description": "Llama 3.3 70B with temperature support",
            "temperature_range": (0.65, 0.75)
        }
    ]

    selected_model = models_to_test[2]
    conditions_to_test = ["A", "B", "C"]

    config = SimulationConfig(
        model=selected_model["model"],
        num_agents=15,
        num_runs=20,
        vote_threshold_percent=0.5,
        max_discussion_turns=15,
        randomize_speaking_order=True,
        save_all_conversations=True,
        temperature_range=selected_model["temperature_range"]
    )
    

    api_key = ""
    api_base = "https://openrouter.ai/api/v1"
    simulator = GPTLeaderElectionSimulator(config, api_key, api_base=api_base)

    logger.info("Examples of UUIDs v4 that will be used as public identifiers:")
    for example_uuid in generate_example_uuids(4):
        logger.info(f"  {example_uuid}")
    logger.info("")

    logger.info("NEW FEATURES ENABLED:")
    logger.info(f"  - Randomization of speaking order: {config.randomize_speaking_order}")
    logger.info(f"  - Saving LLM conversations: {config.save_all_conversations}")
    logger.info("")
    

    all_results = []
    
    
    for condition in conditions_to_test:
        logger.info(f"=" * 60)
        logger.info(f"STARTING CONDITION {condition}")
        logger.info(f"=" * 60)
        
        condition_results = await simulator.run_experiment(condition, config.num_runs)
        all_results.extend(condition_results)
        

        leaders = [r.elected_leader for r in condition_results if r.elected_leader and r.elected_leader != "ex-aequo"]
        total_conversations = sum(len(r.conversation_log) if r.conversation_log else 0 for r in condition_results)
        total_speaking_orders = sum(len(r.speaking_orders) if r.speaking_orders else 0 for r in condition_results)

        logger.info(f"Condition {condition} terminated:")
        logger.info(f"  - {len(condition_results)} runs completed")
        logger.info(f"  - {len(set(leaders))} unique leaders (internal IDs)")
        logger.info(f"  - {len([r for r in condition_results if r.elected_leader == 'ex-aequo'])} ties")
        logger.info(f"  - {total_conversations} recorded LLM conversations")
        logger.info(f"  - {total_speaking_orders} documented speaking orders")
        
        if condition == "A":
            avg_turns = np.mean([r.discussion_turns for r in condition_results])
            avg_proposals = np.mean([r.vote_proposals for r in condition_results])
            logger.info(f"  - {avg_turns:.1f} discussion turns on average")
            logger.info(f"  - {avg_proposals:.1f} vote proposals on average")

        sample_uuids = []
        for result in condition_results[:2]:
            if result.uuid_mapping:
                sample_uuids.extend(list(result.uuid_mapping.values())[:3])
        if sample_uuids:
            logger.info(f"  - Examples of UUIDs used: {', '.join(sample_uuids[:3])}")

        if condition_results and condition_results[0].speaking_orders:
            first_order = condition_results[0].speaking_orders[0]
            logger.info(f"  - Example of speaking order: {first_order}")
    

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path("results") / f"uuid_conversations_randomized_{timestamp}"
    

    simulator.save_results(all_results, output_dir)
    

    detailed_analysis = analyze_results(all_results)
    
    with open(output_dir / "detailed_analysis.json", "w", encoding="utf-8") as f:
        json.dump(detailed_analysis, f, indent=2, ensure_ascii=False, default=str)
    

    logger.info(f"\n" + "=" * 80)
    logger.info(f"EXPERIMENT WITH CONVERSATIONS & RANDOMIZATION TERMINATED")
    logger.info(f"=" * 80)
    logger.info(f"Total runs: {len(all_results)}")
    logger.info(f"Uniques UUIDs generated: {detailed_analysis['uuid_diversity']}")
    logger.info(f"Conversations LLM recorded: {detailed_analysis['conversation_analysis']['total_llm_conversations']}")
    logger.info(f"Unique speaking orders: {detailed_analysis['speaking_order_analysis']['unique_order_patterns']}")
    logger.info(f"Randomization quality: {detailed_analysis['speaking_order_analysis']['randomization_quality']}")


    if detailed_analysis.get('uuid_bias_analysis', {}).get('potential_bias_detected'):
        logger.warning("⚠️  Potential bias detected in UUID prefix distribution")
    else:
        logger.info("✅ UUID distribution appears balanced (UUID4 truly random)")
    

    if 'temporal_analysis' in detailed_analysis:
        temporal = detailed_analysis['temporal_analysis']
        logger.info(f"Durée totale expérience: {temporal.get('total_duration_minutes', 0):.1f} minutes")
        logger.info(f"Vitesse: {temporal.get('runs_per_minute', 0):.1f} runs/minute")
    
    for condition in conditions_to_test:
        condition_results = [r for r in all_results if r.condition == condition]
        if condition_results:
            success_rate = len([r for r in condition_results if r.elected_leader and r.elected_leader != "ex-aequo"]) / len(condition_results)
            logger.info(f"Condition {condition}: {success_rate:.1%} election success rate")

    logger.info(f"\nFiles saved in: {output_dir}")
    logger.info("NEW FILES GENERATED:")
    logger.info("  - all_conversations.json (all LLM conversations)")
    logger.info("  - conversations_summary.csv (summary of conversations)")
    logger.info("  - conversations_by_run/ (conversations by individual run)")
    logger.info("  - speaking_order_analysis.json (analysis of randomization)")
    logger.info("Note: UUIDs v4 truly random, speaking order randomized")

def run_single_test_with_conversations():
    """Function to quickly test with new features"""
    async def test():
        model = "gpt-5-mini"
        config = SimulationConfig(
            model=model,
            num_agents=4,
            num_runs=1,
            vote_threshold_percent=0.5,
            randomize_speaking_order=True,
            save_all_conversations=True
        )
        api_key = ""
        api_base = ""

        if model.startswith("gpt"):
            api_key = ""
        else:
            api_key = ""

        simulator = GPTLeaderElectionSimulator(config, api_key=api_key)


        results = await simulator.run_experiment("A", 1)
        
        for i, result in enumerate(results):
            print(f"\nRUN {i+1} (timestamp: {result.timestamp}):")
            if result.uuid_mapping:
                print(f"  Public UUIDs: {list(result.uuid_mapping.values())}")
                elected_uuid = result.uuid_mapping.get(result.elected_leader, "Unknown") if result.elected_leader else "None"
                print(f"  Elected leader (UUID): {elected_uuid}")
            print(f"  Elected leader (internal ID): {result.elected_leader}")
            print(f"  Discussion turns: {result.discussion_turns}")
            print(f"  Vote proposals: {result.vote_proposals}")
            print(f"  Recorded conversations: {len(result.conversation_log) if result.conversation_log else 0}")
            print(f"  Speaking orders: {len(result.speaking_orders) if result.speaking_orders else 0}")


            if result.conversation_log:
                print(f"  Example conversation:")
                for conv in result.conversation_log[:2]:
                    print(f"    {conv.phase} - {conv.agent_public_uuid}: {len(conv.prompt_sent)} chars prompt, {len(conv.response_received) if conv.response_received else 0} chars response")
    
    asyncio.run(test())

def demonstrate_randomization():
    """Demonstrates the randomization of speaking orders"""
    print("Demonstration of speaking order randomization:")
    print("-" * 60)
    

    agents = [f"agent-{i+1}" for i in range(4)]
    print(f"Agents: {agents}")
    print("\nGenerated speaking orders (10 examples):")

    for i in range(10):
        shuffled = agents.copy()
        random.shuffle(shuffled)
        print(f"Run {i+1}: {shuffled}")

    print("\nNote: Every run uses a different order (truly random)")

def analyze_conversation_patterns(conversation_log: List[ConversationEntry]):
    """Analyze patterns in LLM conversations"""
    analysis = {
        "total_conversations": len(conversation_log),
        "by_phase": defaultdict(int),
        "by_agent": defaultdict(int),
        "by_turn": defaultdict(int),
        "parsing_success_rate": 0,
        "avg_prompt_length": 0,
        "avg_response_length": 0
    }
    
    if not conversation_log:
        return analysis
    
    total_prompt_length = 0
    total_response_length = 0
    successful_parses = 0
    
    for conv in conversation_log:
        analysis["by_phase"][conv.phase] += 1
        analysis["by_agent"][conv.agent_internal_id] += 1
        analysis["by_turn"][conv.turn] += 1
        
        if conv.parsed_action is not None:
            successful_parses += 1
        
        if conv.prompt_sent:
            total_prompt_length += len(conv.prompt_sent)
        if conv.response_received:
            total_response_length += len(conv.response_received)
    
    analysis["parsing_success_rate"] = successful_parses / len(conversation_log)
    analysis["avg_prompt_length"] = total_prompt_length / len(conversation_log)
    analysis["avg_response_length"] = total_response_length / len(conversation_log)
    
    return analysis

if __name__ == "__main__":

    test_json_parsing()
    print("\n" + "="*60 + "\n")
    

    print("DDemonstration of speaking order randomization:")
    print("-" * 60)
    demonstrate_randomization()
    print("\n" + "="*60 + "\n")
    

    demonstrate_randomization()
    print("\n" + "="*60 + "\n")
    

    asyncio.run(main())