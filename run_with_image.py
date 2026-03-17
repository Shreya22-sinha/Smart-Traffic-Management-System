from dqn_agent import DQNAgent
from image_processor import extract_traffic_state
from llm_decision import generate_llm_suggestion

agent = DQNAgent()

image_path = "crossroad.jpg"

# Step 1: Extract traffic features
state, traffic_info = extract_traffic_state(image_path)

print("Traffic Counts:", traffic_info)
print("State:", state)

# Step 2: DQN chooses action
action = agent.choose_action(state)

actions = {
    0: "Give Green to North-South",
    1: "Give Green to East-West",
    2: "Keep Current Signal"
}

print("DQN Decision:", actions[action])

# Step 3: LLM Suggestion
llm_response = generate_llm_suggestion(traffic_info)
print("LLM Suggestion:", llm_response)