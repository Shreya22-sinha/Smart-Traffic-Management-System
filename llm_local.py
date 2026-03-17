import random

# --- LLM Initialization Block ---
LLM_LOADED = False
try:
    from transformers import pipeline
    # Load a small text-generation model (1-time download)
    generator = pipeline("text-generation", model="distilgpt2")
    LLM_LOADED = True
except Exception as e:
    # This block catches the crash during initialization (the likely source of the 500 error)
    print(f"FATAL: LLM dependencies failed to load. Using basic summary. Error: {e}")
    LLM_LOADED = False
# --- End LLM Initialization Block ---


def analyze_traffic_offline(vehicle_count, light_status, wait_time):
    
    # 1. Guaranteed Fallback
    if not LLM_LOADED:
        status_text = "high congestion" if vehicle_count > 25 else "moderate flow"
        return f"LLM offline. System detected {vehicle_count} vehicles, indicating {status_text}. Recommendation: Maintain the current {light_status} light status to clear the backlog."

    # 2. LLM-based generation (if loaded)
    prompt = (
        f"Traffic summary:\n"
        f"- Vehicles detected: {vehicle_count}\n"
        f"- Light status: {light_status}\n"
        f"- Wait time: {wait_time} seconds.\n\n"
        f"Provide a short recommendation for improving the traffic flow."
    )

    try:
        # Use pad_token_id to prevent common generation errors
        output = generator(prompt, max_length=80, num_return_sequences=1, pad_token_id=generator.tokenizer.eos_token_id)[0]["generated_text"]
        # Clean up the output to only return the recommendation part
        return output.split("recommendation for improving the traffic flow.")[-1].strip()
    except Exception as e:
        # 3. Fallback if generation itself fails
        print(f"LLM generation failed: {e}. Returning basic summary.")
        status_text = "high congestion" if vehicle_count > 25 else "moderate flow"
        return f"LLM generation failed. System detected {vehicle_count} vehicles, indicating {status_text}. Recommendation: Temporarily override light to {light_status} for 30 seconds."