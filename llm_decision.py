def generate_llm_suggestion(traffic_info):
    n = traffic_info["north"]
    s = traffic_info["south"]
    e = traffic_info["east"]
    w = traffic_info["west"]

    ns = n + s
    ew = e + w

    if ns > ew:
        return "👉 Prioritize the north-south road"
    elif ew > ns:
        return "👉 Allow right road to clear congestion"
    else:
        return "👉 Extend green signal equally"