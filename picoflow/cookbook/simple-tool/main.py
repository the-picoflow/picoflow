import json
from datetime import date, timedelta

from picoflow.core import flow, State, llm


# ============ fake tool ============

def fake_weather_tomorrow(location: str) -> dict:
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    return {
        "location": location,
        "date": tomorrow,
        "condition": "Cloudy with occasional light rain",
        "temp_c": {"min": 7, "max": 12},
        "wind": "NE 4-5",
        "humidity": "75%",
        "tips": ["Bring a light rain jacket", "Wear layers", "Comfortable walking shoes"],
    }


# ============ flows ============

@flow
def get_weather(ctx: State) -> State:
    location = ctx.data.get("location", "SHANGHAI")
    weather = fake_weather_tomorrow(location)

    weather_json = json.dumps(weather, ensure_ascii=False)
    return ctx.update(evidence=ctx.evidence + [f"Tomorrow weather (JSON): {weather_json}"])


DSN = "llm+openai://api.siliconflow.cn/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B?api_key_env=OPENAI_API_KEY"

travel_advice = llm(
    prompt_template=(
        "You are a travel assistant.\n"
        "Write a practical 1-day travel plan in Chinese for {input}.\n"
        "Requirements:\n"
        "- What to wear\n"
        "- What to pack\n"
        "- Indoor/outdoor balance\n"
        "- 3 safety or comfort tips\n"
        "{evidence}\n"
    ),
    stream=False,
    llm_adapter=DSN,
    final=True,
)

pipeline = get_weather >> travel_advice

# ============ run ============

if __name__ == "__main__":
    ctx = State(
        input="Shanghai (SHANGHAI)",
        data={"location": "SHANGHAI"},
    )

    ctx = pipeline.run(ctx)

    print("EVIDENCE:")
    for e in ctx.evidence:
        print("-", e)

    print("\nOUTPUT:\n", ctx.output)
