# 🌱 AgriVision — AI Crop Health Assistant

AgriVision is an end-to-end agricultural advisory system that turns a simple crop-image classifier into an interactive AI farming assistant. Upload a photo of a plant leaf and AgriVision detects the disease, grounds its advice in your local weather, and explains everything in plain, farmer-friendly language — powered by a **TensorFlow CNN** + **LangGraph** orchestration + **Mistral AI** (AgriExpert).

## ✨ Features

- 🔬 **Disease Detection** — A Keras (EfficientNet-based) CNN classifies leaf/crop images across **29 crop–disease classes** (apple, tomato, potato, grape, corn, peach, cherry, bell pepper, strawberry and more).
- 🧠 **LLM Explanations** — Mistral's *AgriExpert* agent turns the raw prediction into clear, actionable advice: what causes it, symptoms to verify, how it spreads, and organic/chemical treatment options.
- 🌦️ **Weather-Aware Advice** — When you provide a location, live weather (temperature, humidity, wind) is fetched from OpenWeather and used to ground the disease management guidance.
- 🗺️ **Regional Suitability** — Assesses whether your crop is likely to survive and thrive in your region, plus estimated yield.
- 💬 **Follow-up Chat** — Ask follow-up questions about a detected disease; answers stay grounded in the latest analysis.
- 📊 **Farm Dashboard** — A live overview built from real session data: crop checks, average confidence, and diseases detected.
- 🕘 **History & Settings** — Review past analyses and adjust app settings.

## 🏗️ Architecture

```
Image (leaf/crop)
      │
      ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   classify       │────▶│   weather        │────▶│   regional       │
│ (Keras CNN)      │     │ (OpenWeather)    │     │ (Mistral agent)  │
└──────────────────┘     └──────────────────┘     └───────┬──────────┘
                                                           │
                                             confidence > 50%?
                                                           │
                                          ┌────────┐   yes  ▼
                                          │  end   │◀────────────┐
                                          └────────┘              │
                                                          ┌───────▼──────┐
                                                          │   explain    │
                                                          │ (AgriExpert) │
                                                          └──────┬───────┘
                                                                 ▼
                                                          Explanation +
                                                          weather context
```

The pipeline is a **LangGraph** state machine (`worker/agents/graph.py`):

1. **Classify** — CNN predicts disease + confidence from a 256×256 image.
2. **Weather** — Fetch live weather for the farmer-provided location (never breaks the flow on failure).
3. **Regional** — Mistral assesses crop survival/nativity/yield for the region.
4. **Route** — If confidence > 50%, generate a full explanation; otherwise end early to avoid hallucinating on low-confidence predictions.
5. **Explain** — Mistral's AgriExpert produces a farmer-friendly, weather-grounded report.

### Project Structure

```
AgriVision-MVP/
├── frontend/           # Streamlit UI (Chat, Dashboard, History, Settings)
│   ├── app.py          # Entry point + page routing
│   ├── backend.py      # Thin wrappers around the core workflow
│   ├── state.py        # Session-state management
│   └── ui/             # Page & component renderers
├── model/              # Trained Keras CNN model
│   ├── crop_disease_model.keras
│   ├── class_names.json
│   ├── predictor.py    # Loads model & predicts from an image
│   └── main.py         # CLI for single-image prediction
├── worker/             # Core orchestration & services
│   ├── model.py        # Model API wrapper used by the graph
│   ├── weather.py      # OpenWeather client
│   ├── agents/         # LangGraph workflow + Mistral explainer
│   └── prompts/        # System prompts (AgriExpert)
└── tests/              # Unit tests (graph & weather, CNN mocked)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [Mistral AI API key](https://console.mistral.ai/) (required)
- [OpenWeather API key](https://openweathermap.org/api) (optional, for weather)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Sumukh-Saxena/AgriVision-MVP.git
cd AgriVision-MVP

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables (copy the sample and fill in your keys)
#    See env_sample.txt
```

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mistral_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
```

### Run the App

```bash
streamlit run frontend/app.py
```

### Run the CLI Predictor

```bash
python model/main.py path/to/leaf.jpg
```

### Run Tests

```bash
python -m unittest discover -s tests
```

## 🔑 Environment Variables

| Variable              | Required | Description                                    |
| --------------------- | -------- | ---------------------------------------------- |
| `MISTRAL_API_KEY`     | ✅ Yes   | Mistral AI key for the AgriExpert LLM agent    |
| `OPENWEATHER_API_KEY` | ⭕ No    | OpenWeather key for weather-aware advice       |

## 🧪 Supported Classes

AgriVision's model detects 29 crop–disease combinations across **Apple, Bell Pepper, Cherry, Corn (Maize), Grape, Peach, Potato, Strawberry, and Tomato**, including healthy plants for each crop.

## 🛡️ Disclaimer

AgriVision is a decision-support tool, not a replacement for professional agronomy advice. Always verify severe infestations with a local agricultural extension officer before applying treatments.

## 📄 License

This project is for educational/demo purposes. No license file is included — contact the maintainers for usage rights.
