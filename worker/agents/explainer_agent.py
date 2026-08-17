import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

from worker.prompts.get_prompt import get_disease_explainer_prompt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


class DiseaseExplainerAgent:
    """Explains a crop disease flagged by the CNN model using Mistral AI."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "mistral-small-latest",
        temperature: float = 0.3,
    ) -> None:
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "MISTRAL_API_KEY is not set. Provide api_key or set the environment variable."
            )

        self.llm = ChatMistralAI(
            model=model_name,
            api_key=self.api_key,
            temperature=temperature,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", get_disease_explainer_prompt()),
                ("human", "{input}"),
            ]
        )

        self.chain = self.prompt | self.llm | StrOutputParser()

    def explain(
        self,
        disease_name: str,
        confidence: float,
        weather: dict | None = None,
    ) -> str:
        """Generate a farmer-friendly explanation for a flagged disease.

        When weather data for the crop's location is available, it is included
        so the advice can be grounded in local conditions.
        """
        weather_block = ""
        if weather:
            weather_block = (
                "\n\nWeather at the crop's location:\n"
                f"- Location: {weather.get('city')}, {weather.get('country') or weather.get('location') or ''}\n"
                f"- Conditions: {weather.get('description') or 'n/a'}\n"
                f"- Temperature: {weather.get('temperature_c')} \u00b0C\n"
                f"- Humidity: {weather.get('humidity')}%\n"
                f"- Wind speed: {weather.get('wind_speed_ms')} m/s"
            )

        user_input = (
            f"The crop-disease detection model flagged the following disease:\n"
            f"- Disease: {disease_name}\n"
            f"- Confidence score: {confidence * 100:.2f}%\n"
            f"{weather_block}\n\n"
            "Please explain this disease in detail as per your instructions, "
            "and mention how the current local weather may affect its spread or management."
        )
        return self.chain.invoke({"input": user_input})

    def regional_analysis(self, crop: str, location: str) -> str:
        """Generate a regional suitability report for a crop in a location.

        Covers the crop's chances of survival, whether it is native to the
        region, and the average yield it produces there.
        """
        user_input = (
            f"The user is growing the following crop:\n"
            f"- Crop: {crop}\n"
            f"- Region/State: {location}\n\n"
            "Please provide a concise, farmer-friendly regional suitability "
            "report that explicitly covers, using bullet points:\n"
            "1. The estimated chances (%) of this crop surviving and thriving in this region.\n"
            "2. Whether this crop is native to (or traditionally grown in) this region.\n"
            "3. The approximate average yield percentage this crop typically produces in this region.\n"
            "Base your answer on general agricultural knowledge of the crop and the region's "
            "climate/soil. If the region is unknown or the crop is uncommon there, say so clearly."
        )
        return self.chain.invoke({"input": user_input})

    def chat(
        self,
        question: str,
        disease_name: str | None = None,
        confidence: float | None = None,
    ) -> str:
        """Answer a follow-up farming question using the existing AgriExpert chain.

        If a recent analysis is available, its disease/confidence is passed as
        context so the answer stays grounded in the detected condition.
        """
        context = ""
        if disease_name:
            conf = f"{confidence * 100:.2f}%" if confidence is not None else "not available"
            context = (
                "Context: A crop-image analysis recently flagged the disease "
                f"'{disease_name}' with {conf} confidence.\n\n"
            )
        user_input = f"{context}Farmer question: {question}"
        return self.chain.invoke({"input": user_input})
