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
        model_name: str = "mistral-large-latest",
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

    def explain(self, disease_name: str, confidence: float) -> str:
        """Generate a farmer-friendly explanation for a flagged disease."""
        user_input = (
            f"The crop-disease detection model flagged the following disease:\n"
            f"- Disease: {disease_name}\n"
            f"- Confidence score: {confidence * 100:.2f}%\n\n"
            "Please explain this disease in detail as per your instructions."
        )
        return self.chain.invoke({"input": user_input})
