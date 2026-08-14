import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from worker.agents.explainer_agent import DiseaseExplainerAgent
from worker.agents.graph import CropDiseaseWorkflow

st.set_page_config(page_title="Crop Disease Classifier", page_icon="🌾", layout="centered")

st.title("🌾 Crop Disease Classifier")
st.caption("Upload a photo of a crop/leaf. A CNN model detects the disease and a Mistral-powered agent explains it.")


@st.cache_resource
def get_workflow():
    explainer = DiseaseExplainerAgent()
    return CropDiseaseWorkflow(explainer_agent=explainer)


uploaded_file = st.file_uploader("Upload crop image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

    if st.button("Analyze"):
        with st.spinner("Analyzing image..."):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                workflow = get_workflow()
                result = workflow.run(tmp_path)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        st.subheader("Prediction")
        disease = result.get("predicted_disease")
        confidence = result.get("confidence")

        if disease and confidence is not None:
            st.metric("Detected Disease", disease)
            st.metric("Confidence", f"{confidence * 100:.2f}%")

        if result.get("explanation"):
            st.subheader("Explanation")
            st.write(result["explanation"])
        else:
            st.warning(
                "Confidence was below 50%, so the AI explanation was skipped. "
                "Please verify the crop manually or try a clearer image."
            )
