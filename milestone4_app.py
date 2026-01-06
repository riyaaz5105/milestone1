import streamlit as st
import requests

st.set_page_config(page_title="AI Workflow Manager", layout="centered")

st.title("🚀 Multi-Agent Orchestrator")
st.write("Enter a topic to trigger the Research → Write pipeline.")

topic = st.text_input("What should we research?", placeholder="e.g. Quantum Computing")

if st.button("Start Workflow"):
    if topic:
        with st.spinner("Agents are working..."):
            try:
                # Calling the FastAPI backend
                response = requests.post(
                    "http://localhost:8000/run-workflow",
                    json={"topic": topic}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("Workflow Complete!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Researcher Output")
                        st.info(result["research_summary"])
                    
                    with col2:
                        st.subheader("Final Email")
                        st.code(result["final_output"], language="markdown")
                else:
                    st.error("API Error: " + response.text)
            except Exception as e:
                st.error(f"Connection failed: {e}")
    else:
        st.warning("Please enter a topic first.")