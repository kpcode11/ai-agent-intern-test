import sys
from pathlib import Path

# Add src to the path so we can import the agent
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
from dotenv import load_dotenv

from aster_row_support.agent import Agent
from aster_row_support.eval_checks import detect_handoff

# Setup the page layout
st.set_page_config(
    page_title="Aster & Row Support",
    page_icon="🛍️",
    layout="centered"
)

def init_app():
    load_dotenv()
    
    # Initialize the Agent in session state so conversation history persists
    if "agent" not in st.session_state:
        try:
            st.session_state.agent = Agent()
        except Exception as e:
            st.error(f"Error initializing agent: {e}")
            st.info("Make sure GROQ_API_KEY and GEMINI_API_KEY are set in your .env file.")
            st.stop()
            
    # Initialize UI chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Welcome to Aster & Row Support! How can I help you today?", "sources": [], "handoff": False}
        ]

def main():
    init_app()
    
    st.title("Aster & Row Support")
    # st.caption("AI-powered customer service assistant")
    st.divider()

    # Render previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Show sources if any (collapsed by default for clean UX)
            if msg.get("sources"):
                with st.expander("Sources Cited", expanded=False):
                    for source in msg["sources"]:
                        st.markdown(f"- `{source}`")
                        
            # Show handoff flag if applicable
            if msg.get("handoff"):
                st.warning("⚠️ **Handoff Recommended:** A human support agent will take over this conversation.", icon="👩‍💻")

    # Chat Input
    if user_input := st.chat_input("Type your message here..."):
        # Immediately display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get response from the agent
                    response = st.session_state.agent.send_message(user_input)
                    
                    # Extract metadata from the trace
                    sources = st.session_state.agent.last_trace.get("sources_used") or []
                    unique_sources = sorted(set(sources))
                    is_handoff = detect_handoff(response) or st.session_state.agent.last_trace.get("handoff")
                    
                    # Render response
                    st.markdown(response)
                    
                    if unique_sources:
                        with st.expander("Sources Cited", expanded=False):
                            for source in unique_sources:
                                st.markdown(f"- `{source}`")
                                
                    if is_handoff:
                        st.warning("⚠️ **Handoff Recommended:** A human support agent will take over this conversation.", icon="👩‍💻")
                        
                    # Save to state
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response, 
                        "sources": unique_sources, 
                        "handoff": is_handoff
                    })
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
