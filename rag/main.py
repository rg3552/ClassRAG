from rag_chain import setup_rag_chain
import streamlit as st

def handle_feedback(feedback, response, text_feedback):
    # Log the feedback, response, and text feedback to a file or database
    print("in handle_feedback")
    with open("feedback.txt", "a") as f:
        f.write(f"Feedback: {feedback}\nResponse: {response}\nText Feedback: {text_feedback}\n")

def main():
    st.title("COMSE6998-015 Helper Bot")
    # Setup the RAG chain

    if "setup_done" not in st.session_state:
        rag_chain = setup_rag_chain()
        st.session_state.setup_done = True
        st.session_state.rag_chain = rag_chain
    else:
        rag_chain = st.session_state.rag_chain

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ask a question about the class"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_query)
    
        st.session_state.messages.append({"role": "user", "content": user_query})

        result = rag_chain.invoke(user_query)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(result)

        st.session_state.messages.append({"role": "assistant", "content": result})
        
        # Add feedback buttons and text input
        st.markdown("*Was this response helpful?*")
        text_feedback = ""
        col1, col2, col3 = st.columns([1, 1, 10])
        with col1:
            st.button("👍", on_click=handle_feedback("positive", result, text_feedback))
        with col2:
            st.button("👎",  on_click=handle_feedback("negative", result, text_feedback))
        with col3:
            text_feedback = st.text_input("Enter feedback", placeholder="Feedback (optional)", label_visibility="collapsed")

if __name__ == "__main__":
    main()