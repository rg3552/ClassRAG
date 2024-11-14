from rag_chain import setup_rag_chain
import streamlit as st

def main():
    st.title("COMSE6998-015 Helper Bot")
    # Setup the RAG chain
    rag_chain = setup_rag_chain()

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

if __name__ == "__main__":
    main()