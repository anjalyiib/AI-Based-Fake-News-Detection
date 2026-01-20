import streamlit as st
from huggingface_hub import InferenceClient

#hugging face token
HF_TOKEN = st.secrets["HF_TOKEN"]

@st.cache_resource
def load_client():
    return InferenceClient(token=HF_TOKEN)
client = load_client()

#intialise session state
if "chat" not in st.session_state:
    st.session_state.chat = []


tab1,tab2,tab3=st.tabs(["News Chatbot","Fake News Detection","Summary and Key points"])

#chatbot tab
with tab1:
    st.title("News Based Chatbot")
    if "chat" not in st.session_state:
        st.session_state.chat = []
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input=st.chat_input("Ask a news-related question...")
if user_input:
    st.session_state.chat.append({
        "role":"user",
        "content":user_input
    })
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.spinner("Thinking..."):
        completion=client.chat.completions.create(
           model="meta-llama/Meta-Llama-3-8B-Instruct",
                        messages=[
                            {
                                "role": "system",
                                "content": f"""
                                You are an intelligent news analysis assistant.
                                Answer like a human.
                                Detect fake news if news text is provided.extra_body
                                """
                            }
                        ]+st.session_state.chat,
                        temperature=0.4,
                        max_tokens=300
        )
        reply=completion.choices[0].message["content"]

        st.session_state.chat.append({
            "role":"assistant",
            "content":reply
        })

        with st.chat_message("assistant"):
            st.markdown(reply)

if st.sidebar.button("Clear Chat"):
    st.session_state.chat=[]

#fake news detection
with tab2:
    st.title("Fake News Detection")
    news_input=st.text_area("Enter news article or headline to analyze for authenticity:",height=200, key="text_area")
    detect_fake_news=st.button("Detect Fake News")
    if detect_fake_news:
        if news_input.strip()=="":
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing..."):
                completion = client.chat.completions.create(
                    model="meta-llama/Meta-Llama-3-8B-Instruct",
                    messages=[
                        {
                            "role": "system",
                            "content": f"""
                            You are a fake news detection expert.
                            Analyze the following news article or headline and determine its authenticity. 
                            Provide a detailed explanation of your analysis.
                            Text: {news_input}
                            """
                        }
                    ],
                    temperature=0.7,
                    max_tokens=150
                )
                response = completion.choices[0].message['content']
                st.markdown("### Analysis Result")
                st.success(response)

#summmary and key points
with tab3:
    st.title("Summary and key Points")
    summary_input=st.text_area("Enter text to summarise:",height=200)
    analyze_text=st.button("Generate Summary")
    if analyze_text:
        if summary_input.strip()=="":
            st.warning("Please enter some text.")
        else:
            with st.spinner("Genarating..."):
                completion = client.chat.completions.create(
                    model="meta-llama/Meta-Llama-3-8B-Instruct",
                    messages=[
                        {
                            "role": "system",
                            "content": f"""
                            You are news summarization expert.
                            Summarise the news in 4-5 lines and list key points in bullets.
                            """
                        },
                        {
                            "role":"user",
                            "content":summary_input
                        }
                    ],
                    temperature=0.5,
                    max_tokens=200
                )
                response = completion.choices[0].message['content']
                st.markdown("Summary Generated")
                st.success(response)



st.caption("Disclaimer: This System uses AI models for educational purposes. " 
           "Results may be inaccurate.")
