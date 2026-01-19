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


tab1,tab2=st.tabs(["Fake News Detection","Summary and Key points"])

#chatbot sidebar
st.sidebar.title("News Based Chatbot")
chat_input=st.sidebar.text_area("Enter text to analyze or ask a question:",height=160,key="chat_input")
analyze_text=st.sidebar.button("Analyze News Text")
clear_text=st.sidebar.button("Clear History")
if analyze_text:
    if not chat_input.strip():
        st.sidebar.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Detecting..."):
            completion = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                messages=[
                    {
                        "role": "system",
                        "content": f"""
                        You are an intelligent news analysis assistant.
                         Your Tasks are:
                         1.If user asks a question, answer factually.
                         2.If user provides news text,detect fake or real.
                         3.Identify political bias(left/right/center)
                         4.If user asks a general question, answer it normally.
                         Always be concise and informative.
                        """
                    },
                    {
                        "role":"user",
                        "content":chat_input
                    }
                ],
                temperature=0.3,
                max_tokens=300
            )
            response = completion.choices[0].message['content']
            st.sidebar.markdown("### Analysis Result")
            st.sidebar.success(response)
            st.session_state.chat.append((chat_input,response))
if st.session_state.chat:
    st.sidebar.markdown("### Analysis History")
    for i,(input,out) in enumerate(reversed(st.session_state.chat[-3:]),1):
        st.sidebar.markdown(f"**Text {i}:**")
        st.sidebar.write(out)


#fake news detection
with tab1:
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
with tab2:
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
