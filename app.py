import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEndpoint

## Streamlit APP
st.set_page_config(page_title="Text Summarization from URL", page_icon="👨🏻‍💻")
st.title("👨🏻‍💻 Summarize Text From YouTube or Website")
st.subheader('Enter the URL')

## Get the URL (YT or website) to be summarized
generic_url = st.text_input("URL", label_visibility="collapsed")

## Gemma Model Using Groq API
## llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
hf_api_key = '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'  # Your API token here
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=150, temperature=0.7, huggingfacehub_api_token=hf_api_key)

prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize the Content"):
    ## Validate the input
    if not generic_url.strip():
        st.error("Please enter a URL to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YouTube video URL or a website URL")
    else:
        try:
            with st.spinner("Waiting..."):
                ## Loading the website or yt video data
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False,
                                                   headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs = loader.load()

                ## Chain For Summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception: {e}")
