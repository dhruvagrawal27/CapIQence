import os
import time
import pickle
import requests
from typing import List, Tuple
from dataclasses import dataclass
from bs4 import BeautifulSoup
import streamlit as st
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
FAISS_STORE_PATH = "faiss_store_openai.pkl"
MODEL_NAME = "mixtral-8x7b-32768"

@dataclass
class NewsArticle:
    title: str
    link: str
    snippet: str
    source: str
    date: str

class NewsProcessor:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.llm = ChatGroq(
            temperature=1.1,
            model_name=MODEL_NAME,
            max_tokens=500,
            groq_api_key=os.getenv('groq_api_key')
        )

    @staticmethod
    def fetch_google_news(query: str) -> Tuple[List[NewsArticle], List[str]]:
        """Fetch news articles from Google News via SERP API."""
        try:
            response = requests.get(
                "https://serpapi.com/search.json",
                params={
                    "q": query,
                    "tbm": "nws",
                    "api_key": os.getenv('SERP_API_KEY'),
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            articles = []
            urls = []
            for result in data.get("news_results", []):
                article = NewsArticle(
                    title=result.get("title", ""),
                    link=result.get("link", ""),
                    snippet=result.get("snippet", ""),
                    source=result.get("source", ""),
                    date=result.get("date", "")
                )
                articles.append(article)
                urls.append(article.link)
            
            return articles, urls
        except Exception as e:
            st.error(f"Error fetching news: {str(e)}")
            return [], []

    @staticmethod
    def fetch_and_process_urls(urls: List[str]) -> Tuple[List[Document], List[str]]:
        """Fetch and process webpage content."""
        documents = []
        errors = []
        
        # Create a progress bar
        progress_text = "Processing articles..."
        progress_bar = st.progress(0, text=progress_text)
        
        total_urls = len(urls)
        for idx, url in enumerate(urls):
            try:
                # Update progress
                progress = (idx + 1) / total_urls
                progress_bar.progress(progress, text=f"{progress_text} ({idx + 1}/{total_urls})")
                
                response = requests.get(
                    url,
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'},
                    timeout=10
                )
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                main_content = soup.get_text(separator=' ', strip=True)
                
                documents.append(Document(
                    page_content=main_content,
                    metadata={
                        "source": url,
                        "title": soup.title.string if soup.title else "No Title",
                    }
                ))
                
            except Exception as e:
                errors.append(f"Error processing {url}: {str(e)}")
        
        # Complete the progress bar
        progress_bar.progress(1.0, text="Processing complete!")
        time.sleep(0.5)  # Short pause to show completion
        
        return documents, errors

    def process_documents(self, documents: List[Document]) -> FAISS:
        """Process documents and create FAISS index."""
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        docs = text_splitter.split_documents(documents)
        return FAISS.from_documents(docs, self.embeddings)

class StreamlitUI:
    def __init__(self):
        st.set_page_config(
            page_title="CapIQEnce - News Analysis",
            page_icon="📰",
            layout="wide"
        )
        self.processor = NewsProcessor()

    def setup_sidebar(self):
        st.sidebar.title("📰 CapIQEnce")
        st.sidebar.markdown("---")
        company_name = st.sidebar.text_input("Enter Company/Topic Name")
        process_button = st.sidebar.button(
            "Analyze News",
            help="Click to fetch and analyze latest news"
        )
        return company_name, process_button

    def display_articles(self, articles: List[NewsArticle]):
        """Display articles in an organized manner."""
        for article in articles:
            with st.expander(f"📄 {article.title}"):
                st.markdown(f"**Source:** {article.source}")
                st.markdown(f"**Date:** {article.date}")
                st.markdown(f"**Summary:** {article.snippet}")
                st.markdown(f"[Read full article]({article.link})")

    def main(self):
        st.title("CapIQEnce News Analyzer")
        st.markdown("---")

        company_name, process_clicked = self.setup_sidebar()

        if process_clicked and company_name:
            st.info("🔍 Fetching news articles...")
            articles, urls = self.processor.fetch_google_news(company_name)
            
            if not articles:
                st.error("No news articles found. Please try a different search term.")
                return

            st.success(f"Found {len(articles)} news articles!")
            
            st.markdown("### 📑 View Articles")
            self.display_articles(articles)


            st.info("📝 Processing article content...")
            documents, errors = self.processor.fetch_and_process_urls(urls)

            if documents:
                vectorstore = self.processor.process_documents(documents)
                with open(FAISS_STORE_PATH, "wb") as f:
                    pickle.dump(vectorstore, f)
                st.success("✅ Processing complete! You can now ask questions about the articles.")

        st.markdown("---")
        query = st.text_input("🤔 Ask a question about the news articles:")
        
        if query and os.path.exists(FAISS_STORE_PATH):
            try:
                with st.spinner("🤖 Analyzing..."):
                    with open(FAISS_STORE_PATH, "rb") as f:
                        vectorstore = pickle.load(f)
                    chain = RetrievalQAWithSourcesChain.from_llm(
                        llm=self.processor.llm,
                        retriever=vectorstore.as_retriever()
                    )
                    result = chain({"question": query}, return_only_outputs=True)
                    
                    st.markdown("### 💡 Answer")
                    st.write(result["answer"])
                    
                    if sources := result.get("sources"):
                        st.markdown("### 📚 Sources")
                        for source in sources.split("\n"):
                            if source.strip():
                                st.markdown(f"- {source.strip()}")
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")

if __name__ == "__main__":
    app = StreamlitUI()
    app.main()