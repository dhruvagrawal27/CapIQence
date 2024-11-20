# CapIQence  

## Project Overview  
ChatGroq is a streamlined and efficient open-source language model designed for rapid information retrieval and processing. This project integrates web scraping, text processing, and vector-based retrieval to create a fully functional LLM application capable of delivering insights from news articles and custom queries.  


Demo
[![Watch the video](https://raw.githubusercontent.com/dhruvagrawal27/CapIQence/main/thumbnail.png)](https://raw.githubusercontent.com/dhruvagrawal27/CapIQence/main/video.mp4)

---

## Project Flow  

1. **Project Initialization**  
    - Set up a Python environment.  
    - Install dependencies using `requirements.txt`.  
    - Initialize the project structure.  

2. **Input Handling**  
    - Take the name of any **company** or **IPO** from the user.  

3. **Data Collection**  
    - Use **SERP API** to extract the top 10 news articles about the company/IPO from Google News.  
    - Employ **Beautiful Soup** for web scraping and format the data into a specific **JSON structure**.  

4. **Text Processing**  
    - Use the **Recursive Text Splitter** to split the data into manageable chunks for analysis.  

5. **Embedding Creation**  
    - Use a **Hugging Face embedding model** to convert the text data into embeddings.  
    - Store these embeddings locally using **FAISS (Facebook AI Similarity Search)** as the vector database.  

6. **Retrieval Chain Setup**  
    - Implement a **retrieval chain** that uses FAISS to query the vector database.  
    - Test the retrieval chain with sample queries to ensure proper functionality.  

7. **Streamlit Integration**  
    - Create a **Streamlit app** to enable interactive user queries and display the results dynamically.  

8. **Testing and Deployment**  
    - Test the Streamlit app locally using `streamlit run`.  
    - Optimize and document the application for user accessibility.  

---

## Commands and Setup  

### Prerequisites  
Ensure you have Python and the necessary libraries installed:  
```bash
pip install -r requirements.txt
```

### Key Terminal Commands  
1. **Run the Streamlit Application**  
    ```bash
    streamlit run app.py
    ```

2. **Install Dependencies**  
    ```bash
    pip install serpapi beautifulsoup4 faiss-cpu huggingface_hub
    ```
---
