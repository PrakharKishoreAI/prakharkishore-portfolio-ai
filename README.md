# prakharkishore-portfolio-ai
AI-powered portfolio assistant using RAG, Groq and FastAPI
# 🤖 Prakhar Portfolio AI

An AI-powered interactive portfolio assistant that allows recruiters, interviewers, and visitors to ask questions about my **education, experience, skills, internships, and projects**.

The application uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from my portfolio and resume before generating an answer with **Groq's Llama 3.3 70B** model.

## 🚀 Live Demo

👉 **[Try Prakhar Portfolio AI](https://prakharkishore-portfolio-ai.vercel.app)**

You can directly open the application and ask questions without installing anything.

## 📂 GitHub Repository

👉 **[View Source Code](https://github.com/PrakharKishoreAI/prakharkishore-portfolio-ai)**

---

## 🧠 How It Works

```text
                    User
                      │
                      ▼
              React Frontend
                 (Vercel)
                      │
                      │ POST /api/chat
                      ▼
              FastAPI Backend
                 (Render)
                      │
                      ▼
             RAG Retrieval
                      │
              ┌───────┴───────┐
              ▼               ▼
         Resume PDF      Portfolio TXT
              │               │
              └───────┬───────┘
                      ▼
              TF-IDF Retrieval
                      │
                      ▼
              Relevant Context
                      │
                      ▼
              Groq Llama 3.3 70B
                      │
                      ▼
              AI Answer + Sources
                      │
                      ▼
              React Frontend

✨ Features
💬 Interactive AI portfolio assistant
🔎 Retrieval-Augmented Generation (RAG)
📄 Resume and portfolio-based knowledge retrieval
🧠 Lightweight TF-IDF retrieval system
🤖 Groq Llama 3.3 70B
⚡ FastAPI backend
⚛️ React + Vite frontend
📚 Displays information sources used for answers
🌐 Fully deployed application
🔐 API key stored securely as an environment variable
📱 Accessible directly from a web browser
🛠️ Tech Stack
Frontend
React
Vite
JavaScript
CSS
Backend
Python
FastAPI
Uvicorn
PyMuPDF
Groq SDK
AI / RAG
Retrieval-Augmented Generation
TF-IDF
Cosine Similarity
Groq API
Llama 3.3 70B
Deployment
Vercel — Frontend
Render — Backend
GitHub — Source Code
📁 Project Structure
prakharkishore-portfolio-ai/
│
├── api/
│   └── main.py
│
├── data/
│   ├── portfolio.txt
│   └── resume.pdf
│
├── services/
│   └── rag_service.py
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   │
│   ├── package.json
│   └── vite.config.js
│
├── requirements.txt
└── README.md
🔄 RAG Pipeline

The application follows a simple RAG workflow:

1. Document Loading

The system loads information from:

Resume PDF
Portfolio text file
2. Text Extraction

The resume text is extracted using PyMuPDF.

3. Chunking

Large documents are divided into smaller chunks so that relevant information can be retrieved efficiently.

4. TF-IDF Retrieval

The system converts the document chunks into TF-IDF representations.

When a user asks a question, the system calculates similarity between the question and the available document chunks.

5. Context Retrieval

The most relevant chunks are selected as evidence.

6. LLM Generation

The retrieved evidence is passed to Groq Llama 3.3 70B, which generates the final response.

7. Source Display

The application also displays the sources used to generate the response.

💻 Local Setup
1. Clone the repository
git clone https://github.com/PrakharKishoreAI/prakharkishore-portfolio-ai.git
cd prakharkishore-portfolio-ai
2. Install backend dependencies
pip install -r requirements.txt
3. Configure Groq API Key

Create an environment variable:

GROQ_API_KEY=your_groq_api_key

⚠️ Never commit your actual API key to GitHub.

4. Start the FastAPI backend
uvicorn api.main:app --reload --port 8000

Backend:

http://127.0.0.1:8000
5. Start the React frontend

Open another terminal:

cd frontend

Install dependencies:

npm install

Start the development server:

npm run dev

Frontend:

http://localhost:5173
🌐 Deployment
Frontend

The React application is deployed using Vercel.

Production URL:

👉 https://prakharkishore-portfolio-ai.vercel.app

Backend

The FastAPI application is deployed using Render.

Backend URL:

👉 https://prakharkishore-portfolio-ai.onrender.com

The React frontend communicates with the FastAPI backend through:

POST /api/chat
🔌 API Example
Request
POST /api/chat
{
  "question": "Tell me about Prakhar's major project."
}
Response
{
  "question": "Tell me about Prakhar's major project.",
  "answer": "Prakhar's major academic project focuses on...",
  "sources": [
    "Resume PDF",
    "Portfolio TXT"
  ]
}
💡 Example Questions

Visitors can ask:

Where did Prakhar work as a Research Intern?
Tell me about Prakhar's major project.
What are Prakhar's technical skills?
What is Prakhar's educational background?
Tell me about Prakhar's internships.
🔐 Security

The Groq API key is not stored in the frontend.

It is provided to the backend through an environment variable:

GROQ_API_KEY

Never add .env files or API keys to the GitHub repository.

🎯 Purpose

This project was created to demonstrate practical skills in:

Generative AI
Large Language Models
Retrieval-Augmented Generation
Natural Language Processing
API development
React development
AI application deployment
Cloud deployment
Full-stack AI development
👨‍💻 Author
Prakhar Kishore

B.Tech — Computer Science & Engineering

Interested in:

Artificial Intelligence
Machine Learning
Generative AI
LLMs
RAG
Computer Vision
Connect
🔗 GitHub: https://github.com/PrakharKishoreAI
🌐 Portfolio: https://prakharkishore.netlify.app
🤖 AI Portfolio: https://prakharkishore-portfolio-ai.vercel.app
⭐ If you find this project interesting

Feel free to explore the repository and try the live AI portfolio assistant.

Built with React + FastAPI + RAG + Groq 🚀


### One important thing, bro

Your current `portfolio.txt` contains some older technology references such as **FAISS and Sentence Transformers**, while the deployed version was changed to lightweight TF-IDF to solve Render's 512 MB memory limit. 

So **before publishing this README**, I recommend updating those old references in `portfolio.txt` too, so your AI doesn't tell recruiters that this particular deployed application is using FAISS/Sentence Transformers when it isn't.

The README above describes the **actual final deployed architecture**
