import { useState } from "react";
import "./App.css";

// Render backend URL
const API_URL =
  "https://prakharkishore-portfolio-ai.onrender.com/api/chat";

const suggestedQuestions = [
  "Where did Prakhar work as a Research Intern?",
  "Tell me about Prakhar's major project.",
  "What are Prakhar's technical skills?",
  "What is Prakhar's educational background?",
];

function App() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const askQuestion = async (customQuestion = null) => {
    const finalQuestion = customQuestion || question.trim();

    if (!finalQuestion || loading) {
      return;
    }

    // Add user message
    setMessages((prev) => [
      ...prev,
      {
        type: "user",
        text: finalQuestion,
      },
    ]);

    setQuestion("");
    setLoading(true);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: finalQuestion,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();

      // Add AI response
      setMessages((prev) => [
        ...prev,
        {
          type: "ai",
          text: data.answer,
          sources: data.sources || [],
        },
      ]);
    } catch (error) {
      console.error("API Error:", error);

      setMessages((prev) => [
        ...prev,
        {
          type: "ai",
          text: "Sorry, I couldn't connect to the AI server.",
          sources: [],
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    askQuestion();
  };

  const clearChat = () => {
    setMessages([]);
  };

  return (
    <div className="app">
      <div className="container">

        {/* Header */}
        <header className="header">
          <div>
            <h1>Prakhar Portfolio AI</h1>

            <p>
              Ask me anything about my experience, projects, skills and
              education.
            </p>
          </div>

          {messages.length > 0 && (
            <button className="clear-btn" onClick={clearChat}>
              Clear Chat
            </button>
          )}
        </header>

        {/* Suggested Questions */}
        {messages.length === 0 && (
          <div className="suggestions">
            <h3>Try asking</h3>

            <div className="suggestion-grid">
              {suggestedQuestions.map((item, index) => (
                <button
                  key={index}
                  onClick={() => askQuestion(item)}
                  className="suggestion-btn"
                >
                  {item}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Chat Messages */}
        <div className="chat-area">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`message ${
                message.type === "user"
                  ? "user-message"
                  : "ai-message"
              }`}
            >
              <div className="message-label">
                {message.type === "user" ? "You" : "AI"}
              </div>

              <div className="message-text">
                {message.text}
              </div>

              {/* Sources */}
              {message.type === "ai" &&
                message.sources &&
                message.sources.length > 0 && (
                  <div className="sources">
                    <div className="sources-title">
                      Sources
                    </div>

                    <div className="source-list">
                      {message.sources.map((source, sourceIndex) => (
                        <span
                          className="source-tag"
                          key={sourceIndex}
                        >
                          📄 {source}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
            </div>
          ))}

          {/* Loading */}
          {loading && (
            <div className="message ai-message">
              <div className="message-label">AI</div>

              <div className="typing">
                <span></span>
                <span></span>
                <span></span>
                Thinking...
              </div>
            </div>
          )}
        </div>

        {/* Input */}
        <form className="input-area" onSubmit={handleSubmit}>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask something about Prakhar..."
            rows="3"
            disabled={loading}
          />

          <button
            type="submit"
            className="ask-btn"
            disabled={loading || !question.trim()}
          >
            {loading ? "Thinking..." : "Ask AI"}
          </button>
        </form>

        {/* Footer */}
        <div className="footer">
          Powered by RAG + Groq + FastAPI + React
        </div>

      </div>
    </div>
  );
}

export default App;