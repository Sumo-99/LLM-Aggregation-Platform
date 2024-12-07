import React, { useState, useEffect } from 'react';
import { useWebSocket } from '../services/WebSocketContext';
import axios from 'axios';
import './MainLayout.css';

const MainLayout = () => {
  const { wsClient, isConnected, updateWebSocketUrl } = useWebSocket();
  const [selectedModels, setSelectedModels] = useState({
    "123": false,
    model2: false,
    model3: false,
    model4: false,
  });
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [sessionId, setSessionId] = useState(null);

  useEffect(() => {
    if (wsClient && isConnected) {
      console.log('WebSocket is ready for communication');
      wsClient.onMessage((data) => {
        console.log('Received from backend:', data);
        setMessages((prev) => [...prev, { sender: 'Backend', content: data }]);
      });
    }
  }, [wsClient, isConnected]);

  const handleCheckboxChange = (model) => {
    setSelectedModels((prevState) => ({
      ...prevState,
      [model]: !prevState[model],
    }));
  };

  const handleStartSession = async () => {
    const activeModels = Object.keys(selectedModels).filter(
      (model) => selectedModels[model]
    );
  
    if (activeModels.length === 0) {
      console.error('No models selected');
      return;
    }
  
    try {
      const sessionData = {
        model_ids: activeModels,
        prompt: 'Initial Prompt',
        session_id: `session-${Date.now()}`,
        user_id: 'USER123',
      };
  
      const response = await axios.post('http://127.0.0.1:8000/start-session', sessionData);
      console.log('Session started:', response.data);
  
      setSessionId(response.data.ws_session_id);
      console.log("Making request to ", `ws://127.0.0.1:8000/ws/${response.data.ws_session_id}`);
      updateWebSocketUrl(`ws://127.0.0.1:8000/ws/${response.data.ws_session_id}`);
  
      // Wait for WebSocket to connect
      const checkConnection = setInterval(() => {
        if (wsClient && isConnected) {
          console.log('WebSocket connected, setting up handlers...');
          wsClient.onMessage((data) => {
            console.log('Received from backend:', data);
            setMessages((prev) => [...prev, { sender: 'Backend', content: data }]);
          });
          clearInterval(checkConnection); // Stop checking once connected
        }
      }, 100);
    } catch (error) {
      console.error('Error starting session:', error);
    }
  };
  
  

  const sendMessage = () => {
    if (!sessionId) {
      console.error('No active session. Start a session first.');
      return;
    }

    if (!inputMessage.trim()) {
      console.error('Input message is empty');
      return;
    }

    if (isConnected && wsClient) {
      const messagePayload = {
        user_id: 'USER123',
        session_id: sessionId,
        prompt: inputMessage,
        models: Object.keys(selectedModels).filter((model) => selectedModels[model]),
      };

      console.log('Sending message:', messagePayload);
      wsClient.sendMessage(messagePayload);
      setMessages((prev) => [...prev, { sender: 'You', content: inputMessage }]);
      setInputMessage('');
    } else {
      console.error('WebSocket is not connected');
    }
  };

  return (
    <div className="main-layout">
      <div className="header">
        <h1>Multi-LLM Platform</h1>
      </div>

      <div className="main">
        {/* History Section */}
        <div className="history">
          <h4>History</h4>
          <ul>
            {/* History remains blank */}
          </ul>
        </div>

        {/* Chat Section */}
        <div className="chat">
          <div className="messages">
            {messages.map((msg, index) => (
              <div key={index} className={`message ${msg.sender === 'You' ? 'user-message' : 'backend-message'}`}>
                <strong>{msg.sender}:</strong> {msg.content}
              </div>
            ))}
          </div>
          <div className="search-bar">
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="Enter your prompt..."
            />
            <button onClick={sendMessage}>Send</button>
          </div>
        </div>

        {/* Model Selector and Session Management */}
        <div className="model-selector">
          <h4>Select Models</h4>
          {Object.keys(selectedModels).map((model) => (
            <label key={model}>
              <input
                type="checkbox"
                checked={selectedModels[model]}
                onChange={() => handleCheckboxChange(model)}
              />
              {model.toUpperCase()}
            </label>
          ))}
          <button onClick={handleStartSession}>Start Session</button>
        </div>
      </div>
    </div>
  );
};

export default MainLayout;
