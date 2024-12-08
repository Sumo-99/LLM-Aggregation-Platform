import React, { useState, useEffect } from 'react';
import { useWebSocket } from '../services/WebSocketContext';
import axios from 'axios'; // For REST API requests
import './MainLayout.css';

const MainLayout = () => {
  const { wsClient, isConnected, updateWebSocketUrl } = useWebSocket(); // Added `updateWebSocketUrl`
  const [selectedModels, setSelectedModels] = useState({
    "123": false,
    "234": false,
    model3: false,
    model4: false,
  });
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [sessionId, setSessionId] = useState(null); // Track session ID

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
        session_id: `session-${Date.now()}`, // Generate unique session ID
        user_id: 'USER123', // Replace with actual user ID if available
      };

      const response = await axios.post('http://127.0.0.1:8000/start-session', sessionData);
      console.log('Session started:', response.data);

      setSessionId(response.data.ws_session_id); // Store session ID
      console.log("Making requerst to ", `ws://127.0.0.1:8000/ws/${response.data.ws_session_id}`);
      updateWebSocketUrl(`ws://127.0.0.1:8000/ws/${response.data.ws_session_id}`); // Update WebSocket URL
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
        user_id: 'USER123', // Replace with actual user ID if available
        session_id: sessionId,
        prompt: inputMessage,
        models: Object.keys(selectedModels).filter((model) => selectedModels[model]),
      };

      console.log('Sending message:', messagePayload);
      wsClient.sendMessage(messagePayload);
      setMessages((prev) => [...prev, { sender: 'You', content: messagePayload }]);
      setInputMessage('');
    } else {
      console.error('WebSocket is not connected');
    }
  };

  return (
    <div className="main-layout">
      <h1>Multi-LLM Platform</h1>

      {/* Session Management */}
      <div className="session-management">
        <button onClick={handleStartSession}>Start Session</button>
      </div>

      {/* History Section */}
      <div className="history">
        <h4>History</h4>
        <ul>
          {messages.map((msg, index) => (
            <li key={index}>
              <strong>{msg.sender}:</strong> {JSON.stringify(msg.content)}
            </li>
          ))}
        </ul>
      </div>

      {/* Main Content */}
      <div className="content">
        <div className="search-bar">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="Enter your prompt..."
          />
          <button onClick={sendMessage}>Send</button>
        </div>

        <div className="model-selector">
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
        </div>
      </div>
    </div>
  );
};

export default MainLayout;
