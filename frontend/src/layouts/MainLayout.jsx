import React, { useState, useEffect } from 'react';
import { useWebSocket } from '../services/WebSocketContext';
import axios from 'axios';
import './MainLayout.css';

const MainLayout = () => {
  const { wsClient, isConnected, updateWebSocketUrl } = useWebSocket();
  const [models, setModels] = useState([]); // Store models fetched from the API
  const [selectedModels, setSelectedModels] = useState({});
  const [modelOutputs, setModelOutputs] = useState({});
  const [inputMessage, setInputMessage] = useState('');
  const [sessionId, setSessionId] = useState(null);

  useEffect(() => {
    // Fetch models from the API
    const fetchModels = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:8000/models', {
          params: { user_id: 'USER123' },
        });
        const fetchedModels = response.data.models;

        // Initialize the `selectedModels` and `modelOutputs` state based on the fetched models
        const initialSelectedModels = fetchedModels.reduce((acc, model) => {
          acc[model.model_id] = false; // Default to not selected
          return acc;
        }, {});

        const initialModelOutputs = fetchedModels.reduce((acc, model) => {
          acc[model.model_id] = []; // Default to an empty chat history
          return acc;
        }, {});

        setModels(fetchedModels); // Store fetched models
        setSelectedModels(initialSelectedModels);
        setModelOutputs(initialModelOutputs);
      } catch (error) {
        console.error('Error fetching models:', error);
      }
    };

    fetchModels();
  }, []);

  useEffect(() => {
    if (wsClient && isConnected) {
      console.log('WebSocket is ready for communication');

      wsClient.onMessage((data) => {
        let parsedData;

        // Ensure correct parsing of WebSocket messages
        if (typeof data === 'string') {
          try {
            parsedData = JSON.parse(data);
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
            return;
          }
        } else {
          parsedData = data;
        }

        console.log('Parsed Data:', parsedData);

        const { model_id, response } = parsedData;
        if (model_id && response) {
          // Append the model's response to the chat history for the model
          setModelOutputs((prev) => ({
            ...prev,
            [model_id]: [
              ...prev[model_id],
              { sender: `Model ${model_id}`, message: response },
            ],
          }));
        }
      });
    }
  }, [wsClient, isConnected]);

  const handleCheckboxChange = (model_id) => {
    setSelectedModels((prevState) => ({
      ...prevState,
      [model_id]: !prevState[model_id],
    }));
  };

  const handleStartSession = async () => {
    const activeModels = Object.keys(selectedModels).filter(
      (model_id) => selectedModels[model_id]
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
      updateWebSocketUrl(`ws://127.0.0.1:8000/ws/${response.data.ws_session_id}`);
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

    const activeModels = Object.keys(selectedModels).filter(
      (model_id) => selectedModels[model_id]
    );

    // Append the user's prompt to the chat history for each selected model
    activeModels.forEach((model_id) => {
      setModelOutputs((prev) => ({
        ...prev,
        [model_id]: [...prev[model_id], { sender: 'You', message: inputMessage }],
      }));
    });

    if (isConnected && wsClient) {
      const messagePayload = {
        user_id: 'USER123',
        session_id: sessionId,
        prompt: inputMessage,
        models: activeModels,
      };

      console.log('Sending message:', messagePayload);
      wsClient.sendMessage(messagePayload);
      setInputMessage('');
    } else {
      console.error('WebSocket is not connected');
    }
  };

  const handleStopSession = async () => {
    if (!sessionId) {
      console.error('No active session. Start a session first.');
      return;
    }

    try {
      const response = await axios.post('http://127.0.0.1:8000/stop-session', {
        session_id: sessionId,
        user_id: 'USER123',
      });
      console.log('Session stopped:', response.data);

      // Reset session-related state
      setSessionId(null);
      setModelOutputs({});
      setSelectedModels({});
      setModels([]);

      if (wsClient) {
        wsClient.closeConnection(); // Close the WebSocket connection
      }
    } catch (error) {
      console.error('Error stopping session:', error);
    }
  };

  return (
    <div className="main-layout">
      <h1>Multi-LLM Platform</h1>

      {/* Session Management */}
      <div className="session-management">
        <button onClick={handleStartSession}>Start Session</button>
        {/* <button onClick={handleStopSession} disabled={!sessionId}>Stop Session</button> */}
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
          {models.map((model) => (
            <label key={model.model_id}>
              <input
                type="checkbox"
                checked={selectedModels[model.model_id]}
                onChange={() => handleCheckboxChange(model.model_id)}
              />
              {model.model_name}
            </label>
          ))}
        </div>

        {/* Model Panels */}
        <div className="models">
          {Object.keys(modelOutputs).map((model_id) => (
            <div key={model_id} className="model-box">
              <h4>Model: {models.find((model) => model.model_id === model_id)?.model_name || model_id}</h4>
              <div className="chat-panel">
                {modelOutputs[model_id].length > 0 ? (
                  modelOutputs[model_id].map((msg, index) => (
                    <p key={index}>
                      <strong>{msg.sender}:</strong> {msg.message}
                    </p>
                  ))
                ) : (
                  <p>No chat history yet...</p>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default MainLayout;
