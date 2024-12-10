// import React, { useState, useEffect } from 'react';
// import { useWebSocket } from '../services/WebSocketContext';
// import axios from 'axios'; // For REST API requests
// import './MainLayout.css';

// const MainLayout = () => {
//   const { wsClient, isConnected, updateWebSocketUrl } = useWebSocket(); // Added `updateWebSocketUrl`
//   const [selectedModels, setSelectedModels] = useState({
//     "123": false,
//     model2: false,
//     model3: false,
//     model4: false,
//   });
//   const [messages, setMessages] = useState([]);
//   const [inputMessage, setInputMessage] = useState('');
//   const [sessionId, setSessionId] = useState(null); // Track session ID

//   useEffect(() => {
//     if (wsClient && isConnected) {
//       console.log('WebSocket is ready for communication');
//       wsClient.onMessage((data) => {
//         console.log('Received from backend:', data);
//         setMessages((prev) => [...prev, { sender: 'Backend', content: data }]);
//       });
//     }
//   }, [wsClient, isConnected]);

//   const handleCheckboxChange = (model) => {
//     setSelectedModels((prevState) => ({
//       ...prevState,
//       [model]: !prevState[model],
//     }));
//   };

//   const handleStartSession = async () => {
//     const activeModels = Object.keys(selectedModels).filter(
//       (model) => selectedModels[model]
//     );

//     if (activeModels.length === 0) {
//       console.error('No models selected');
//       return;
//     }

//     try {
//       const sessionData = {
//         model_ids: activeModels,
//         prompt: 'Initial Prompt',
//         session_id: `session-${Date.now()}`, // Generate unique session ID
//         user_id: 'USER123', // Replace with actual user ID if available
//       };

//       const response = await axios.post('http://127.0.0.1:8000/start-session', sessionData);
//       console.log('Session started:', response.data);

//       setSessionId(response.data.ws_session_id); // Store session ID
//       console.log("Making requerst to ", `ws://127.0.0.1:8000/ws/${response.data.ws_session_id}`);
//       updateWebSocketUrl(`ws://127.0.0.1:8000/ws/${response.data.ws_session_id}`); // Update WebSocket URL
//     } catch (error) {
//       console.error('Error starting session:', error);
//     }
//   };

//   const sendMessage = () => {
//     if (!sessionId) {
//       console.error('No active session. Start a session first.');
//       return;
//     }

//     if (!inputMessage.trim()) {
//       console.error('Input message is empty');
//       return;
//     }

//     if (isConnected && wsClient) {
//       const messagePayload = {
//         user_id: 'USER123', // Replace with actual user ID if available
//         session_id: sessionId,
//         prompt: inputMessage,
//         models: Object.keys(selectedModels).filter((model) => selectedModels[model]),
//       };

//       console.log('Sending message:', messagePayload);
//       wsClient.sendMessage(messagePayload);
//       setMessages((prev) => [...prev, { sender: 'You', content: messagePayload }]);
//       setInputMessage('');
//     } else {
//       console.error('WebSocket is not connected');
//     }
//   };

//   return (
//     <div className="main-layout">
//       <h1>Multi-LLM Platform</h1>

//       {/* Session Management */}
//       <div className="session-management">
//         <button onClick={handleStartSession}>Start Session</button>
//       </div>

//       {/* History Section */}
//       <div className="history">
//         <h4>History</h4>
//         <ul>
//           {messages.map((msg, index) => (
//             <li key={index}>
//               <strong>{msg.sender}:</strong> {JSON.stringify(msg.content)}
//             </li>
//           ))}
//         </ul>
//       </div>

//       {/* Main Content */}
//       <div className="content">
//         <div className="search-bar">
//           <input
//             type="text"
//             value={inputMessage}
//             onChange={(e) => setInputMessage(e.target.value)}
//             placeholder="Enter your prompt..."
//           />
//           <button onClick={sendMessage}>Send</button>
//         </div>

//         <div className="model-selector">
//           {Object.keys(selectedModels).map((model) => (
//             <label key={model}>
//               <input
//                 type="checkbox"
//                 checked={selectedModels[model]}
//                 onChange={() => handleCheckboxChange(model)}
//               />
//               {model.toUpperCase()}
//             </label>
//           ))}
//         </div>
//       </div>
//     </div>
//   );
// };

// export default MainLayout;


import React, { useState, useEffect } from 'react';
import { useWebSocket } from '../services/WebSocketContext';
import axios from 'axios';
import './MainLayout.css';

const MainLayout = () => {
  const { wsClient, isConnected, updateWebSocketUrl } = useWebSocket();
  const [selectedModels, setSelectedModels] = useState({
    "123": false,
    "234": false,
    model3: false,
    model4: false,
  });
  const [modelOutputs, setModelOutputs] = useState({
    "123": [],
    "234": [],
    model3: [],
    model4: [],
  });
  const [inputMessage, setInputMessage] = useState('');
  const [sessionId, setSessionId] = useState(null);

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
      (model) => selectedModels[model]
    );
  
    // Append the user's prompt to the chat history for each selected model
    activeModels.forEach((model) => {
      setModelOutputs((prev) => ({
        ...prev,
        [model]: [...prev[model], { sender: 'You', message: inputMessage }],
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
      setModelOutputs({
        "123": [],
        "234": [],
        model3: [],
        model4: [],
      });
  
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
        <button onClick={handleStopSession} disabled={!sessionId}>Stop Session</button>
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

        {/* Model Panels */}
        <div className="models">
  {Object.keys(modelOutputs).map((model) => (
    <div key={model} className="model-box">
      <h4>Model: {model.toUpperCase()}</h4>
      <div className="chat-panel">
        {modelOutputs[model].length > 0 ? (
          modelOutputs[model].map((msg, index) => (
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


