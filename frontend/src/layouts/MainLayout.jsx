// import React, { useState } from 'react';
// import { useWebSocket } from '../services/WebSocketContext';

// // Component for rendering each history item
// const HistoryItem = ({ item, index }) => {
//   const [isExpanded, setIsExpanded] = useState(false);

//   return (
//     <li
//       key={index}
//       style={{ marginBottom: '10px', cursor: 'pointer' }}
//       onClick={() => setIsExpanded(!isExpanded)}
//     >
//       <strong>{item.sender === 'You' ? 'You' : 'Backend'}:</strong> {item.sender === 'You' ? item.content.prompt : item.content.reply}
//       {isExpanded && (
//         <div style={{ marginTop: '5px', fontSize: '0.9em', color: '#555' }}>
//           <pre>{JSON.stringify(item.content, null, 2)}</pre>
//         </div>
//       )}
//     </li>
//   );
// };

// // MainLayout component
// const MainLayout = () => {
//   const { wsClient, isConnected } = useWebSocket(); // Assuming WebSocket context is imported
//   const [selectedModels, setSelectedModels] = useState({
//     model1: false,
//     model2: false,
//     model3: false,
//     model4: false,
//   });
//   const [messages, setMessages] = useState([]);
//   const [inputMessage, setInputMessage] = useState('');

//   const sendMessage = () => {
//     const activeModels = Object.keys(selectedModels).filter((model) => selectedModels[model]);

//     if (!inputMessage.trim()) {
//       console.error('Input message is empty');
//       return;
//     }

//     if (activeModels.length === 0) {
//       console.error('No models selected');
//       return;
//     }

//     if (isConnected && wsClient) {
//       const messagePayload = {
//         user_id: 'USER123',
//         session_id: 'SESSION456',
//         prompt: inputMessage,
//         models: activeModels,
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

//       {/* History Section */}
//       <div className="history">
//         <h4>History</h4>
//         <ul style={{ listStyle: 'none', padding: '0' }}>
//           {messages.map((msg, index) => (
//             <HistoryItem key={index} item={msg} />
//           ))}
//         </ul>
//       </div>

//       {/* Rest of the UI */}
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
//                 onChange={() => setSelectedModels((prev) => ({ ...prev, [model]: !prev[model] }))}
//               />
//               {model.toUpperCase()}
//             </label>
//           ))}
//         </div>

//         <div className="models">
//           {Object.keys(selectedModels).map(
//             (model) =>
//               selectedModels[model] && (
//                 <div key={model} className="model-box">
//                   <h4>{model.toUpperCase()}</h4>
//                   <p>Output will appear here...</p>
//                 </div>
//               )
//           )}
//         </div>
//       </div>
//     </div>
//   );
// };


// export default MainLayout;



import React, { useState, useEffect } from 'react';
import { useWebSocket } from '../services/WebSocketContext';
import './MainLayout.css';

const MainLayout = () => {
  const { wsClient, isConnected } = useWebSocket(); // Use connection state
  const [selectedModels, setSelectedModels] = useState({
    model1: false,
    model2: false,
    model3: false,
    model4: false,
  });
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');

  const handleCheckboxChange = (model) => {
    setSelectedModels((prevState) => ({
      ...prevState,
      [model]: !prevState[model],
    }));
  };

  useEffect(() => {
    if (wsClient && isConnected) {
      console.log('WebSocket is ready for communication');
      wsClient.onMessage((data) => {
        console.log('Received from backend:', data);
        setMessages((prev) => [...prev, { sender: 'Backend', content: data }]);
      });
    }
  }, [wsClient, isConnected]);

  const sendMessage = () => {
    const activeModels = Object.keys(selectedModels).filter(
      (model) => selectedModels[model]
    );

    if (!inputMessage.trim()) {
      console.error('Input message is empty');
      return;
    }

    if (activeModels.length === 0) {
      console.error('No models selected');
      return;
    }

    if (isConnected && wsClient) {
      const messagePayload = {
        user_id: 'USER123', // Placeholder for now
        session_id: 'SESSION456', // Placeholder for now
        prompt: inputMessage,
        models: activeModels,
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

        <div className="models">
          {Object.keys(selectedModels).map(
            (model) =>
              selectedModels[model] && (
                <div key={model} className="model-box">
                  <h4>{model.toUpperCase()}</h4>
                  <p>Output will appear here...</p>
                </div>
              )
          )}
        </div>
      </div>
    </div>
  );
};

export default MainLayout;
