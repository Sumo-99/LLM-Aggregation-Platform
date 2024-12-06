import React, { createContext, useContext, useEffect, useRef, useState } from 'react';
import WebSocketClient from './WebSocketClient';

const WebSocketContext = createContext(null);

export const WebSocketProvider = ({ children }) => {
  const [isConnected, setIsConnected] = useState(false); // Track connection state
  const wsClientRef = useRef(null);

  useEffect(() => {
    console.log('Initializing WebSocket connection...');
    wsClientRef.current = new WebSocketClient('ws://localhost:8080'); // Adjust URL if needed

    wsClientRef.current.onOpen(() => {
      console.log('WebSocket connection established');
      setIsConnected(true); // Update state on successful connection
    });

    wsClientRef.current.onMessage((message) => {
      console.log('Message received from WebSocket:', message);
    });

    wsClientRef.current.onClose(() => {
      console.warn('WebSocket connection closed. Attempting to reconnect...');
      setIsConnected(false); // Update state on disconnection
    });

    wsClientRef.current.onError((error) => {
      console.error('WebSocket error:', error);
    });

    return () => {
      if (wsClientRef.current) {
        console.log('Closing WebSocket connection...');
        wsClientRef.current.closeConnection();
      }
    };
  }, []);

  return (
    <WebSocketContext.Provider value={{ wsClient: wsClientRef.current, isConnected }}>
      {children}
    </WebSocketContext.Provider>
  );
};

export const useWebSocket = () => useContext(WebSocketContext);
