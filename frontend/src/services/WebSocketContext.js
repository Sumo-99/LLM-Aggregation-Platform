import React, { createContext, useContext, useEffect, useRef, useState } from 'react';
import WebSocketClient from './WebSocketClient';

const WebSocketContext = createContext(null);

export const WebSocketProvider = ({ children }) => {
  const [isConnected, setIsConnected] = useState(false);
  const [wsUrl, setWsUrl] = useState(null);
  const wsClientRef = useRef(null);

  useEffect(() => {
    if (!wsUrl) return;

    console.log('Connecting to WebSocket with URL:', wsUrl);
    wsClientRef.current = new WebSocketClient(wsUrl);

    wsClientRef.current.onOpen(() => {
      console.log('WebSocket connection established');
      setIsConnected(true);
    });

    wsClientRef.current.onMessage((message) => {
      console.log('WebSocket message received:', message);
    });

    wsClientRef.current.onClose(() => {
      console.warn('WebSocket connection closed');
      setIsConnected(false);
    });

    wsClientRef.current.onError((error) => {
      console.error('WebSocket error:', error);
    });

    return () => {
      if (wsClientRef.current) {
        wsClientRef.current.closeConnection();
        setIsConnected(false);
      }
    };
  }, [wsUrl]);

  const updateWebSocketUrl = (url) => {
    setWsUrl(url);
  };

  return (
    <WebSocketContext.Provider value={{ wsClient: wsClientRef.current, isConnected, updateWebSocketUrl }}>
      {children}
    </WebSocketContext.Provider>
  );
};

export const useWebSocket = () => useContext(WebSocketContext);
