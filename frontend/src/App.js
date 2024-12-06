import React from 'react';
import './App.css';
import MainLayout from './layouts/MainLayout';
import { WebSocketProvider } from './services/WebSocketContext';

const App = () => {
  return (
    <WebSocketProvider>
      <MainLayout />
    </WebSocketProvider>
  );
};

export default App;


