import React from 'react';
import './SplitScreen.css';

const SplitScreen = ({ models }) => {
  return (
    <div className="split-screen">
      {models.map((model, index) => (
        <div key={index} className="model-container">
          <h3>{model.name}</h3>
          <div className="model-content">{model.content}</div>
        </div>
      ))}
    </div>
  );
};

export default SplitScreen;
