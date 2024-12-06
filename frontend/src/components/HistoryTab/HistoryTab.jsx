import React from 'react';
import './HistoryTab.css';

const HistoryTab = ({ history, onSelect }) => {
  return (
    <div className="history-tab">
      <h4>History</h4>
      <ul>
        {history.map((item, index) => (
          <li key={index} onClick={() => onSelect(item)}>
            {item}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default HistoryTab;
