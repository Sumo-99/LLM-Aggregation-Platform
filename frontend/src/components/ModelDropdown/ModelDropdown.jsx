import React from 'react';

const ModelDropdown = ({ models, onChange }) => {
  return (
    <select onChange={(e) => onChange(e.target.value)}>
      {models.map((model, index) => (
        <option key={index} value={model.name}>
          {model.name}
        </option>
      ))}
    </select>
  );
};

export default ModelDropdown;
