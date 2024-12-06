import React from 'react';
import './SearchBar.css';

const SearchBar = ({ onSearch }) => {
  return (
    <div className="search-bar">
      <input
        type="text"
        placeholder="Search..."
        onKeyPress={(e) => {
          if (e.key === 'Enter') onSearch(e.target.value);
        }}
      />
    </div>
  );
};

export default SearchBar;
