import axios from 'axios';

export const fetchModelResponse = async (modelName, prompt) => {
  const response = await axios.post('/api/model', { modelName, prompt });
  return response.data;
};