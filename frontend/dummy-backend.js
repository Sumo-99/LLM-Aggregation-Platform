const WebSocket = require('ws');

const server = new WebSocket.Server({ port: 8080 });

console.log('WebSocket server running on ws://localhost:8080');

server.on('connection', (ws) => {
  console.log('Client connected');

  ws.on('message', (message) => {
    console.log('Received:', message);
    // Echo back the received message with a dummy response
    const response = {
      status: 'success',
      received: JSON.parse(message),
      reply: 'This is a dummy response.',
    };
    ws.send(JSON.stringify(response));
  });

  ws.on('close', () => console.log('Client disconnected'));
});
