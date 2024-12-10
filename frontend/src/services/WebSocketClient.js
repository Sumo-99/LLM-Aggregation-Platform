class WebSocketClient {
    constructor(url) {
      this.url = url;
      this.ws = null;
      this.eventHandlers = {};
      this.connect();
    }
  
    connect() {
      console.log('Attempting to connect to WebSocket...');
      this.ws = new WebSocket(this.url);
  
      this.ws.onopen = () => {
        console.log('WebSocket connected');
        if (this.eventHandlers.open) this.eventHandlers.open();
      };
  
      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('Message received:', data);
          if (this.eventHandlers.message) this.eventHandlers.message(data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
  
      this.ws.onclose = () => {
        console.log('WebSocket disconnected');
        if (this.eventHandlers.close) this.eventHandlers.close();
      };
  
      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        if (this.eventHandlers.error) this.eventHandlers.error(error);
      };
    }
  
    onOpen(handler) {
      this.eventHandlers.open = handler;
    }
  
    onMessage(handler) {
      this.eventHandlers.message = handler;
    }
  
    onClose(handler) {
      this.eventHandlers.close = handler;
    }
  
    onError(handler) {
      this.eventHandlers.error = handler;
    }
  
    sendMessage(message) {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        console.log('Sending message:', message);
        this.ws.send(JSON.stringify(message));
      } else {
        console.error('WebSocket is not connected');
      }
    }
  
    closeConnection() {
      if (this.ws) this.ws.close();
    }
  }
  
  export default WebSocketClient;
  