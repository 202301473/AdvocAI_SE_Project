import React, { useState, useEffect } from 'react';
import { Upload, Bot, MessageCircle, Send, FileText, Search, Loader2, History, X, ChevronRight } from 'lucide-react';
import { uploadDocument, sendChatMessage, getUserSessions, getChatHistory } from '../utils/api';

const DocumentAnalyzer = () => {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [summary, setSummary] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [chatMessage, setChatMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const [sessions, setSessions] = useState([]);
  const [loadingSessions, setLoadingSessions] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setUploadedFile(file);
    setError('');
    setUploading(true);
    setSummary('');

    try {
      const response = await uploadDocument(file);
      
      if (response.success) {
        setSummary(response.summary);
        setSessionId(response.session_id);
        // Initialize chat with welcome message
        setChatHistory([{
          id: 1,
          sender: 'AdvocAI',
          message: "Hi! I've analyzed your document. Feel free to ask me any questions about the terms, risks, or anything else you'd like to understand better.",
          timestamp: new Date().toLocaleTimeString()
        }]);
      }
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Failed to upload document');
      setUploadedFile(null);
    } finally {
      setUploading(false);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const files = event.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      const input = document.getElementById('file-upload');
      if (input) {
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        input.files = dataTransfer.files;
        handleFileUpload({ target: { files: dataTransfer.files } });
      }
    }
  };

  const handleSendMessage = async () => {
    if (!chatMessage.trim() || !sessionId) return;

    const userMessage = chatMessage.trim();
    setChatMessage('');
    setLoading(true);

    // Add user message to chat
    const newUserMessage = {
      id: Date.now(),
      sender: 'User',
      message: userMessage,
      timestamp: new Date().toLocaleTimeString()
    };
    setChatHistory(prev => [...prev, newUserMessage]);

    try {
      const response = await sendChatMessage(sessionId, userMessage);
      
      if (response.response) {
        const aiMessage = {
          id: Date.now() + 1,
          sender: 'AdvocAI',
          message: response.response,
          timestamp: new Date().toLocaleTimeString()
        };
        setChatHistory(prev => [...prev, aiMessage]);
      }
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Failed to send message');
      // Remove user message on error
      setChatHistory(prev => prev.filter(msg => msg.id !== newUserMessage.id));
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter') {
      handleSendMessage();
    }
  };

  // Fetch user sessions on component mount
  useEffect(() => {
    fetchSessions();
  }, []);

  // Fetch sessions when a new document is uploaded
  useEffect(() => {
    if (sessionId) {
      fetchSessions();
    }
  }, [sessionId]);

  const fetchSessions = async () => {
    setLoadingSessions(true);
    try {
      const response = await getUserSessions();
      if (response.sessions) {
        setSessions(response.sessions);
      }
    } catch (err) {
      console.error('Failed to fetch sessions:', err);
    } finally {
      setLoadingSessions(false);
    }
  };

  const handleSessionClick = async (selectedSessionId) => {
    if (selectedSessionId === sessionId) return; // Already loaded
    
    setLoading(true);
    setError('');
    
    try {
      const response = await getChatHistory(selectedSessionId);
      if (response.session && response.messages) {
        setSessionId(response.session.id);
        setSummary(response.session.summary);
        
        // Convert messages to chat history format
        const history = response.messages.map(msg => ({
          id: msg.id,
          sender: msg.is_user ? 'User' : 'AdvocAI',
          message: msg.message,
          timestamp: new Date(msg.timestamp).toLocaleTimeString()
        }));
        
        setChatHistory(history);
        setUploadedFile({ name: 'Previous Document' }); // Placeholder
      }
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Failed to load session');
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now - date);
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    
    if (diffDays === 1) return 'Today';
    if (diffDays === 2) return 'Yesterday';
    if (diffDays < 7) return `${diffDays - 1} days ago`;
    
    return date.toLocaleDateString();
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="flex relative">
        {/* Side Panel - Previous Sessions */}
        <div className={`${sidebarOpen ? 'w-80' : 'w-0'} transition-all duration-300 overflow-hidden border-r border-gray-200 bg-white ${sidebarOpen ? 'px-4 py-6' : 'p-0'} min-h-screen sticky top-20`}>
          {sidebarOpen && (
            <div className="h-full flex flex-col">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center">
                  <History className="w-5 h-5 text-blue-600 mr-2" />
                  <h2 className="text-xl font-bold text-gray-900">Chat History</h2>
                </div>
                <button
                  onClick={() => setSidebarOpen(false)}
                  className="p-1 hover:bg-gray-100 rounded-lg transition-colors"
                >
                  <X className="w-5 h-5 text-gray-500" />
                </button>
              </div>

              <div className="flex-1 overflow-y-auto">
                {loadingSessions ? (
                  <div className="flex items-center justify-center py-8">
                    <Loader2 className="w-6 h-6 animate-spin text-blue-600" />
                  </div>
                ) : sessions.length > 0 ? (
                  <div className="space-y-2">
                    {sessions.map((session) => (
                      <button
                        key={session.id}
                        onClick={() => handleSessionClick(session.id)}
                        className={`w-full text-left p-3 rounded-lg border transition-all ${
                          sessionId === session.id
                            ? 'bg-blue-50 border-blue-200 shadow-sm'
                            : 'bg-gray-50 border-gray-200 hover:bg-gray-100 hover:border-gray-300'
                        }`}
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center mb-1">
                              <FileText className="w-4 h-4 text-gray-400 mr-2 flex-shrink-0" />
                              <p className="text-xs text-gray-500 font-medium truncate">
                                Session #{session.id}
                              </p>
                            </div>
                            <p className="text-sm text-gray-700 line-clamp-2 mb-2">
                              {session.summary_preview}
                            </p>
                            <div className="flex items-center justify-between text-xs text-gray-500">
                              <span>{formatDate(session.created_at)}</span>
                              {session.message_count > 0 && (
                                <span className="bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full">
                                  {session.message_count} msgs
                                </span>
                              )}
                            </div>
                          </div>
                          {sessionId === session.id && (
                            <ChevronRight className="w-5 h-5 text-blue-600 ml-2 flex-shrink-0" />
                          )}
                        </div>
                      </button>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    <History className="w-12 h-12 text-gray-300 mx-auto mb-3" />
                    <p className="text-sm">No previous sessions</p>
                    <p className="text-xs mt-1">Upload a document to get started</p>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Toggle Sidebar Button (when closed) */}
        {!sidebarOpen && (
          <button
            onClick={() => setSidebarOpen(true)}
            className="fixed left-0 top-1/2 -translate-y-1/2 bg-blue-600 text-white p-2 rounded-r-lg shadow-lg hover:bg-blue-700 transition-colors z-10"
          >
            <History className="w-5 h-5" />
          </button>
        )}

        {/* Main Content */}
        <div className={`flex-1 transition-all duration-300`}>
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header Section */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center mb-4">
            <FileText className="w-8 h-8 text-blue-600 mr-3" />
            <h1 className="text-4xl font-bold text-gray-900">AI Document Analyser</h1>
          </div>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Upload your legal document and get instant AI-powered analysis, risk assessment, and plain English explanations of complex legal terms.
          </p>
        </div>

        {/* Main Content Cards */}
        <div className="grid lg:grid-cols-2 gap-8 mb-12">
          {/* Upload Document Card */}
          <div className="bg-white rounded-2xl shadow-lg p-8">
            <div className="flex items-center mb-6">
              <Upload className="w-6 h-6 text-blue-600 mr-3" />
              <h2 className="text-2xl font-bold text-gray-900">Upload Document</h2>
            </div>
            
            <div
              className="border-2 border-dashed border-gray-300 rounded-xl p-12 text-center hover:border-blue-500 transition-colors cursor-pointer"
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onClick={() => document.getElementById('file-upload').click()}
            >
              <input
                id="file-upload"
                type="file"
                accept=".pdf,.docx,.txt"
                onChange={handleFileUpload}
                className="hidden"
                disabled={uploading}
              />
              
              <div className="flex flex-col items-center">
                {uploading ? (
                  <>
                    <Loader2 className="w-16 h-16 text-blue-600 mb-4 animate-spin" />
                    <p className="text-lg font-medium text-gray-700 mb-2">
                      Uploading and analyzing document...
                    </p>
                  </>
                ) : (
                  <>
                    <FileText className="w-16 h-16 text-gray-400 mb-4" />
                    <p className="text-lg font-medium text-gray-700 mb-2">
                      Drag and drop your legal document here
                    </p>
                    <p className="text-sm text-gray-500">
                      or click to browse files (PDF, DOCX, TXT)
                    </p>
                  </>
                )}
              </div>
            </div>

            {error && (
              <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-red-800 text-sm">{error}</p>
              </div>
            )}

            {uploadedFile && !uploading && (
              <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
                <div className="flex items-center">
                  <FileText className="w-5 h-5 text-green-600 mr-2" />
                  <span className="text-green-800 font-medium">{uploadedFile.name}</span>
                </div>
              </div>
            )}
          </div>

          {/* AI Analysis Results Card */}
          <div className="bg-white rounded-2xl shadow-lg p-8">
            <div className="flex items-center mb-6">
              <Bot className="w-6 h-6 text-blue-600 mr-3" />
              <h2 className="text-2xl font-bold text-gray-900">AI Analysis Results</h2>
            </div>
            
            {summary ? (
              <div className="max-h-64 overflow-y-auto">
                <div className="prose max-w-none">
                  <p className="text-gray-700 whitespace-pre-wrap leading-relaxed">{summary}</p>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-64 text-center">
                <Search className="w-20 h-20 text-gray-300 mb-4" />
                <p className="text-gray-500 text-lg">
                  Upload a document to see AI-powered analysis results here
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Chat Interface */}
        <div className="bg-white rounded-2xl shadow-lg p-8">
          <div className="flex items-center mb-6">
            <MessageCircle className="w-6 h-6 text-blue-600 mr-3" />
            <h2 className="text-2xl font-bold text-gray-900">Ask Questions About Your Document</h2>
          </div>

          {/* Chat Messages */}
          <div className="space-y-4 mb-6 max-h-96 overflow-y-auto">
            {chatHistory.length > 0 ? (
              chatHistory.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.sender === 'User' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-xs lg:max-w-md px-4 py-3 rounded-lg ${
                      message.sender === 'User'
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-200 text-gray-800'
                    }`}
                  >
                    <p className="text-sm whitespace-pre-wrap">{message.message}</p>
                    <p className={`text-xs mt-1 ${
                      message.sender === 'User' ? 'text-blue-100' : 'text-gray-500'
                    }`}>
                      {message.timestamp}
                    </p>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center text-gray-500 py-8">
                Upload and analyze a document to start chatting
              </div>
            )}
            {loading && (
              <div className="flex justify-start">
                <div className="bg-gray-200 text-gray-800 px-4 py-3 rounded-lg">
                  <Loader2 className="w-5 h-5 animate-spin inline mr-2" />
                  <span className="text-sm">Thinking...</span>
                </div>
              </div>
            )}
          </div>

          {/* Chat Input */}
          <div className="flex space-x-4">
            <input
              type="text"
              value={chatMessage}
              onChange={(e) => setChatMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={sessionId ? "Ask me anything about your document...." : "Upload a document first..."}
              disabled={!sessionId || loading}
              className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-600 focus:border-transparent outline-none disabled:bg-gray-100 disabled:cursor-not-allowed"
            />
            <button
              onClick={handleSendMessage}
              disabled={!sessionId || !chatMessage.trim() || loading}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <>
                  <Send className="w-5 h-5" />
                  <span>Send</span>
                </>
              )}
            </button>
          </div>
        </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DocumentAnalyzer;