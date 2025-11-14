import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { FileText, MessageCircle, Send, ArrowLeft, Loader2 } from 'lucide-react';
import api from '../api/axios';

const SessionDetail = () => {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  const [session, setSession] = useState(null);
  const [messages, setMessages] = useState([]);
  const [userMessage, setUserMessage] = useState('');
  const [loading, setLoading] = useState(true);
  const [chatLoading, setChatLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchSessionDetail();
  }, [sessionId]);

  const fetchSessionDetail = async () => {
    try {
      const response = await api.get(`/api/documents/sessions/${sessionId}/`);
      setSession(response.data.session);
      setMessages(response.data.messages);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to fetch session details');
    } finally {
      setLoading(false);
    }
  };

  const handleSendMessage = async () => {
    if (!userMessage.trim()) return;

    const newMessage = { message: userMessage, is_user: true };
    setMessages([...messages, newMessage]);
    setUserMessage('');
    setChatLoading(true);

    try {
      const response = await api.post('/api/documents/chat/', {
        message: userMessage,
        session_id: sessionId,
      });

      const aiMessage = { message: response.data.response, is_user: false };
      setMessages(prev => [...prev, aiMessage]);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to send message');
    } finally {
      setChatLoading(false);
    }
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <Loader2 className="w-12 h-12 animate-spin text-indigo-600" />
      </div>
    );
  }

  if (error && !session) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-12 px-4">
        <div className="max-w-4xl mx-auto">
          <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-red-700">
            {error}
          </div>
          <button
            onClick={() => navigate('/document-sessions')}
            className="mt-4 text-indigo-600 hover:text-indigo-700 flex items-center"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Sessions
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <button
          onClick={() => navigate('/document-sessions')}
          className="mb-6 text-indigo-600 hover:text-indigo-700 flex items-center font-medium"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Sessions
        </button>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Document Summary Section */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-semibold flex items-center">
                <FileText className="mr-2" />
                Document Summary
              </h2>
              <span className="text-sm text-gray-500">
                {formatDate(session.created_at)}
              </span>
            </div>

            <div className="mb-6">
              <h3 className="text-lg font-semibold mb-3">Summary</h3>
              <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                <p className="text-gray-700 whitespace-pre-wrap">{session.summary}</p>
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold mb-3">Document Preview</h3>
              <div className="bg-gray-50 p-4 rounded-lg border border-gray-200 max-h-64 overflow-y-auto">
                <p className="text-gray-700 whitespace-pre-wrap text-sm">
                  {session.document_text}
                </p>
              </div>
            </div>
          </div>

          {/* Chat Section */}
          <div className="bg-white rounded-lg shadow-lg p-6 flex flex-col">
            <h2 className="text-2xl font-semibold mb-6 flex items-center">
              <MessageCircle className="mr-2" />
              Conversation
            </h2>

            <div className="flex-1 overflow-y-auto mb-4 space-y-4 max-h-96">
              {messages.length === 0 ? (
                <div className="text-center text-gray-400 py-8">
                  No messages yet. Start a conversation!
                </div>
              ) : (
                messages.map((msg, index) => (
                  <div
                    key={index}
                    className={`flex ${msg.is_user ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                        msg.is_user
                          ? 'bg-indigo-600 text-white'
                          : 'bg-gray-200 text-gray-800'
                      }`}
                    >
                      <p className="whitespace-pre-wrap">{msg.message}</p>
                    </div>
                  </div>
                ))
              )}
              {chatLoading && (
                <div className="flex justify-start">
                  <div className="bg-gray-200 text-gray-800 px-4 py-2 rounded-lg">
                    <Loader2 className="animate-spin" />
                  </div>
                </div>
              )}
            </div>

            {error && (
              <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
                {error}
              </div>
            )}

            <div className="flex gap-2">
              <input
                type="text"
                value={userMessage}
                onChange={(e) => setUserMessage(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                placeholder="Ask a question about the document..."
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
              <button
                onClick={handleSendMessage}
                disabled={!userMessage.trim() || chatLoading}
                className="bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SessionDetail;
