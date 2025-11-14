import React, { useState } from 'react';
import { FileText, PenTool, Send } from 'lucide-react';

const DocumentCreation = () => {
  const [formData, setFormData] = useState({
    documentTitle: '',
    firstParty: '',
    documentContent: ''
  });
  const [chatMessage, setChatMessage] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);

  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleGenerateDocument = async () => {
    if (!formData.documentTitle || !formData.firstParty || !formData.documentContent) {
      alert('Please fill in all required fields');
      return;
    }

    setIsGenerating(true);
    
    // Simulate API call
    setTimeout(() => {
      setIsGenerating(false);
      alert('Document generated successfully! (This would integrate with your backend)');
    }, 2000);
  };

  const handleSendMessage = () => {
    if (chatMessage.trim()) {
      console.log('Chat message:', chatMessage);
      setChatMessage('');
      // Here you would typically send the message to your AI backend
    }
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter') {
      handleSendMessage();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-8">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Main Content Card */}
        <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
          {/* Header Section */}
          <div className="text-center mb-8">
            <div className="flex items-center justify-center mb-4">
              <div className="relative">
                <FileText className="w-10 h-10 text-blue-600 mr-3" />
                <PenTool className="w-5 h-5 text-yellow-500 absolute -bottom-1 -right-1" />
              </div>
              <h1 className="text-4xl font-bold text-gray-900">Create New Document</h1>
            </div>
            <p className="text-xl text-gray-600">
              Build your legal document from scratch with AI assistance
            </p>
          </div>

          {/* Form Fields */}
          <div className="space-y-6">
            {/* Document Title */}
            <div>
              <label htmlFor="documentTitle" className="block text-lg font-semibold text-gray-700 mb-3">
                Document Title
              </label>
              <input
                id="documentTitle"
                type="text"
                value={formData.documentTitle}
                onChange={(e) => handleInputChange('documentTitle', e.target.value)}
                placeholder="Enter your document title here..."
                className="w-full px-6 py-4 text-lg border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-600 focus:border-transparent outline-none transition-all duration-200"
              />
            </div>

            {/* First Party */}
            <div>
              <label htmlFor="firstParty" className="block text-lg font-semibold text-gray-700 mb-3">
                First Party
              </label>
              <input
                id="firstParty"
                type="text"
                value={formData.firstParty}
                onChange={(e) => handleInputChange('firstParty', e.target.value)}
                placeholder="Enter name of first party (person or organization)"
                className="w-full px-6 py-4 text-lg border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-600 focus:border-transparent outline-none transition-all duration-200"
              />
            </div>

            {/* Document Content */}
            <div>
              <label htmlFor="documentContent" className="block text-lg font-semibold text-gray-700 mb-3">
                Document Content
              </label>
              <textarea
                id="documentContent"
                value={formData.documentContent}
                onChange={(e) => handleInputChange('documentContent', e.target.value)}
                placeholder="Describe what you want in your document..."
                rows={8}
                className="w-full px-6 py-4 text-lg border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-600 focus:border-transparent outline-none transition-all duration-200 resize-none"
              />
            </div>
          </div>

          {/* Generate Button */}
          <div className="mt-8">
            <button
              onClick={handleGenerateDocument}
              disabled={isGenerating}
              className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 disabled:from-gray-400 disabled:to-gray-500 text-white font-semibold py-4 px-8 rounded-xl text-lg transition-all duration-200 transform hover:scale-105 disabled:scale-100 disabled:cursor-not-allowed"
            >
              {isGenerating ? 'Generating Document...' : 'Generate Document'}
            </button>
          </div>
        </div>

        {/* Chat Interface */}
        <div className="bg-white rounded-2xl shadow-xl p-6">
          <div className="flex space-x-4">
            <input
              type="text"
              value={chatMessage}
              onChange={(e) => setChatMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask any related questions here..."
              className="flex-1 px-6 py-4 text-lg border border-blue-200 rounded-xl focus:ring-2 focus:ring-blue-600 focus:border-transparent outline-none bg-blue-50"
            />
            <button
              onClick={handleSendMessage}
              className="px-6 py-4 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-colors flex items-center space-x-2"
            >
              <Send className="w-5 h-5" />
              <span>Send</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DocumentCreation;
