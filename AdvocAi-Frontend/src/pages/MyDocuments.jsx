import React, { useState } from 'react';
import { FileText, Plus, Search, Eye, Edit, Download } from 'lucide-react';

const MyDocuments = () => {
  const [searchTerm, setSearchTerm] = useState('');

  // Sample document data - you can replace this with data from your backend
  const documents = [
    {
      id: 1,
      title: "Document 1 - 1",
      details: "Contract - Dec 15, 2024",
      status: "Signed",
      statusColor: "bg-green-500"
    },
    {
      id: 2,
      title: "Document 1 - 2",
      details: "Agreement - Dec 14, 2024",
      status: "Pending",
      statusColor: "bg-yellow-400"
    },
    {
      id: 3,
      title: "Document 1 - 3",
      details: "Contract - Dec 13, 2024",
      status: "Signed",
      statusColor: "bg-green-500"
    },
    {
      id: 4,
      title: "Document 1 - 4",
      details: "Agreement - Dec 12, 2024",
      status: "Signed",
      statusColor: "bg-green-500"
    },
    {
      id: 5,
      title: "Document 1 - 5",
      details: "Contract - Dec 11, 2024",
      status: "Signed",
      statusColor: "bg-green-500"
    }
  ];

  const handleNewDocument = () => {
    console.log('Creating new document');
    // Here you would typically navigate to document creation or open a modal
  };

  const handleView = (documentId) => {
    console.log('Viewing document:', documentId);
    // Here you would typically open a document viewer or navigate to document details
  };

  const handleEdit = (documentId) => {
    console.log('Editing document:', documentId);
    // Here you would typically navigate to document editor
  };

  const handleDownload = (documentId) => {
    console.log('Downloading document:', documentId);
    // Here you would typically trigger a download
  };

  const filteredDocuments = documents.filter(doc =>
    doc.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
    doc.details.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-green-50">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header Section */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center">
            <FileText className="w-8 h-8 text-gray-400 mr-4" />
            <h1 className="text-4xl font-bold text-gray-900">My Documents</h1>
          </div>
          <button
            onClick={handleNewDocument}
            className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors duration-200"
          >
            <Plus className="w-5 h-5" />
            <span>New Document</span>
          </button>
        </div>

        {/* Search Bar */}
        <div className="mb-8">
          <div className="relative max-w-md">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search documents..."
              className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-600 focus:border-transparent outline-none"
            />
          </div>
        </div>
      </div>

      {/* Documents List - Full Width White Background */}
      <div className="bg-white min-h-screen">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="overflow-hidden">
            {filteredDocuments.length > 0 ? (
              <div className="divide-y divide-gray-200">
                {filteredDocuments.map((document) => (
                  <div
                    key={document.id}
                    className="p-6 hover:bg-gray-50 transition-colors duration-200"
                  >
                    <div className="flex items-center justify-between">
                      {/* Left Section - Document Info */}
                      <div className="flex items-center space-x-4 flex-1">
                        <FileText className="w-8 h-8 text-gray-400 flex-shrink-0" />
                        <div className="flex-1 min-w-0">
                          <h3 className="text-lg font-bold text-gray-900 truncate">
                            {document.title}
                          </h3>
                          <p className="text-sm text-gray-600 mt-1">
                            {document.details}
                          </p>
                        </div>
                      </div>

                      {/* Right Section - Status and Actions */}
                      <div className="flex items-center space-x-4">
                        {/* Status Tag */}
                        <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                          document.statusColor
                        } ${document.status === 'Signed' ? 'text-white' : 'text-black'}`}>
                          {document.status}
                        </span>

                        {/* Action Buttons */}
                        <div className="flex items-center space-x-2">
                          <button
                            onClick={() => handleView(document.id)}
                            className="flex items-center space-x-1 px-3 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors duration-200"
                          >
                            <Eye className="w-4 h-4 text-gray-600" />
                            <span className="text-sm text-gray-700">View</span>
                          </button>
                          <button
                            onClick={() => handleEdit(document.id)}
                            className="flex items-center space-x-1 px-3 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors duration-200"
                          >
                            <Edit className="w-4 h-4 text-gray-600" />
                            <span className="text-sm text-gray-700">Edit</span>
                          </button>
                          <button
                            onClick={() => handleDownload(document.id)}
                            className="flex items-center space-x-1 px-3 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors duration-200"
                          >
                            <Download className="w-4 h-4 text-gray-600" />
                            <span className="text-sm text-gray-700">Download</span>
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="p-12 text-center">
                <FileText className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No documents found</h3>
                <p className="text-gray-600">
                  {searchTerm ? 'Try adjusting your search terms' : 'Create your first document to get started'}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MyDocuments;
