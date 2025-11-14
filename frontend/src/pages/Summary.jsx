import React, { useState } from 'react';
import { FileText, Upload, Download } from 'lucide-react';

const Summary = () => {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [summary, setSummary] = useState('');

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setUploadedFile(file);
      // Here you would typically send the file to your backend for summarization
      console.log('File uploaded:', file.name);
      // Mock summary for now
      setSummary('Document summary will appear here after processing...');
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">Document Summary</h1>
          <p className="text-gray-600">Upload a legal document to get an AI-powered summary</p>
        </div>

        <div className="bg-white rounded-2xl shadow-lg p-8 mb-6">
          <div className="flex items-center mb-6">
            <FileText className="w-6 h-6 text-blue-600 mr-3" />
            <h2 className="text-2xl font-bold text-gray-900">Upload Document</h2>
          </div>

          <div
            className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center hover:border-blue-500 transition-colors cursor-pointer"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const files = e.dataTransfer.files;
              if (files.length > 0) {
                setUploadedFile(files[0]);
              }
            }}
          >
            <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-600 mb-2">
              Drag and drop a document here, or click to browse
            </p>
            <input
              type="file"
              accept=".pdf,.doc,.docx"
              onChange={handleFileUpload}
              className="hidden"
              id="file-upload"
            />
            <label
              htmlFor="file-upload"
              className="inline-block mt-4 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 cursor-pointer"
            >
              Choose File
            </label>
          </div>

          {uploadedFile && (
            <div className="mt-4 p-4 bg-gray-50 rounded-lg">
              <p className="text-gray-700">
                <strong>Uploaded:</strong> {uploadedFile.name}
              </p>
            </div>
          )}
        </div>

        {summary && (
          <div className="bg-white rounded-2xl shadow-lg p-8">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-900">Summary</h2>
              <button className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 flex items-center space-x-2">
                <Download className="w-5 h-5" />
                <span>Download</span>
              </button>
            </div>
            <div className="prose max-w-none">
              <p className="text-gray-700 leading-relaxed">{summary}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Summary;

