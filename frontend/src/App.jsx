import React from "react";
import { BrowserRouter as Router, Routes, Route, useLocation } from "react-router-dom";
import { GoogleOAuthProvider } from '@react-oauth/google';
import { AuthProvider } from "./context/AuthContext";
import Navbar from "./Components/Navbar/Navbar";
import Home from "./pages/Home";
import Signup from "./pages/Signup";
import Login from "./pages/Login";
import VerifyOTP from "./pages/VerifyOTP";
import Lawyers from "./pages/Lawyers";
import DocumentAnalyzer from "./pages/DocumentAnalyzer";
import DocumentCreation from "./pages/DocumentCreation";
import LawyerConnect from "./pages/LawyerConnect";
import MyDocuments from "./pages/MyDocuments";

const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID || '';

function AppContent() {
  const location = useLocation();
  const hideNavbarRoutes = ['/login', '/signup', '/verify-otp'];
  const shouldShowNavbar = !hideNavbarRoutes.includes(location.pathname);

  return (
      <>
        {/* Conditionally render navbar */}
        {shouldShowNavbar && <Navbar />}
        
        {/* Page routes with conditional top padding for fixed navbar */}
        <main className={shouldShowNavbar ? "pt-20 bg-gray-50" : "bg-gray-50"}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/document-analyser" element={<DocumentAnalyzer />} />
          <Route path="/document-creation" element={<DocumentCreation />} />
          <Route path="/lawyer-connect" element={<LawyerConnect />} />
          <Route path="/my-documents" element={<MyDocuments />} />
          <Route path="/lawyers" element={<Lawyers />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/verify-otp" element={<VerifyOTP />} />
        </Routes>
      </main>
    </>
  );
}

function App() {
  const appContent = (
    <Router>
      <AuthProvider>
        <AppContent />
      </AuthProvider>
    </Router>
  );

  // Only wrap with GoogleOAuthProvider if client ID is available
  if (GOOGLE_CLIENT_ID) {
    return (
      <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
        {appContent}
      </GoogleOAuthProvider>
    );
  }

  return appContent;
}

export default App;
