import React, { createContext, useState, useContext, useEffect } from 'react';
import { authAPI } from '../api/auth';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // Initialize auth state from localStorage
  useEffect(() => {
    const initAuth = async () => {
      const storedUser = localStorage.getItem('user');
      const token = localStorage.getItem('access_token');
      
      if (storedUser && token) {
        try {
          setUser(JSON.parse(storedUser));
          setIsAuthenticated(true);
          
          // Verify token by fetching profile
          const profile = await authAPI.getProfile();
          setUser(profile);
          localStorage.setItem('user', JSON.stringify(profile));
        } catch (error) {
          // Token invalid, clear auth
          logout();
        }
      }
      setLoading(false);
    };

    initAuth();
  }, []);

  const login = async (credentials) => {
    try {
      const data = await authAPI.login(credentials);
      
      // Check if OTP verification is required
      if (data.requires_verification) {
        return { 
          success: true, 
          requiresVerification: true,
          email: data.email,
          message: data.message 
        };
      }
      
      // Store tokens and user data
      localStorage.setItem('access_token', data.tokens.access);
      localStorage.setItem('refresh_token', data.tokens.refresh);
      localStorage.setItem('user', JSON.stringify(data.user));
      
      setUser(data.user);
      setIsAuthenticated(true);
      
      return { success: true, data };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.error || 'Login failed',
      };
    }
  };

  const signup = async (userData) => {
    try {
      const data = await authAPI.signup(userData);
      
      // Check if OTP verification is required
      if (data.requires_verification) {
        return { 
          success: true, 
          requiresVerification: true,
          email: data.email,
          message: data.message 
        };
      }
      
      // If no verification needed (shouldn't happen for manual signup)
      localStorage.setItem('access_token', data.tokens.access);
      localStorage.setItem('refresh_token', data.tokens.refresh);
      localStorage.setItem('user', JSON.stringify(data.user));
      
      setUser(data.user);
      setIsAuthenticated(true);
      
      return { success: true, data };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data || 'Signup failed',
      };
    }
  };

  const googleLogin = async (credentialResponse) => {
    try {
      // credentialResponse.credential contains the JWT id_token from Google
      const data = await authAPI.googleAuth(credentialResponse.credential);
      
      // Store tokens and user data
      localStorage.setItem('access_token', data.tokens.access);
      localStorage.setItem('refresh_token', data.tokens.refresh);
      localStorage.setItem('user', JSON.stringify(data.user));
      
      setUser(data.user);
      setIsAuthenticated(true);
      
      return { success: true, data };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.error || 'Google authentication failed',
      };
    }
  };

  const logout = async () => {
    try {
      const refreshToken = localStorage.getItem('refresh_token');
      if (refreshToken) {
        await authAPI.logout(refreshToken);
      }
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      // Clear local storage and state
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
      localStorage.removeItem('user');
      setUser(null);
      setIsAuthenticated(false);
    }
  };

  const verifyOTP = async (otpData) => {
    try {
      const data = await authAPI.verifyOTP(otpData);
      
      // Store tokens and user data after successful OTP verification
      localStorage.setItem('access_token', data.tokens.access);
      localStorage.setItem('refresh_token', data.tokens.refresh);
      localStorage.setItem('user', JSON.stringify(data.user));
      
      setUser(data.user);
      setIsAuthenticated(true);
      
      return { success: true, data };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.error || 'OTP verification failed',
      };
    }
  };

  const resendOTP = async (emailData) => {
    try {
      const data = await authAPI.resendOTP(emailData);
      return { success: true, message: data.message };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.error || 'Failed to resend OTP',
      };
    }
  };

  const value = {
    user,
    loading,
    isAuthenticated,
    login,
    signup,
    googleLogin,
    logout,
    verifyOTP,
    resendOTP,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
