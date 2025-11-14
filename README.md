# AdvocAI Project Setup Guide

This repository contains both the frontend and backend components of the AdvocAI project.

## Project Structure
```
AdvocAI_SE_Project/
├── AdvocAi-Backend/    # Django backend
└── AdvocAi-Frontend/   # React frontend
```

## Prerequisites
- Python 3.8 or higher
- Node.js 16.x or higher
- npm 8.x or higher
- Git

## Backend Setup (Django)

1. Navigate to the backend directory:
   ```bash
   cd AdvocAi-Backend/AdvocAi
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - Linux/MacOS:
     ```bash
     source venv/bin/activate
     ```

4. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

5. Create a `.env` file in the AdvocAi directory with the following content:
   ```
   SECRET_KEY=your_secret_key_here
   DEBUG=True
   ALLOWED_HOSTS=localhost,127.0.0.1
   ```

6. Run database migrations:
   ```bash
   python manage.py migrate
   ```

7. Start the Django development server:
   ```bash
   python manage.py runserver
   ```
   The backend server will run at http://localhost:8000

## Frontend Setup (React)

1. Navigate to the frontend directory:
   ```bash
   cd AdvocAi-Frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```
   The frontend development server will run at http://localhost:5173

## Running the Complete Project

1. Start the backend server (in one terminal):
   ```bash
   cd AdvocAi-Backend/AdvocAi
   python manage.py runserver
   ```

2. Start the frontend development server (in another terminal):
   ```bash
   cd AdvocAi-Frontend
   npm run dev
   ```

3. Access the application at http://localhost:5173

## Additional Notes

- Make sure both backend and frontend servers are running simultaneously
- The backend API will be accessible at http://localhost:8000
- For development purposes, the frontend will proxy API requests to the backend
- Make sure all required ports (8000 for backend, 5173 for frontend) are available

## Troubleshooting

If you encounter any issues:

1. Make sure all dependencies are installed correctly
2. Check if the virtual environment is activated for the backend
3. Verify that the `.env` file is properly configured
4. Ensure no other services are using the required ports
5. Clear browser cache if facing frontend issues

## Development

- Backend API documentation will be available at http://localhost:8000/api/docs/
- The project uses ESLint for JavaScript/React code linting
- Follow the project's coding standards and git workflow
