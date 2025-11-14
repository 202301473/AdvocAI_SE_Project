"""
Script to view all users in MongoDB
Run this with: python view_users.py
"""

import os
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AdvocAi.settings')
django.setup()

from authentication.models import User
from django.utils import timezone

def view_all_users():
    """Display all users from MongoDB"""
    users = User.objects.all()
    
    print("\n" + "="*80)
    print(f"TOTAL USERS IN DATABASE: {users.count()}")
    print("="*80 + "\n")
    
    if users.count() == 0:
        print("No users found in database.")
        print("Create a user by:")
        print("1. Signing up on the website")
        print("2. Running: python manage.py createsuperuser")
        return
    
    for i, user in enumerate(users, 1):
        print(f"USER #{i}")
        print("-" * 80)
        print(f"ID:            {user.id}")
        print(f"Email:         {user.email}")
        print(f"Username:      {user.username}")
        print(f"Name:          {user.name}")
        print(f"Auth Provider: {user.auth_provider}")
        print(f"Is Active:     {user.is_active}")
        print(f"Is Staff:      {user.is_staff}")
        print(f"Is Superuser:  {user.is_superuser}")
        print(f"Date Joined:   {user.date_joined}")
        print(f"Last Login:    {user.last_login}")
        if user.google_id:
            print(f"Google ID:     {user.google_id}")
        print("-" * 80 + "\n")

def view_by_provider():
    """View users grouped by authentication provider"""
    email_users = User.objects.filter(auth_provider='email')
    google_users = User.objects.filter(auth_provider='google')
    
    print("\n" + "="*80)
    print("USERS BY AUTHENTICATION PROVIDER")
    print("="*80 + "\n")
    
    print(f"📧 Email/Password Users: {email_users.count()}")
    for user in email_users:
        print(f"   - {user.email} ({user.name})")
    
    print(f"\n🔵 Google OAuth Users: {google_users.count()}")
    for user in google_users:
        print(f"   - {user.email} ({user.name})")
    print()

def view_superusers():
    """View all superusers"""
    superusers = User.objects.filter(is_superuser=True)
    
    print("\n" + "="*80)
    print(f"SUPERUSERS: {superusers.count()}")
    print("="*80 + "\n")
    
    if superusers.count() == 0:
        print("No superusers found.")
        print("Create one with: python manage.py createsuperuser")
        return
    
    for user in superusers:
        print(f"👑 {user.email} ({user.username})")
    print()

if __name__ == "__main__":
    print("\n" + "="*80)
    print("ADVOCAI - MONGODB USER VIEWER")
    print("="*80)
    
    try:
        view_all_users()
        view_by_provider()
        view_superusers()
        
        print("\n" + "="*80)
        print("To view in MongoDB Compass:")
        print("1. Open MongoDB Compass")
        print("2. Connect to: mongodb://localhost:27017")
        print("3. Database: advocai_db")
        print("4. Collection: users")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("1. MongoDB is running")
        print("2. You've run migrations: python manage.py migrate")
        print("3. Virtual environment is activated")
