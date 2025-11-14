import pymongo
from django.conf import settings

# MongoDB connection
def get_mongo_db():
    """
    Get MongoDB database instance
    Usage:
    from mongo_utils import get_mongo_db
    db = get_mongo_db()
    users = db['users'].find()
    """
    return settings.MONGO_DB

def get_mongo_collection(collection_name):
    """
    Get a MongoDB collection
    """
    return settings.MONGO_DB[collection_name]

def create_user(email, name, auth_provider="email", google_id=None):
    """
    Create a user directly in MongoDB
    """
    users_col = get_mongo_collection('users')
    user_data = {
        'email': email,
        'name': name,
        'auth_provider': auth_provider,
        'is_active': True,
        'is_staff': False,
        'is_superuser': False,
        'date_joined': datetime.utcnow(),
        'last_login': None
    }
    if google_id:
        user_data['google_id'] = google_id
    
    result = users_col.insert_one(user_data)
    return result.inserted_id
