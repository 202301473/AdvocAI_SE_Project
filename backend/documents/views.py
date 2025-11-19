import json
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import JSONParser

from documents.mongo_client import get_all_conversations, get_conversation_by_id, save_conversation, update_conversation, delete_conversation, get_document_version_content, delete_document_version, update_share_permissions, update_user_share_permissions
from channels.layers import get_channel_layer # Import get_channel_layer
from asgiref.sync import async_to_sync # Import async_to_sync
from ai_generator.utils import get_gemini_response # Import the AI generation function


from .comment_mongo_client import get_comments_for_document, add_comment, serialize_comment


@api_view(['GET', 'POST'])
def conversation_list(request):
    """
    List all conversations or create a new one.
    """
    if request.method == 'GET':
        user_id = request.user.username if request.user.is_authenticated else None
        if user_id:
            conversations = get_all_conversations(user=user_id)
        else:
            # If user is not authenticated, they should not see any conversations
            conversations = []
        return Response(conversations)

    elif request.method == 'POST':
        title = request.data.get('title')
        messages = request.data.get('messages')
        initial_document_content = request.data.get('initial_document_content')
        notes = request.data.get('notes', 'Initial Version')
        shared_with_users = request.data.get('shared_with_users', []) # New: get shared_with_users

        print(f"[DEBUG Backend] conversation_list (POST) - Received messages: {messages}")

        if not title or not messages:
            return Response({'error': 'Title and messages are required'}, status=status.HTTP_400_BAD_REQUEST)
        
        conversation_id = save_conversation(
            title=title,
            messages=messages,
            initial_document_content=initial_document_content,
            uploaded_by=(request.user.username if request.user.is_authenticated else 'anonymous'),
            notes=notes,
            shared_with_users=shared_with_users # New: pass shared_with_users
        )
        if conversation_id:
            return Response({'id': conversation_id}, status=status.HTTP_201_CREATED)
        else:
            return Response({'error': 'Failed to save conversation'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

