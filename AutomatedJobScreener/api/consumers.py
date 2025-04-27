"""
WebSocket consumers for handling real-time communication.
"""
import json
import asyncio
import logging
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.conf import settings

logger = logging.getLogger(__name__)

class InterviewConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for handling real-time interview sessions.
    This replaces the Streamlit interview_page.py functionality.
    """
    async def connect(self):
        self.interview_id = self.scope['url_route']['kwargs']['interview_id']
        self.interview_group_name = f'interview_{self.interview_id}'
        
        # Join the interview group
        await self.channel_layer.group_add(
            self.interview_group_name,
            self.channel_name
        )
        await self.accept()
        await self.send(text_data=json.dumps({
            'type': 'connection_established',
            'message': 'Connected to interview session'
        }))

    async def disconnect(self, close_code):
        # Leave the interview group
        await self.channel_layer.group_discard(
            self.interview_group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        """
        Handle incoming WebSocket messages for interview features.
        """
        try:
            data = json.loads(text_data)
            message_type = data.get('type', '')
            
            if message_type == 'generate_questions':
                # Generate interview questions
                resume_id = data.get('resume_id')
                job_id = data.get('job_id')
                
                # This would call the question generation logic
                # We'll implement this later
                
                await self.send(text_data=json.dumps({
                    'type': 'questions_generated',
                    'message': 'Questions generated successfully',
                    'questions': []  # This will be populated with actual questions
                }))
                
            elif message_type == 'record_response':
                # Handle recording response
                await self.send(text_data=json.dumps({
                    'type': 'recording_started',
                    'message': 'Voice recording started'
                }))
                
        except Exception as e:
            logger.error(f"Error in InterviewConsumer: {str(e)}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Error: {str(e)}'
            }))

    async def interview_message(self, event):
        """
        Handle messages from the interview group.
        """
        await self.send(text_data=json.dumps(event))


class VoiceRecordingConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for handling voice recording sessions.
    This replaces the Streamlit voice recording functionality.
    """
    async def connect(self):
        self.session_id = self.scope['url_route']['kwargs']['session_id']
        self.group_name = f'voice_{self.session_id}'
        
        # Join the voice recording group
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )
        await self.accept()

    async def disconnect(self, close_code):
        # Leave the voice recording group
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )

    async def receive(self, text_data=None, bytes_data=None):
        """
        Handle incoming WebSocket messages for voice recording.
        Can receive both text messages and binary audio data.
        """
        if bytes_data:
            # Handle incoming audio data
            # In a real implementation, this would save the audio chunks
            await self.channel_layer.group_send(
                self.group_name,
                {
                    'type': 'audio_chunk_received',
                    'message': 'Audio chunk received'
                }
            )
        
        elif text_data:
            try:
                data = json.loads(text_data)
                message_type = data.get('type', '')
                
                if message_type == 'start_recording':
                    await self.channel_layer.group_send(
                        self.group_name,
                        {
                            'type': 'recording_started',
                            'message': 'Voice recording started'
                        }
                    )
                
                elif message_type == 'stop_recording':
                    await self.channel_layer.group_send(
                        self.group_name,
                        {
                            'type': 'recording_stopped',
                            'message': 'Voice recording stopped'
                        }
                    )
                    
                    # In a real implementation, this would process the collected audio data
                    
            except Exception as e:
                logger.error(f"Error in VoiceRecordingConsumer: {str(e)}")
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': f'Error: {str(e)}'
                }))

    async def recording_started(self, event):
        await self.send(text_data=json.dumps(event))

    async def recording_stopped(self, event):
        await self.send(text_data=json.dumps(event))

    async def audio_chunk_received(self, event):
        await self.send(text_data=json.dumps(event))


class CodeAssessmentConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for handling real-time coding assessments.
    This replaces the Streamlit code editor and assessment functionality.
    """
    async def connect(self):
        self.assessment_id = self.scope['url_route']['kwargs']['assessment_id']
        self.group_name = f'code_{self.assessment_id}'
        
        # Join the code assessment group
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )
        await self.accept()
        await self.send(text_data=json.dumps({
            'type': 'connection_established',
            'message': 'Connected to code assessment session'
        }))

    async def disconnect(self, close_code):
        # Leave the code assessment group
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        """
        Handle incoming WebSocket messages for code assessment.
        """
        try:
            data = json.loads(text_data)
            message_type = data.get('type', '')
            
            if message_type == 'code_update':
                # Handle code updates
                code = data.get('code', '')
                
                # In a real implementation, this would save the code
                await self.channel_layer.group_send(
                    self.group_name,
                    {
                        'type': 'code_updated',
                        'code': code
                    }
                )
                
            elif message_type == 'run_tests':
                # Handle test execution
                code = data.get('code', '')
                
                # In a real implementation, this would execute the code and tests
                # This would be done in a sandbox environment
                
                await self.channel_layer.group_send(
                    self.group_name,
                    {
                        'type': 'test_results',
                        'results': [
                            {'test_case': 1, 'passed': True, 'output': 'Test passed'},
                            {'test_case': 2, 'passed': True, 'output': 'Test passed'}
                        ]
                    }
                )
                
        except Exception as e:
            logger.error(f"Error in CodeAssessmentConsumer: {str(e)}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Error: {str(e)}'
            }))

    async def code_updated(self, event):
        await self.send(text_data=json.dumps(event))

    async def test_results(self, event):
        await self.send(text_data=json.dumps(event))